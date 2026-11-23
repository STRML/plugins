//! `extendCold` prefill lead split: when a cold prompt starts without an existing Engram read-ahead, split chunk 0
//! into an encoder-only leading sub-chunk (`extendState`) while passing the rest of chunk 0 as `ahead`, overlapping
//! cold Engram SSD reads with GPU compute of the lead sub-chunk.
const std = @import("std");
const prefill_plan = @import("sushi/prefill_plan.zig");

/// Minimum chunk length for `extendCold` lead splitting (matches `mlx-serve` `LEAD_MIN`).
pub const LEAD_MIN: usize = 4096;

/// Default attention span alignment (`dsv41_model.SPAN`).
pub const DEFAULT_ALIGN: usize = 256;

/// Nominal widest prefill chunk (`dsv41_config.prefillWidthFor("")`).
pub const MAX_CHUNK_ROWS: usize = 8192;

pub const Plan = struct {
    chunk_width: usize = MAX_CHUNK_ROWS,
    lead_min: usize = LEAD_MIN,
    lead_align: usize = DEFAULT_ALIGN,
    tail_merge_max: usize = prefill_plan.TAIL_MERGE_MAX,
};

pub const LeadArgs = struct {
    at: usize,
    state_n: usize,
    chunk_end: usize,
    prefix: usize,
    chunk_width: usize,
    window: usize,
    lead_min: usize,
    lead_align: usize,
    has_ced_split: bool,
    has_engrams: bool,
    has_ahead: bool,
};

pub const Step = struct {
    at: usize,
    end: usize,
    next: usize,
    state_only: bool,
    cold_lead: bool,
    tap_on: bool,
};

/// End offset of the chunk starting at `at` within `0..prefix`, merging a short tail below `tail_merge_max`
/// without ever exceeding `MAX_CHUNK_ROWS` (8,192).
pub fn chunkEnd(at: usize, prefix: usize, chunk_width: usize, tail_merge_max: usize) usize {
    const capped = @min(chunk_width, MAX_CHUNK_ROWS);
    var end = @min(at + capped, prefix);
    if (end < prefix and prefix - end < tail_merge_max and prefix - at <= MAX_CHUNK_ROWS) end = prefix;
    return end;
}

/// Rows for the leading encoder-only sub-chunk of chunk 0 (`0` when chunk 0 should run whole).
/// Mirrors `mlx-serve`'s `leadRows` (`L / 2` aligned to `SPAN`) and skips short prompts, cached tail resumes of at
/// most one chunk, already-prefetched chunks, models without CED or Engram, and chunks too narrow to leave a
/// replayable window in the rest of chunk 0.
pub fn coldLeadRows(args: LeadArgs) usize {
    if (args.at != 0 or args.has_ahead or !args.has_ced_split or !args.has_engrams) return 0;
    // A cached tail resume of <= 1 chunk stays whole; multi-chunk resumes still overlap chunk 0's cold Engram reads.
    if (args.state_n > 0 and args.prefix <= args.chunk_width) return 0;
    if (args.chunk_end <= args.at) return 0;
    const len = args.chunk_end - args.at;
    if (len < args.lead_min) return 0;
    const align_to = @max(args.lead_align, 1);
    const lead = (len / 2 / align_to) * align_to;
    if (lead <= args.window) return 0;
    if (args.chunk_end - lead <= args.window) return 0;
    if (args.prefix <= lead + args.window) return 0;
    return lead;
}

/// Chunk schedule iterator for `plugin.zig`'s `run()`.
pub const Iter = struct {
    prefix: usize,
    total_len: usize,
    state_n: usize,
    window: usize,
    has_ced_split: bool,
    has_engrams: bool,
    has_ahead: bool,
    plan: Plan = .{},
    at: usize = 0,
    cold_rest_end: ?usize = null,

    pub fn next(self: *Iter) ?Step {
        if (self.at >= self.prefix) return null;
        const at = self.at;
        var end: usize = undefined;
        var cold_lead = false;
        if (self.cold_rest_end) |rest_end| {
            end = rest_end;
            self.cold_rest_end = null;
        } else {
            end = chunkEnd(at, self.prefix, self.plan.chunk_width, self.plan.tail_merge_max);
            if (at == 0) {
                const lead = coldLeadRows(.{
                    .at = at,
                    .state_n = self.state_n,
                    .chunk_end = end,
                    .prefix = self.prefix,
                    .chunk_width = self.plan.chunk_width,
                    .window = self.window,
                    .lead_min = self.plan.lead_min,
                    .lead_align = self.plan.lead_align,
                    .has_ced_split = self.has_ced_split,
                    .has_engrams = self.has_engrams,
                    .has_ahead = self.has_ahead,
                });
                if (lead > 0) {
                    self.cold_rest_end = end;
                    end = lead;
                    cold_lead = true;
                }
            }
        }
        const ahead_end = if (self.cold_rest_end) |rest_end|
            rest_end
        else if (end < self.prefix)
            chunkEnd(end, self.prefix, self.plan.chunk_width, self.plan.tail_merge_max)
        else
            self.total_len;
        const state_only = cold_lead or
            (end < self.prefix and
            end - at > self.window and
            self.prefix - end > self.window and
            self.has_ced_split);
        self.at = end;
        return .{
            .at = at,
            .end = end,
            .next = ahead_end,
            .state_only = state_only,
            .cold_lead = cold_lead,
            .tap_on = end == self.prefix,
        };
    }
};

fn collectSteps(alloc: std.mem.Allocator, iter_init: Iter) ![]Step {
    var iter = iter_init;
    var list: std.ArrayList(Step) = .empty;
    errdefer list.deinit(alloc);
    while (iter.next()) |step| try list.append(alloc, step);
    return list.toOwnedSlice(alloc);
}

test "sushi_ds41 extendCold: splits cold 32k chunk 0 into 4096 lead + 4096 rest and preserves chunks 1..N <= 8192" {
    const t = std.testing;
    const steps = try collectSteps(t.allocator, .{
        .prefix = 32768,
        .total_len = 32769,
        .state_n = 0,
        .window = 128,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    });
    defer t.allocator.free(steps);
    try t.expectEqual(@as(usize, 5), steps.len);
    // Lead sub-chunk: 0..4096 encoder-only, ahead = 4096..8192 (rest of chunk 0).
    try t.expectEqual(Step{ .at = 0, .end = 4096, .next = 8192, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
    // Rest of chunk 0: 4096..8192 encoder-only (since chunks 1..3 follow), ahead = 8192..16384.
    try t.expectEqual(Step{ .at = 4096, .end = 8192, .next = 16384, .state_only = true, .cold_lead = false, .tap_on = false }, steps[1]);
    try t.expectEqual(Step{ .at = 8192, .end = 16384, .next = 24576, .state_only = true, .cold_lead = false, .tap_on = false }, steps[2]);
    try t.expectEqual(Step{ .at = 16384, .end = 24576, .next = 32768, .state_only = true, .cold_lead = false, .tap_on = false }, steps[3]);
    // Last prefix chunk runs full CED replay + tap, prefetching final token 32768..32769.
    try t.expectEqual(Step{ .at = 24576, .end = 32768, .next = 32769, .state_only = false, .cold_lead = false, .tap_on = true }, steps[4]);
    for (steps) |s| try t.expect(s.end - s.at <= MAX_CHUNK_ROWS);
}

test "sushi_ds41 extendCold: splits cold 4096..8500 single/tail prompts with every chunk <= 8192" {
    const t = std.testing;
    // 4096 prefix -> 2048 lead + 2048 rest
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 4096,
            .total_len = 4097,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 2048, .next = 4096, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 2048, .end = 4096, .next = 4097, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 8500 prefix -> never merges past 8192: 4096 lead + 4096 rest + 308 tail, all <= 8192
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8500,
            .total_len = 8501,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 3), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 4096, .next = 8192, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 4096, .end = 8192, .next = 8500, .state_only = true, .cold_lead = false, .tap_on = false }, steps[1]);
        try t.expectEqual(Step{ .at = 8192, .end = 8500, .next = 8501, .state_only = false, .cold_lead = false, .tap_on = true }, steps[2]);
        for (steps) |s| try t.expect(s.end - s.at <= MAX_CHUNK_ROWS);
    }
    // 16500 prefix -> never emits chunk > 8192; every ahead covers the exact next chunk.
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 16500,
            .total_len = 16501,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 4), steps.len);
        try t.expectEqual(@as(usize, 8192), steps[0].next);
        try t.expectEqual(@as(usize, 16384), steps[1].next);
        try t.expectEqual(@as(usize, 16500), steps[2].next);
        try t.expectEqual(@as(usize, 16500), steps[3].end);
        for (steps) |s| try t.expect(s.end - s.at <= MAX_CHUNK_ROWS);
    }
}

test "sushi_ds41 extendCold: skips short prompts, cached <= 1 chunk resumes, prefetched prompts, no-CED, and no-Engram" {
    const t = std.testing;
    // Short prompt (< LEAD_MIN)
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 4095,
            .total_len = 4096,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 1), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(!steps[0].state_only);
        try t.expect(steps[0].tap_on);
    }
    // Cached tail resume <= 1 chunk (state_n > 0, prefix <= chunk_width)
    for ([_]usize{ 512, 4096, 8192 }) |p| {
        const steps = try collectSteps(t.allocator, .{
            .prefix = p,
            .total_len = p + 1,
            .state_n = 16384,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 1), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(!steps[0].state_only);
    }
    // Cached tail resume > 1 chunk (still benefits from cold lead when not prefetched)
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 16384,
            .total_len = 16385,
            .state_n = 8192,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expect(steps[0].cold_lead);
    }
    // Already prefetched
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 16384,
            .total_len = 16385,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = true,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(steps[0].state_only);
    }
    // ced_split == null
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 16384,
            .total_len = 16385,
            .state_n = 0,
            .window = 128,
            .has_ced_split = false,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(!steps[0].state_only);
    }
    // engrams.len == 0
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 16384,
            .total_len = 16385,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = false,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(steps[0].state_only);
    }
    // prefix <= lead + window (huge window guard)
    try t.expectEqual(@as(usize, 0), coldLeadRows(.{
        .at = 0,
        .state_n = 0,
        .chunk_end = 4096,
        .prefix = 4096,
        .chunk_width = 8192,
        .window = 2048,
        .lead_min = 4096,
        .lead_align = 256,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    }));
    // at > 0 never splits
    try t.expectEqual(@as(usize, 0), coldLeadRows(.{
        .at = 8192,
        .state_n = 0,
        .chunk_end = 16384,
        .prefix = 16384,
        .chunk_width = 8192,
        .window = 128,
        .lead_min = 4096,
        .lead_align = 256,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    }));
}
