//! `extendCold` prefill lead split: when a single-chunk cold prompt starts without an existing Engram read-ahead,
//! split chunk 0 into an encoder-only leading sub-chunk (`extendState`) while passing the rest of chunk 0 as `ahead`,
//! overlapping cold Engram SSD reads with GPU compute of the lead sub-chunk.
const std = @import("std");
const prefill_plan = @import("sushi/prefill_plan.zig");

/// Minimum chunk length for `extendCold` lead splitting (covers 2k..8k single-chunk cold prompts).
pub const LEAD_MIN: usize = 2048;

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
/// Applies to single-chunk cold prompts (`prefix <= chunk_width` and `state_n == 0`) with `L / 2` aligned to `SPAN`;
/// skips multi-chunk prompts (where chunk 0 already runs encoder-only), short prompts (`< lead_min`), cached tail
/// resumes, already-prefetched chunks, models without CED or Engram, and chunks too narrow to leave a replayable
/// window in the rest of chunk 0.
pub fn coldLeadRows(args: LeadArgs) usize {
    if (args.at != 0 or args.state_n > 0 or args.has_ahead or !args.has_ced_split or !args.has_engrams) return 0;
    if (args.prefix > args.chunk_width) return 0;
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

test "sushi_ds41 extendCold: splits cold single-chunk prompts (2048..8192) and keeps multi-chunk prompts unsplit <= 8192" {
    const t = std.testing;
    // 2048 prefix -> 1024 lead + 1024 rest
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 2048,
            .total_len = 2049,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 1024, .next = 2048, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 1024, .end = 2048, .next = 2049, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 4095 prefix (prompt = 4096) -> 1792 lead + 2303 rest
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
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 1792, .next = 4095, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 1792, .end = 4095, .next = 4096, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 4681 prefix (prompt = 4682) -> 2304 lead + 2377 rest
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 4681,
            .total_len = 4682,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 2304, .next = 4681, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 2304, .end = 4681, .next = 4682, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 8192 prefix -> 4096 lead + 4096 rest
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8192,
            .total_len = 8193,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 4096, .next = 8192, .state_only = true, .cold_lead = true, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 4096, .end = 8192, .next = 8193, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 8193 prefix (multi-chunk) -> does NOT split
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8193,
            .total_len = 8194,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 2), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 8192, .next = 8193, .state_only = false, .cold_lead = false, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 8192, .end = 8193, .next = 8194, .state_only = false, .cold_lead = false, .tap_on = true }, steps[1]);
    }
    // 32768 prefix (multi-chunk) -> does NOT split; stays 4 standard 8192-row chunks
    {
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
        try t.expectEqual(@as(usize, 4), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 8192, .next = 16384, .state_only = true, .cold_lead = false, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 8192, .end = 16384, .next = 24576, .state_only = true, .cold_lead = false, .tap_on = false }, steps[1]);
        try t.expectEqual(Step{ .at = 16384, .end = 24576, .next = 32768, .state_only = true, .cold_lead = false, .tap_on = false }, steps[2]);
        try t.expectEqual(Step{ .at = 24576, .end = 32768, .next = 32769, .state_only = false, .cold_lead = false, .tap_on = true }, steps[3]);
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
        try t.expectEqual(@as(usize, 3), steps.len);
        try t.expectEqual(Step{ .at = 0, .end = 8192, .next = 16384, .state_only = true, .cold_lead = false, .tap_on = false }, steps[0]);
        try t.expectEqual(Step{ .at = 8192, .end = 16384, .next = 16500, .state_only = false, .cold_lead = false, .tap_on = false }, steps[1]);
        try t.expectEqual(Step{ .at = 16384, .end = 16500, .next = 16501, .state_only = false, .cold_lead = false, .tap_on = true }, steps[2]);
        for (steps) |s| try t.expect(s.end - s.at <= MAX_CHUNK_ROWS);
    }
}

test "sushi_ds41 extendCold: skips short prompts, cached tail resumes, prefetched prompts, no-CED, and no-Engram" {
    const t = std.testing;
    // Short prompt (< LEAD_MIN)
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 2047,
            .total_len = 2048,
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
    // Cached tail resumes (state_n > 0) never split
    for ([_]usize{ 512, 4096, 8192, 16384 }) |p| {
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
        for (steps) |s| try t.expect(!s.cold_lead);
    }
    // Already prefetched
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8192,
            .total_len = 8193,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = true,
            .has_ahead = true,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 1), steps.len);
        try t.expect(!steps[0].cold_lead);
    }
    // ced_split == null
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8192,
            .total_len = 8193,
            .state_n = 0,
            .window = 128,
            .has_ced_split = false,
            .has_engrams = true,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 1), steps.len);
        try t.expect(!steps[0].cold_lead);
        try t.expect(!steps[0].state_only);
    }
    // engrams.len == 0
    {
        const steps = try collectSteps(t.allocator, .{
            .prefix = 8192,
            .total_len = 8193,
            .state_n = 0,
            .window = 128,
            .has_ced_split = true,
            .has_engrams = false,
            .has_ahead = false,
        });
        defer t.allocator.free(steps);
        try t.expectEqual(@as(usize, 1), steps.len);
        try t.expect(!steps[0].cold_lead);
    }
    // prefix <= lead + window (huge window guard)
    try t.expectEqual(@as(usize, 0), coldLeadRows(.{
        .at = 0,
        .state_n = 0,
        .chunk_end = 4096,
        .prefix = 4096,
        .chunk_width = 8192,
        .window = 2048,
        .lead_min = 2048,
        .lead_align = 256,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    }));
    // at > 0 never splits
    try t.expectEqual(@as(usize, 0), coldLeadRows(.{
        .at = 4096,
        .state_n = 0,
        .chunk_end = 8192,
        .prefix = 8192,
        .chunk_width = 8192,
        .window = 128,
        .lead_min = 2048,
        .lead_align = 256,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    }));
}
