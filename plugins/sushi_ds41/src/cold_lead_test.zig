//! Live Metal GPU + Engram parity tests for `cold_lead.zig` (`extendCold` lead split).
//! Run via `zig build gpu-test -Dmlx-lib=<dir>` (runs the self-contained Engram prefetch + GPU dequant parity test
//! unconditionally on real Metal GPU, plus full-model `DSV41_TINY` parity tests when `-Ddsv41-tiny=<dir>` is passed).
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");
const engram = @import("sushi/dsv41/dsv41_engram.zig");
const cfg_mod = @import("sushi/dsv41/dsv41_config.zig");
const cold_lead = @import("cold_lead.zig");

const testing = std.testing;
const Scope = sc_mod.Scope;
const A = sc_mod.A;

test {
    _ = cold_lead;
}

fn skipNoGpu() !void {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    try mlx.requireGpu();
}

/// Writes a 12-row, head_dim-64 FP8 E4M3 + UE8M0 Engram shard for layer 1 with distinct values per row/col.
fn writeEngramShard(io: std.Io, dir: std.Io.Dir) !void {
    const rows: usize = 12;
    const hd: usize = 64;
    const sdim: usize = hd / 32;
    const hdr = "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[12,64],\"data_offsets\":[0,768]}," ++
        "\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[12,2],\"data_offsets\":[768,792]}}";
    var shard: [8 + hdr.len + rows * hd + rows * sdim]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    const data = shard[8 + hdr.len ..];
    const codes = [_]u8{ 0x38, 0x3c, 0x40, 0x42, 0x44, 0x48, 0xb8, 0xbc }; // +/-1.0, 1.5, 2.0, 2.5, 3.0, 4.0
    for (0..rows) |r| {
        for (0..hd) |c| {
            data[r * hd + c] = codes[(r * 7 + c * 3) % codes.len];
        }
        data[rows * hd + r * sdim] = @intCast(125 + (r % 5));
        data[rows * hd + r * sdim + 1] = @intCast(126 + ((r * 3) % 4));
    }
    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = &shard });
}

fn hashSlice(hs: *const engram.Hash, hist: []const u32, tokens: []const u32, alloc: std.mem.Allocator) ![]u32 {
    const out = try alloc.alloc(u32, tokens.len * hs.n_layers * hs.cols());
    hs.rows(hist, tokens, out);
    return out;
}

fn gpuBytesOf(sc: *Scope, a: A) ![]u8 {
    defer _ = mlx.mlx_array_free(a);
    const vals = try sc_mod.hostF32(try sc.astype(a, .float32));
    try testing.expect(vals.len > 0);
    for (vals) |v| try testing.expect(std.math.isFinite(v));
    return testing.allocator.dupe(u8, std.mem.sliceAsBytes(vals));
}

test "sushi_ds41 gpu: extendCold schedule drives Engram ahead prefetch + GPU dequant bit-for-bit across split chunk 0 and chunks 1..N" {
    try skipNoGpu();
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    var sc = Scope.plain(testing.allocator, s);
    defer sc.deinit();

    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeEngramShard(io, td.dir);
    var path_buf: [512]u8 = undefined;
    const dir_path = path_buf[0..try td.dir.realPath(io, &path_buf)];

    const e_cfg: cfg_mod.Engram = .{
        .n = 1,
        .layers = .{ 1, 0, 0, 0 },
        .rows = .{ 12, 0, 0, 0 },
        .max_ngram = 2,
        .vocab_size = 5,
        .heads = 2,
        .head_dim = 64,
        .pad_id = 2,
        .compressed_vocab = 100,
    };
    const hs = try engram.Hash.init(e_cfg, 2);
    var table = try engram.Table.open(testing.allocator, dir_path, 1, 12, 64);
    defer table.close();
    const tables = [_]*const engram.Table{&table};

    // Single-chunk cold prompt (8192 prefix + 1 last token = 8193) -> 2 steps (4096 lead + 4096 rest), all <= 8192.
    const ids = try testing.allocator.alloc(u32, 8193);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast((i * 13 + 3) % 97);

    // Ground truth unsplit Engram rows across the entire prompt.
    const all_row_ids = try hashSlice(&hs, &.{}, ids, testing.allocator);
    defer testing.allocator.free(all_row_ids);
    const nc = hs.cols();

    var iter: cold_lead.Iter = .{
        .prefix = ids.len - 1,
        .total_len = ids.len,
        .state_n = 0,
        .window = 128,
        .has_ced_split = true,
        .has_engrams = true,
        .has_ahead = false,
    };
    var ahead_pf: ?*engram.Prefetch = null;
    defer if (ahead_pf) |pf| pf.finish();

    var step_idx: usize = 0;
    while (iter.next()) |step| : (step_idx += 1) {
        try testing.expect(step.end - step.at <= cold_lead.MAX_CHUNK_ROWS);
        const cur_rows = all_row_ids[step.at * nc .. step.end * nc];
        const got_bytes = blk: {
            if (ahead_pf) |pf| {
                // Rest of chunk 0 must hit the lead sub-chunk's ahead prefetch!
                try testing.expect(pf.matches(step.at, cur_rows));
                const raw = try pf.take(step.at, 0, 0);
                defer raw.deinit(testing.allocator);
                pf.finish();
                ahead_pf = null;
                break :blk try gpuBytesOf(&sc, try table.dequant(raw, cur_rows.len, s));
            } else {
                try testing.expectEqual(@as(usize, 0), step_idx);
                try testing.expect(step.cold_lead);
                break :blk try gpuBytesOf(&sc, try table.gather(testing.allocator, cur_rows, s));
            }
        };
        defer testing.allocator.free(got_bytes);
        const want_bytes = try gpuBytesOf(&sc, try table.gather(testing.allocator, cur_rows, s));
        defer testing.allocator.free(want_bytes);
        try testing.expectEqualSlices(u8, want_bytes, got_bytes);

        // Start ahead prefetch for [step.end .. step.next), as aheadEngram does.
        const ahead_len = step.next - step.end;
        const ahead_rows = try hashSlice(&hs, ids[0..step.end], ids[step.end..step.next], testing.allocator);
        defer testing.allocator.free(ahead_rows);
        ahead_pf = try engram.Prefetch.start(testing.allocator, &tables, &.{.{ .at = step.end, .len = ahead_len, .ids = ahead_rows }}, nc, ahead_len);
    }
    try testing.expectEqual(@as(usize, 2), step_idx);
    // Final token (8192..8193) also adopts ahead_pf from the rest of chunk 0.
    try testing.expect(ahead_pf != null);
    const last_rows = all_row_ids[(ids.len - 1) * nc .. ids.len * nc];
    try testing.expect(ahead_pf.?.matches(ids.len - 1, last_rows));
    const last_raw = try ahead_pf.?.take(ids.len - 1, 0, 0);
    defer last_raw.deinit(testing.allocator);
    const got_last = try gpuBytesOf(&sc, try table.dequant(last_raw, last_rows.len, s));
    defer testing.allocator.free(got_last);
    const want_last = try gpuBytesOf(&sc, try table.gather(testing.allocator, last_rows, s));
    defer testing.allocator.free(want_last);
    try testing.expectEqualSlices(u8, want_last, got_last);

    // Non-trigger cases must never emit cold_lead and must keep every chunk <= 8192.
    for ([_]cold_lead.Iter{
        .{ .prefix = 2047, .total_len = 2048, .state_n = 0, .window = 128, .has_ced_split = true, .has_engrams = true, .has_ahead = false },
        .{ .prefix = 8192, .total_len = 8193, .state_n = 16384, .window = 128, .has_ced_split = true, .has_engrams = true, .has_ahead = false },
        .{ .prefix = 32768, .total_len = 32769, .state_n = 0, .window = 128, .has_ced_split = true, .has_engrams = true, .has_ahead = false },
        .{ .prefix = 8192, .total_len = 8193, .state_n = 0, .window = 128, .has_ced_split = true, .has_engrams = true, .has_ahead = true },
        .{ .prefix = 8192, .total_len = 8193, .state_n = 0, .window = 128, .has_ced_split = false, .has_engrams = true, .has_ahead = false },
        .{ .prefix = 8192, .total_len = 8193, .state_n = 0, .window = 128, .has_ced_split = true, .has_engrams = false, .has_ahead = false },
    }) |case_iter| {
        var ci_ = case_iter;
        while (ci_.next()) |st| {
            try testing.expect(!st.cold_lead);
            try testing.expect(st.end - st.at <= cold_lead.MAX_CHUNK_ROWS);
        }
    }
}

const TinyRun = struct {
    logits: []u8,
    slots: []dsv.Slot,
    state_only_used: u64,
    engram_ahead_used: u64,

    fn deinit(self: TinyRun) void {
        testing.allocator.free(self.logits);
        for (self.slots) |*sl| dsv.freeSlot(sl);
        testing.allocator.free(self.slots);
    }
};

fn runTinySchedule(
    m: *dsv.Model,
    sc: *Scope,
    ids: []const u32,
    vision: ?dsv.VisionRows,
    plan: cold_lead.Plan,
    force_extend_ahead: bool,
) !TinyRun {
    const st = try dsv.ensureState(m);
    const prefix = ids.len - 1;
    var rows = vision;
    const tapping = m.stream_tap != null;
    if (m.stream_tap) |t| t.clear();
    defer if (m.stream_tap) |t| {
        t.on = false;
    };
    const state_used0 = m.state_only_used;
    const ahead_used0 = m.engram_ahead_used;
    var iter: cold_lead.Iter = .{
        .prefix = prefix,
        .total_len = ids.len,
        .state_n = st.n,
        .window = m.cfg.window,
        .has_ced_split = m.ced_split != null,
        .has_engrams = m.engrams.len > 0,
        .has_ahead = m.engram_ahead != null,
        .plan = plan,
    };
    while (iter.next()) |step| {
        if (tapping) m.stream_tap.?.on = step.tap_on;
        if (step.state_only and !force_extend_ahead) {
            try dsv.extendState(m, ids[step.at..step.end], dsv.SPAN, rows, ids[step.end..step.next]);
        } else {
            _ = mlx.mlx_array_free(try dsv.extendAhead(m, ids[step.at..step.end], .last, dsv.SPAN, rows, ids[step.end..step.next]));
        }
        if (rows) |*r| r.offset += std.mem.count(u32, ids[step.at..step.end], &.{m.cfg.image_token_id});
    }
    try dsv.pipeDrain(m);
    if (tapping) m.stream_tap.?.on = true;
    const lg = try dsv.extendAhead(m, ids[prefix..], .last, dsv.SPAN, rows, &.{});
    const logits = try gpuBytesOf(sc, lg);
    errdefer testing.allocator.free(logits);
    const slots = try testing.allocator.alloc(dsv.Slot, dsv.slotCount(m));
    @memset(slots, dsv.nil_slot);
    try dsv.captureState(m, slots);
    return .{
        .logits = logits,
        .slots = slots,
        .state_only_used = m.state_only_used - state_used0,
        .engram_ahead_used = m.engram_ahead_used - ahead_used0,
    };
}

fn expectSlotsLive(sc: *Scope, m: *const dsv.Model, want: []const dsv.Slot, got: []const dsv.Slot, n: usize) !void {
    const split = m.ced_split.?;
    for (want, got, 0..) |x, y, i| {
        if (i % 2 != 0 or i >= 2 * m.cfg.layers or i / 2 < split) {
            try dsv.expectSlotsBits(sc, want[i..][0..1], got[i..][0..1]);
            continue;
        }
        const cap = mlx.getShape(x[0])[0];
        const hx = try dsv.hostRows(sc, x[0]);
        defer testing.allocator.free(hx);
        const hy = try dsv.hostRows(sc, y[0]);
        defer testing.allocator.free(hy);
        const w = @as(usize, @intCast(mlx.getShape(x[0])[1]));
        const ucap: usize = @intCast(cap);
        for (n - m.cfg.window..n) |p| {
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(hx[p % ucap * w ..][0..w]), std.mem.sliceAsBytes(hy[p % ucap * w ..][0..w]));
        }
        try dsv.expectSlotsBits(sc, &.{.{ .{}, x[1], x[2], x[3] }}, &.{.{ .{}, y[1], y[2], y[3] }});
    }
}

fn argmaxBytes(bytes: []const u8) usize {
    const floats: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, bytes));
    var best: usize = 0;
    for (floats, 0..) |v, i| if (v > floats[best]) {
        best = i;
    };
    return best;
}

test "sushi_ds41 gpu: extendCold split produces identical state and greedy logits vs unsplit on single-chunk text and vision prompts (DSV41_TINY)" {
    try skipNoGpu();
    const dir = try dsv.fixtureDir();
    const s = mlx.gpuStream();
    var sc = Scope.plain(testing.allocator, s);
    defer sc.deinit();

    const m = try dsv.tinyModel(dir, .{ .act = .bfloat16, .qat = true, .dspark = true }, s);
    defer m.deinit();
    try testing.expect(m.ced_split != null and m.engrams.len > 0);
    // Tiny model has window = 8; use a scaled plan that stays within its 128-position RoPE table.
    const plan: cold_lead.Plan = .{ .chunk_width = 48, .lead_min = 24, .lead_align = 8, .tail_merge_max = 8 };
    const unsplit_plan: cold_lead.Plan = .{ .chunk_width = 48, .lead_min = std.math.maxInt(usize), .lead_align = 8, .tail_merge_max = 8 };

    var ids: [97]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = if (i == 0) 2 else 3 + @as(u32, @intCast((i * 37) % 90));
    const single = ids[0..49]; // prefix = 48 -> split [0..24] lead + [24..48] rest + [48..49] last

    // 1. Plain text + Engram single-chunk parity (extendCold vs same-cut extendAhead bit-for-bit, and vs unsplit greedy).
    {
        dsv.resetState(m);
        const split_run = try runTinySchedule(m, &sc, single, null, plan, false);
        defer split_run.deinit();
        try testing.expectEqual(@as(u64, 1), split_run.state_only_used);
        try testing.expectEqual(@as(u64, 2), split_run.engram_ahead_used);

        dsv.resetState(m);
        const same_cut_full = try runTinySchedule(m, &sc, single, null, plan, true);
        defer same_cut_full.deinit();
        try testing.expectEqualSlices(u8, same_cut_full.logits, split_run.logits);
        try expectSlotsLive(&sc, m, same_cut_full.slots, split_run.slots, single.len);

        dsv.resetState(m);
        const unsplit_run = try runTinySchedule(m, &sc, single, null, unsplit_plan, false);
        defer unsplit_run.deinit();
        try testing.expectEqual(argmaxBytes(unsplit_run.logits), argmaxBytes(split_run.logits));
    }

    // 2. Vision tokens straddling the lead split boundary (0..24 and 24..48).
    {
        const image_id: u32 = m.cfg.vocab - 1;
        m.cfg.image_token_id = image_id;
        for (m.layers[0..m.cfg.layers]) |*ly| {
            const bias = try sc_mod.hostF32(ly.gate_bias);
            const shifted = try testing.allocator.alloc(f32, bias.len);
            defer testing.allocator.free(shifted);
            for (bias, shifted, 0..) |b, *o, e| o.* = b + (if (e % 2 == 0) @as(f32, 6) else -6);
            ly.gate_bias_vl = try sc.keep(mlx.mlx_array_new_data(shifted.ptr, &[_]c_int{ 1, sc_mod.ci(bias.len) }, 2, .float32));
        }
        const vids = try testing.allocator.dupe(u32, single);
        defer testing.allocator.free(vids);
        // Span crosses lead=24 (indices 20..28).
        for (vids[20..28]) |*t| t.* = image_id;

        var key = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&key, 0x1a6e));
        _ = try sc.keep(key);
        var vraw = mlx.mlx_array_new();
        const vrows = try sc.res(mlx.mlx_random_normal(&vraw, &.{ 8, sc_mod.ci(m.cfg.dim) }, 2, .float32, 0, 1, key, sc.s), &vraw);
        const vr: dsv.VisionRows = .{ .rows = vrows, .offset = 0 };

        dsv.resetState(m);
        const split_v = try runTinySchedule(m, &sc, vids, vr, plan, false);
        defer split_v.deinit();
        try testing.expectEqual(@as(u64, 1), split_v.state_only_used);
        try testing.expectEqual(@as(u64, 2), split_v.engram_ahead_used);

        dsv.resetState(m);
        const same_cut_v = try runTinySchedule(m, &sc, vids, vr, plan, true);
        defer same_cut_v.deinit();
        try testing.expectEqualSlices(u8, same_cut_v.logits, split_v.logits);
        try expectSlotsLive(&sc, m, same_cut_v.slots, split_v.slots, vids.len);

        dsv.resetState(m);
        const unsplit_v = try runTinySchedule(m, &sc, vids, vr, unsplit_plan, false);
        defer unsplit_v.deinit();
        try testing.expectEqual(argmaxBytes(unsplit_v.logits), argmaxBytes(split_v.logits));
    }

    // 3. Non-trigger cases on live model: short prompt, multi-chunk prompt, cached tail resume, no CED.
    {
        // Short prompt (< lead_min)
        dsv.resetState(m);
        const short_run = try runTinySchedule(m, &sc, ids[0..21], null, plan, false);
        defer short_run.deinit();
        try testing.expectEqual(@as(u64, 0), short_run.state_only_used);

        // Cached tail resume (st.n = 21, tail prefix = 40 <= chunk_width = 48)
        const tail_run = try runTinySchedule(m, &sc, ids[21..62], null, plan, false);
        defer tail_run.deinit();
        try testing.expectEqual(@as(u64, 0), tail_run.state_only_used);

        // Multi-chunk prompt (prefix = 96 > chunk_width = 48): only normal chunk 0 (0..48) uses state_only, no lead split.
        dsv.resetState(m);
        const multi_run = try runTinySchedule(m, &sc, &ids, null, plan, false);
        defer multi_run.deinit();
        try testing.expectEqual(@as(u64, 1), multi_run.state_only_used);

        // ced_split == null
        const saved_split = m.ced_split;
        m.ced_split = null;
        defer m.ced_split = saved_split;
        dsv.resetState(m);
        const no_ced = try runTinySchedule(m, &sc, single, null, plan, false);
        defer no_ced.deinit();
        try testing.expectEqual(@as(u64, 0), no_ced.state_only_used);
    }
}
