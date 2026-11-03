//! Scratch microbench (`SUSHI_DS41_INDEX_DECODE_BENCH=1` with the gpu-test step): what DeepSeek-V4.1's indexer costs
//! a decode round as the context grows. A verify forward of S rows (1..16, the narrow path) runs, per index source:
//! layers 2, 8 and 14 score every ratio-2 group (D/2 keys) and keep the top 512; layer 20 scores every ratio-1 group
//! (D keys), keeps the top 512 and pins the top 2,048 blocks of 8 for the layers after it; layers 24, 28, 32 and 36
//! score only those 16,384 keys. Each stage is timed alone over cycled input sets; the round estimate sums them.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const dsv = @import("sushi/dsv41/dsv41_model.zig");
const qat = @import("sushi/dsv41/dsv41_qat.zig");
const inax = @import("index_nax.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const ci = sc_mod.ci;

fn randn(sc: *Scope, shape: []const c_int, seed: u64, dt: mlx.mlx_dtype) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    const x = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1.5, key, sc.s), &o);
    return sc.astype(x, dt);
}

const IH = 32;
const IHD = 128;
const K = 512;
const KB = 2048;
const BS = 8;
const n_sets = 4;
const rounds = 6;
// unpack_r1: the fp4 store unpacked to bf16 (what the matrix scorer reads); nax_r1: that unpack plus the prefill
// matmul2d scorer at decode width, the alternative to score_r1.
const Stage = enum { score_r2, topk_r2, score_r1, topk_r1, cand_blocks, gather_fixed, unpack_r1, nax_r1 };
const n_stages = 8;

test "sushi_ds41 gpu: index decode-width cost by depth (scratch bench)" {
    if (std.c.getenv("SUSHI_DS41_INDEX_DECODE_BENCH") == null) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
    const depths = [_]usize{ 65536, 131072, 262144, 524288, 1048576 };
    const widths = [_]usize{ 1, 4, 6, 16 };
    for (depths) |D| {
        var sc = Scope.init(alloc, mlx.gpuStream());
        defer sc.deinit();
        // Key stores at ratio 1 (D groups); the ratio-2 stages read the first D/2 of the same store.
        var rows: [n_sets]dsv.Rows = undefined;
        for (0..n_sets) |si| {
            rows[si] = .{ .width = IHD, .dtype = .bfloat16, .pack = .{ .mode = .fp4_ue8m0, .group = 32 } };
            var at: usize = 0;
            while (at < D) : (at += 65536) {
                const n = @min(65536, D - at);
                try rows[si].append(&sc, try qat.qat(&sc, try randn(&sc, &.{ ci(n), IHD }, 0x5eed00 + D + si * 977 + at, .bfloat16), .fp4_ue8m0, 32));
                try mlx.check(mlx.mlx_array_eval(rows[si].buf));
            }
        }
        defer for (&rows) |*rw| rw.deinit();
        for (widths) |S| {
            var rs0 = Scope.init(alloc, sc.s);
            defer rs0.deinit();
            var qs: [n_sets]A = undefined;
            var ws: [n_sets]A = undefined;
            var sc_r2: [n_sets]A = undefined;
            var sc_r1: [n_sets]A = undefined;
            const blk_h = try alloc.alloc(i32, S * KB);
            defer alloc.free(blk_h);
            const stride = D / BS / KB;
            for (0..S) |l| for (0..KB) |b| {
                blk_h[l * KB + b] = @intCast(b * stride);
            };
            const blocks = try rs0.ints(blk_h, &.{ ci(S), KB });
            for (0..n_sets) |si| {
                qs[si] = try rs0.astype(try randn(&rs0, &.{ ci(S), IH, IHD }, 0x9000 + D + S * 31 + si, .bfloat16), .float32);
                ws[si] = try randn(&rs0, &.{ ci(S), IH }, 0xa000 + D + S * 31 + si, .bfloat16);
                sc_r2[si] = try kernels.indexScoresPacked(&rs0, qs[si], try rs0.slice(rows[si].buf, 0, 0, D / 2), try rs0.slice(rows[si].sbuf, 0, 0, D / 2), ws[si], wscale, D - S, 2, .fp4_ue8m0, 32);
                sc_r1[si] = try kernels.indexScoresPacked(&rs0, qs[si], try rs0.slice(rows[si].buf, 0, 0, D), try rs0.slice(rows[si].sbuf, 0, 0, D), ws[si], wscale, D - S, 1, .fp4_ue8m0, 32);
                try mlx.check(mlx.mlx_array_eval(sc_r2[si]));
                try mlx.check(mlx.mlx_array_eval(sc_r1[si]));
            }
            var best: [n_stages]f64 = @splat(std.math.inf(f64));
            var sum: [n_stages]f64 = @splat(0);
            for (0..rounds) |round| for (0..n_stages) |ai_| {
                const st: Stage = @enumFromInt((ai_ + round) % n_stages);
                var rs = Scope.init(alloc, sc.s);
                defer rs.deinit();
                var outs: [n_sets]A = undefined;
                _ = mlx.mlx_synchronize(sc.s);
                const t0 = std.Io.Timestamp.now(io, .awake);
                for (0..n_sets) |si| {
                    const rw = &rows[si];
                    outs[si] = switch (st) {
                        .score_r2 => try kernels.indexScoresPacked(&rs, qs[si], try rs.slice(rw.buf, 0, 0, D / 2), try rs.slice(rw.sbuf, 0, 0, D / 2), ws[si], wscale, D - S, 2, .fp4_ue8m0, 32),
                        .topk_r2 => try kernels.topkIndices(&rs, sc_r2[si], K),
                        .score_r1 => try kernels.indexScoresPacked(&rs, qs[si], try rs.slice(rw.buf, 0, 0, D), try rs.slice(rw.sbuf, 0, 0, D), ws[si], wscale, D - S, 1, .fp4_ue8m0, 32),
                        .topk_r1 => try kernels.topkIndices(&rs, sc_r1[si], K),
                        .cand_blocks => blk: {
                            const bm = try rs.max(try rs.reshape(sc_r1[si], &.{ ci(S), ci(D / BS), BS }), -1, false);
                            break :blk try kernels.topkIndices(&rs, bm, KB);
                        },
                        .unpack_r1 => try rw.rows(&rs, D),
                        .nax_r1 => try inax.indexScoresNax(&rs, qs[si], try rw.rows(&rs, D), ws[si], wscale, D - S, 1, 0),
                        .gather_fixed => blk: {
                            const g = try kernels.indexScoresGather(&rs, qs[si], rw.buf, rw.sbuf, ws[si], wscale, D - S, 1, .fp4_ue8m0, 32, blocks, BS);
                            break :blk try kernels.candidateKeys(&rs, try kernels.topkIndices(&rs, g, K), blocks, BS);
                        },
                    };
                }
                const v = mlx.mlx_vector_array_new_data(&outs, n_sets);
                defer _ = mlx.mlx_vector_array_free(v);
                try mlx.check(mlx.mlx_eval(v));
                _ = mlx.mlx_synchronize(sc.s);
                const ms = @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e6 / n_sets;
                if (round == 0) continue;
                const ai = @intFromEnum(st);
                best[ai] = @min(best[ai], ms);
                sum[ai] += ms;
            };
            const m = struct {
                fn mean(s: [n_stages]f64, i: Stage) f64 {
                    return s[@intFromEnum(i)] / (rounds - 1);
                }
            };
            for (0..n_stages) |ai| std.debug.print("idx-decode D={d} S={d} {s}: best {d:.3} ms, mean {d:.3} ms\n", .{ D, S, @tagName(@as(Stage, @enumFromInt(ai))), best[ai], sum[ai] / (rounds - 1) });
            const full = 3 * (m.mean(sum, .score_r2) + m.mean(sum, .topk_r2)) + m.mean(sum, .score_r1) + m.mean(sum, .topk_r1) + m.mean(sum, .cand_blocks);
            const fixed = 4 * m.mean(sum, .gather_fixed);
            std.debug.print("idx-decode D={d} S={d} ROUND: full scans {d:.3} ms + candidate layers {d:.3} ms = {d:.3} ms a verify forward\n", .{ D, S, full, fixed, full + fixed });
        }
    }
}
