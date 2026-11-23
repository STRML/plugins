//! The Metal tests of `index_nax.zig` (`indexScoresNax`, `idx_nax_bmax`, `idx_nax_gather`, `dec_topk`, and
//! `indexerOnlineTopk`). The root of `zig build gpu-test -Dmlx-lib=<dir>` only (it links that MLX and sets
//! SUSHI_DS41_GPU_TEST); skipped on a virtual GPU or without an M5. Also runs `vision_key.zig`, `ablation_test.zig`,
//! and `prefill_pipe_test.zig` (which needs `-Ddsv41-tiny=<dir>`).
const std = @import("std");
pub const mlx = @import("sushi/mlx.zig");
pub const log = @import("sushi/log.zig");
pub const io_util = @import("sushi/io_util.zig");
const mlx_serve = @import("mlx_serve");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const qat_mod = @import("sushi/dsv41/dsv41_qat.zig");
const cfg_mod = @import("sushi/dsv41/dsv41_config.zig");
const inax = @import("index_nax.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const ci = sc_mod.ci;
const none = sc_mod.none;
const testing = std.testing;

test {
    _ = @import("vision_key.zig");
    _ = @import("prefill_pipe_test.zig");
    _ = @import("ablation_test.zig");
}

fn requireM5Gpu() !void {
    if (std.c.getenv("SUSHI_DS41_GPU_TEST") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend() or mlx_serve.mlx.virtualGpu()) return error.SkipZigTest;
    if (!inax.fits(32, 128, .bfloat16)) return error.SkipZigTest;
}

fn randn(sc: *Scope, shape: []const c_int, seed: u64, dt: mlx.mlx_dtype) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    const x = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1.5, key, sc.s), &o);
    return sc.astype(x, dt);
}

fn fp4Qat(sc: *Scope, x: A) !A {
    return qat_mod.qat(sc, x, .fp4_ue8m0, 32);
}

/// Builds `nb` FP4-quantized keys of `ihd` channels, optionally repeating `distinct` base rows to induce ties across
/// keys and candidate blocks.
fn makeFp4Keys(sc: *Scope, nb: usize, ihd: usize, distinct: usize, seed: u64, dt: mlx.mlx_dtype) !struct { packed_k: qat_mod.Packed, unpacked: A } {
    const base_n = @min(nb, distinct);
    const raw = try fp4Qat(sc, try randn(sc, &.{ ci(base_n), ci(ihd) }, seed, dt));
    const pk_base = try qat_mod.packFp4(sc, raw, .fp4_ue8m0, 32);
    if (base_n == nb) {
        return .{ .packed_k = pk_base, .unpacked = try qat_mod.unpackFp4(sc, pk_base, .fp4_ue8m0, 32, dt) };
    }
    const reps = (nb + base_n - 1) / base_n;
    const codes_list = try sc.a.alloc(A, reps);
    defer sc.a.free(codes_list);
    const scales_list = try sc.a.alloc(A, reps);
    defer sc.a.free(scales_list);
    var done: usize = 0;
    var i: usize = 0;
    while (done < nb) : (i += 1) {
        const take = @min(nb - done, base_n);
        codes_list[i] = try sc.slice(pk_base.codes, 0, 0, take);
        scales_list[i] = try sc.slice(pk_base.scales, 0, 0, take);
        done += take;
    }
    const pk = qat_mod.Packed{
        .codes = try sc.concat(codes_list[0..i], 0),
        .scales = try sc.concat(scales_list[0..i], 0),
    };
    return .{ .packed_k = pk, .unpacked = try qat_mod.unpackFp4(sc, pk, .fp4_ue8m0, 32, dt) };
}

fn maskScores(sc: *Scope, scores: A, lens: A) !A {
    const nb = sc_mod.dim(scores, 1);
    const idx = try kernels.topkIndices(sc, scores, @min(512, nb));
    return sc.where(try sc.lt(idx, lens), idx, try sc.i(-1));
}

/// CPU reference candidate block ids and masked post-candidate scores from full `scores` `[L, nb]`.
fn refCandMaskAndScores(
    gpa: std.mem.Allocator,
    ref_s: []const f32,
    L: usize,
    nb: usize,
    p0: usize,
    ratio: usize,
    bs: usize,
    kb: usize,
) !struct { cand_ids: []i32, post_scores: []f32 } {
    const n_blocks = (nb + bs - 1) / bs;
    const cand_ids = try gpa.alloc(i32, L * kb);
    errdefer gpa.free(cand_ids);
    const post_scores = try gpa.dupe(f32, ref_s);
    errdefer gpa.free(post_scores);

    const BEntry = struct { v: f32, id: usize };
    const b_arr = try gpa.alloc(BEntry, n_blocks);
    defer gpa.free(b_arr);
    const keep = try gpa.alloc(bool, nb);
    defer gpa.free(keep);

    const cmp = struct {
        fn less(_: void, a: BEntry, b: BEntry) bool {
            if (a.v != b.v) return a.v > b.v;
            return a.id < b.id;
        }
    }.less;

    for (0..L) |l| {
        const vis = @min(nb, (p0 + l + 1) / ratio);
        const vis_b = (vis + bs - 1) / bs;
        for (0..n_blocks) |b| {
            var mx: f32 = -std.math.inf(f32);
            const s0 = b * bs;
            const s1 = @min(vis, s0 + bs);
            if (s0 < s1) {
                for (s0..s1) |j| mx = @max(mx, ref_s[l * nb + j]);
            }
            b_arr[b] = .{ .v = mx, .id = b };
        }
        if (vis > 0) b_arr[(vis - 1) / bs].v = std.math.inf(f32);
        std.mem.sort(BEntry, b_arr, {}, cmp);

        const row_c = cand_ids[l * kb ..][0..kb];
        @memset(row_c, -1);
        @memset(keep, false);
        var kept: usize = 0;
        const take = @min(kb, n_blocks);
        // Selected block ids are sorted ascending, invisible blocks masked to -1.
        const picked = try gpa.alloc(i32, take);
        defer gpa.free(picked);
        for (0..take) |j| {
            const bid = b_arr[j].id;
            picked[j] = if (bid < vis_b) @intCast(bid) else -1;
        }
        std.mem.sort(i32, picked, {}, struct {
            fn asc(_: void, a: i32, b: i32) bool {
                // Match radix select order: valid ids ascending? Note: dec_topk puts larger score first then lower id,
                // emitted in ascending slot/id order! Since all invisible blocks have -inf, valid ids < vis_b have
                // finite/inf scores so they are picked first, and emitted in ascending id order.
                if (a < 0 and b < 0) return false;
                if (a < 0) return false;
                if (b < 0) return true;
                return a < b;
            }
        }.asc);
        for (picked, 0..) |bid, j| {
            row_c[j] = bid;
            if (bid >= 0) {
                kept += 1;
                const s0 = @as(usize, @intCast(bid)) * bs;
                for (s0..@min(nb, s0 + bs)) |c| keep[c] = true;
            }
        }
        for (0..nb) |c| {
            if (!keep[c]) post_scores[l * nb + c] = -std.math.inf(f32);
        }
    }
    return .{ .cand_ids = cand_ids, .post_scores = post_scores };
}

fn expectI32Eq(label: []const u8, want: []const i32, got: []const i32, cols: usize) !void {
    try testing.expectEqual(want.len, got.len);
    for (0..want.len / cols) |r| {
        const w = want[r * cols ..][0..cols];
        const g = got[r * cols ..][0..cols];
        if (!std.mem.eql(i32, w, g)) {
            const at = std.mem.indexOfDiff(i32, w, g).?;
            std.debug.print("{s}: row {d} col {d} differs: want {d} got {d}\n", .{ label, r, at, w[at], g[at] });
            return error.TopkMismatch;
        }
    }
}

test "sushi_ds41 gpu: indexScoresNax gives indexScoresTile's scores, and a slice scores the whole list's bits" {
    try requireM5Gpu();
    const IH = 32;
    const IHD = 128;
    const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
    const ICase = struct { L: usize, p0: usize, ratio: usize, nb: usize, dt: mlx.mlx_dtype = .bfloat16 };
    const cases = [_]ICase{
        .{ .L = 256, .p0 = 47104, .ratio = 2, .nb = 23680 },
        .{ .L = 256, .p0 = 47104, .ratio = 2, .nb = 23680, .dt = .float16 },
        .{ .L = 77, .p0 = 931, .ratio = 2, .nb = 504 },
        .{ .L = 128, .p0 = 0, .ratio = 2, .nb = 64 },
        .{ .L = 5, .p0 = 3000, .ratio = 4, .nb = 751 },
    };
    for (cases, 0..) |c, ci_| {
        var sc = Scope.plain(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const seed: u64 = 0x1dd0 + ci_ * 8;
        const q = try sc.astype(try randn(&sc, &.{ ci(c.L), IH, IHD }, seed, c.dt), .float32);
        const keys = try randn(&sc, &.{ ci(c.nb), IHD }, seed + 1, c.dt);
        const w = try randn(&sc, &.{ ci(c.L), IH }, seed + 2, c.dt);
        const want = try sc_mod.hostF32(try kernels.indexScoresTile(&sc, q, keys, w, wscale, c.p0, c.ratio, 0));
        const got = try sc_mod.hostF32(try inax.indexScoresNax(&sc, q, keys, w, wscale, c.p0, c.ratio, 0));
        for (0..c.L) |l| {
            var peak: f64 = 1e-30;
            for (want[l * c.nb ..][0..c.nb]) |v| if (!std.math.isInf(v)) {
                peak = @max(peak, @abs(@as(f64, v)));
            };
            for (got[l * c.nb ..][0..c.nb], want[l * c.nb ..][0..c.nb]) |g, v| {
                errdefer std.debug.print("case {d} row {d}: got {e} want {e} peak {e}\n", .{ ci_, l, g, v, peak });
                try testing.expectEqual(std.math.isInf(v), std.math.isInf(g));
                if (!std.math.isInf(v)) try testing.expect(@abs(@as(f64, g) - @as(f64, v)) <= 1e-6 * peak);
            }
        }
        const t0 = (c.nb / 2) / 32 * 32;
        if (t0 > 0) {
            const part = try sc_mod.hostF32(try inax.indexScoresNax(&sc, q, try sc.slice(keys, 0, t0, c.nb), w, wscale, c.p0, c.ratio, t0));
            const w2 = c.nb - t0;
            for (0..c.L) |l| try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[l * c.nb + t0 ..][0..w2]), std.mem.sliceAsBytes(part[l * w2 ..][0..w2]));
        }
    }
}

test "sushi_ds41 gpu: idx_nax_bmax + idx_nax_gather matches indexScoresNax & f32 reference 100% across L=256..8192, K=4k..131k, causal masking, and ties" {
    try requireM5Gpu();
    const gpa = testing.allocator;
    const gpu = mlx.gpuStream();
    try inax.probeDecTopk(gpa, gpu);

    const Case = struct {
        L: usize,
        nb: usize,
        p0: usize,
        ratio: usize,
        ih: usize = 32,
        ihd: usize = 128,
        bs: usize = 8,
        kb: usize = 2048,
        distinct: usize,
        dt: mlx.mlx_dtype = .bfloat16,
        check_f32_scalar: bool = false,
    };
    const cases = [_]Case{
        // Causal start with unseen rows (row 0 sees 0 keys, row 1 sees 1), ties via distinct=37, K=4k
        .{ .L = 256, .nb = 4096, .p0 = 0, .ratio = 2, .bs = 8, .kb = 513, .distinct = 37, .check_f32_scalar = true },
        // K=32k with ties across tiles and blocks
        .{ .L = 256, .nb = 32768, .p0 = 65280, .ratio = 2, .bs = 8, .kb = 2048, .distinct = 509, .check_f32_scalar = true },
        // K=131k (131,072 keys) with partial causal tail and ties
        .{ .L = 256, .nb = 131072, .p0 = 261900, .ratio = 2, .bs = 8, .kb = 2048, .distinct = 1021 },
        // Large batch L=8192 over K=4096 (single launch with BM=32 across 256 row tiles)
        .{ .L = 8192, .nb = 4096, .p0 = 8192, .ratio = 2, .bs = 8, .kb = 513, .distinct = 251 },
        // L=8192 chunk in SPAN=256 steps over K=32768
        .{ .L = 256, .nb = 32768, .p0 = 60000, .ratio = 2, .bs = 16, .kb = 1024, .distinct = 313, .dt = .float16 },
        // BS=32 + IH=8
        .{ .L = 77, .nb = 17000, .p0 = 33923, .ratio = 2, .ih = 8, .ihd = 64, .bs = 32, .kb = 513, .distinct = 199 },
    };

    for (cases, 0..) |cs, ci_| {
        var sc = Scope.plain(gpa, gpu);
        defer sc.deinit();
        const seed: u64 = 0x9000 + ci_ * 17;
        const q = try fp4Qat(&sc, try randn(&sc, &.{ ci(cs.L), ci(cs.ih), ci(cs.ihd) }, seed, .float32));
        const keys = try makeFp4Keys(&sc, cs.nb, cs.ihd, cs.distinct, seed + 1, cs.dt);
        const w = try randn(&sc, &.{ ci(cs.L), ci(cs.ih) }, seed + 2, cs.dt);
        const wscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cs.ihd))) / @sqrt(@as(f32, @floatFromInt(cs.ih)));
        const k = @min(@as(usize, 512), cs.nb);

        const lens_h = try gpa.alloc(i32, cs.L);
        defer gpa.free(lens_h);
        for (lens_h, 0..) |*v, l| v.* = @intCast((cs.p0 + l + 1) / cs.ratio);
        const lens = try sc.ints(lens_h, &.{ ci(cs.L), 1 });

        // Reference 1: indexScoresNax (full [L, nb] f32) + topkIndices
        const ref_nax_a = try inax.indexScoresNax(&sc, q, keys.unpacked, w, wscale, cs.p0, cs.ratio, 0);
        const ref_nax_s = try gpa.dupe(f32, try sc_mod.hostF32(ref_nax_a));
        defer gpa.free(ref_nax_s);
        const want_plain = try gpa.dupe(i32, try sc_mod.hostI32(try maskScores(&sc, ref_nax_a, lens)));
        defer gpa.free(want_plain);

        // Reference 2: scalar f32 FP4 reference (indexScoresPacked) when enabled
        if (cs.check_f32_scalar) {
            const ref_f32_a = try kernels.indexScoresPacked(&sc, q, keys.packed_k.codes, keys.packed_k.scales, w, wscale, cs.p0, cs.ratio, .fp4_ue8m0, 32);
            const ref_f32_s = try sc_mod.hostF32(ref_f32_a);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ref_f32_s), std.mem.sliceAsBytes(ref_nax_s));
        }

        // Reference candidate & post-candidate picks from ref_nax_s
        const n_blocks = (cs.nb + cs.bs - 1) / cs.bs;
        const post_ref = try refCandMaskAndScores(gpa, ref_nax_s, cs.L, cs.nb, cs.p0, cs.ratio, cs.bs, cs.kb);
        defer gpa.free(post_ref.cand_ids);
        defer gpa.free(post_ref.post_scores);
        const want_post = if (n_blocks > cs.kb)
            try gpa.dupe(i32, try sc_mod.hostI32(try maskScores(&sc, try sc.floats(post_ref.post_scores, &.{ ci(cs.L), ci(cs.nb) }), lens)))
        else
            try gpa.dupe(i32, want_plain);
        defer gpa.free(want_post);

        var cfg = cfg_mod.Config{};
        cfg.index_heads = @intCast(cs.ih);
        cfg.index_head_dim = @intCast(cs.ihd);
        cfg.index_topk = 512;
        cfg.cand_block_size = @intCast(cs.bs);
        cfg.cand_blocks = @intCast(cs.kb);

        const ks_fp4: inax.KeySource = .{ .fp4 = .{ .codes = keys.packed_k.codes, .scales = keys.packed_k.scales, .dt = cs.dt } };
        const ks_plain: inax.KeySource = .{ .plain = keys.unpacked };

        inline for (.{ .{ "fp4", ks_fp4 }, .{ "plain", ks_plain } }) |arm| {
            // 1. Plain layer (cand_source = -1)
            cfg.cand_source = -1;
            var cand_none = none;
            const got_plain_a = (try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 0, arm[1], w, q, wscale, cs.p0, cs.ratio, &cand_none)).?;
            try expectI32Eq(arm[0] ++ " plain", want_plain, try sc_mod.hostI32(got_plain_a), k);
            try testing.expect(cand_none.ctx == null);

            // 2. Candidate source layer (cand_source = 0)
            cfg.cand_source = 0;
            var sh_cand = none;
            defer if (sh_cand.ctx != null) {
                _ = mlx.mlx_array_free(sh_cand);
            };
            const got_csrc_a = (try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 0, arm[1], w, q, wscale, cs.p0, cs.ratio, &sh_cand)).?;
            try expectI32Eq(arm[0] ++ " cand_src", want_plain, try sc_mod.hostI32(got_csrc_a), k);
            if (n_blocks > cs.kb) {
                try expectI32Eq(arm[0] ++ " cand_ids", post_ref.cand_ids, try sc_mod.hostI32(sh_cand), cs.kb);
            } else {
                try testing.expect(sh_cand.ctx == null);
            }

            // 3. Post-candidate layer (li = 1 > cand_source = 0)
            const got_post_a = (try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 1, arm[1], w, q, wscale, cs.p0, cs.ratio, &sh_cand)).?;
            try expectI32Eq(arm[0] ++ " post_cand", want_post, try sc_mod.hostI32(got_post_a), k);
        }
    }
}

test "sushi_ds41 gpu: idx_nax_bmax and idx_nax_gather preserve exact sequential h=0..IH-1 order under +2^25/-2^25 cancellation" {
    try requireM5Gpu();
    const gpa = testing.allocator;
    const gpu = mlx.gpuStream();
    try inax.probeDecTopk(gpa, gpu);

    var sc = Scope.plain(gpa, gpu);
    defer sc.deinit();
    const Lc: usize = 17;
    const nbc: usize = 32;
    const qh = try gpa.alloc(f32, Lc * 32 * 128);
    defer gpa.free(qh);
    const kh = try gpa.alloc(f32, nbc * 128);
    defer gpa.free(kh);
    const wh = try gpa.alloc(f32, Lc * 32);
    defer gpa.free(wh);
    @memset(qh, 0);
    @memset(kh, 0);
    @memset(wh, 0);
    // Exact FP4 grid values (1.0, 1.5, 1.25 -> wait: 1.25 is not in E2M1 {0,0.5,1,1.5,2,3,4,6}, use 1.5 and 1.0, or 1.5 and 0.5*2)
    // Wait: test both plain bf16 (exact match to 1.25) AND fp4-packed (using exact E2M1 values 1.5 and 1.0 + 0.5)!
    kh[0 * 128 + 0] = 1.0;
    kh[8 * 128 + 1] = 1.5;
    kh[9 * 128 + 1] = 1.0;
    for (0..Lc) |row| {
        for ([_]usize{ 0, 1, 4, 5, 8 }) |h| qh[(row * 32 + h) * 128 + if (h == 1) @as(usize, 1) else 0] = 1.0;
        // At row 0: key 0 has h=0 (+2^25), h=4 (-2^25), h=5 (+1.0), h=8 (+1.0) -> sequential sum = 2.0, tree sum = 1.0;
        // key 8 has h=1 (+1.5), key 9 has h=1 (+1.0). With top-k=2, sequential picks {0, 8}, tree picks {8, 0/9} or ties!
        // Make key 9 have score 1.5 by setting kh[9*128+1] = 1.5? Wait: we want key 0 to beat key 9 ONLY under sequential!
        // With kh[8*128+1]=1.5 -> key 8 scores 1.5; kh[9*128+1]=1.0 -> if key 9 has weight 1.25*... wait: 1.5 > 1.0,
        // wait: if key 0 has h=0 (+2^25), h=1 (+1.0), h=4 (-2^25), h=5 (+0.5), tree sums (+2^25+1.0) -> 2^25, (-2^25+0.5) -> -2^25 -> 0.0!
        // Sequential sums (((+2^25 + 1.0) = 2^25) - 2^25 = 0.0) + 0.5 = 0.5!
        wh[row * 32 + 0] = 64.0 * 33554432.0; // +2^25 after wscale=1/64
        wh[row * 32 + 1] = 64.0;
        wh[row * 32 + 4] = -64.0 * 33554432.0; // -2^25 after wscale=1/64
        wh[row * 32 + 5] = 64.0;
        wh[row * 32 + 8] = 64.0;
    }
    const q_c = try sc.floats(qh, &.{ ci(Lc), 32, 128 });
    const k_f32 = try sc.floats(kh, &.{ ci(nbc), 128 });
    const k_c = try sc.astype(k_f32, .bfloat16);
    const pk_c = try qat_mod.packFp4(&sc, k_c, .fp4_ue8m0, 32);
    const w_c = try sc.astype(try sc.floats(wh, &.{ ci(Lc), 32 }), .bfloat16);

    var cfg = cfg_mod.Config{};
    cfg.index_heads = 32;
    cfg.index_head_dim = 128;
    cfg.index_topk = 2;
    cfg.cand_source = 0;
    cfg.cand_block_size = 8;
    cfg.cand_blocks = 3;
    const ws: f32 = 1.0 / 64.0;

    for ([_]inax.KeySource{
        .{ .plain = k_c },
        .{ .fp4 = .{ .codes = pk_c.codes, .scales = pk_c.scales, .dt = .bfloat16 } },
    }) |ks| {
        var sh_c = none;
        defer if (sh_c.ctx != null) {
            _ = mlx.mlx_array_free(sh_c);
        };
        const p_src = try sc_mod.hostI32((try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 0, ks, w_c, q_c, ws, 31, 1, &sh_c)).?);
        try testing.expectEqualSlices(i32, &.{ 0, 8 }, p_src[0..2]);
        const p_post = try sc_mod.hostI32((try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 1, ks, w_c, q_c, ws, 31, 1, &sh_c)).?);
        try testing.expectEqualSlices(i32, &.{ 0, 8 }, p_post[0..2]);
    }
}

test "sushi_ds41 gpu: fp4 indexer round trip is preserved bit-for-bit and differs from unquantized bf16" {
    try requireM5Gpu();
    const gpa = testing.allocator;
    const gpu = mlx.gpuStream();
    try inax.probeDecTopk(gpa, gpu);

    var sc = Scope.plain(gpa, gpu);
    defer sc.deinit();
    const L: usize = 32;
    const nb: usize = 2048;
    const ih: usize = 32;
    const ihd: usize = 128;
    const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd * ih)));

    // Continuous random normals (NOT on the FP4 grid initially)
    const q_raw = try randn(&sc, &.{ ci(L), ci(ih), ci(ihd) }, 0xfa1, .float32);
    const k_raw = try randn(&sc, &.{ ci(nb), ci(ihd) }, 0xfa2, .bfloat16);
    const w = try randn(&sc, &.{ ci(L), ci(ih) }, 0xfa3, .bfloat16);

    // Apply DS4.1's exact fp4 (/32, ue8m0) round trip on q and k
    const q_fp4 = try fp4Qat(&sc, q_raw);
    const k_fp4_bf16 = try fp4Qat(&sc, k_raw);
    const k_pk = try qat_mod.packFp4(&sc, k_fp4_bf16, .fp4_ue8m0, 32);
    const k_unpacked = try qat_mod.unpackFp4(&sc, k_pk, .fp4_ue8m0, 32, .bfloat16);

    // Reference scalar FP4 scores + topk
    const ref_f32_s = try kernels.indexScoresPacked(&sc, q_fp4, k_pk.codes, k_pk.scales, w, ws, nb - L, 1, .fp4_ue8m0, 32);
    const lens_h = try gpa.alloc(i32, L);
    defer gpa.free(lens_h);
    for (lens_h, 0..) |*v, l| v.* = @intCast(nb - L + l + 1);
    const lens = try sc.ints(lens_h, &.{ ci(L), 1 });
    const want = try gpa.dupe(i32, try sc_mod.hostI32(try maskScores(&sc, ref_f32_s, lens)));
    defer gpa.free(want);

    var cfg = cfg_mod.Config{};
    cfg.index_heads = ih;
    cfg.index_head_dim = ihd;
    cfg.index_topk = 512;
    cfg.cand_source = 0;
    cfg.cand_block_size = 8;
    cfg.cand_blocks = 1024; // > 512, < nb/bs? wait: nb=2048 -> n_blocks=256 <= 1024, so use kb=100? Wait: kb>512 needed for cand_src, or use nb=16384; test plain + bmax directly here!
    cfg.cand_source = -1;

    var cand_a = none;
    const got_fp4 = try gpa.dupe(i32, try sc_mod.hostI32((try inax.indexerOnlineTopk(
        &sc,
        &cfg,
        true,
        false,
        cfg_mod.index_stream_tile,
        0,
        .{ .fp4 = .{ .codes = k_pk.codes, .scales = k_pk.scales, .dt = .bfloat16 } },
        w,
        q_fp4,
        ws,
        nb - L,
        1,
        &cand_a,
    )).?));
    defer gpa.free(got_fp4);

    const got_unpacked = try gpa.dupe(i32, try sc_mod.hostI32((try inax.indexerOnlineTopk(
        &sc,
        &cfg,
        true,
        false,
        cfg_mod.index_stream_tile,
        0,
        .{ .plain = k_unpacked },
        w,
        q_fp4,
        ws,
        nb - L,
        1,
        &cand_a,
    )).?));
    defer gpa.free(got_unpacked);

    try expectI32Eq("fp4 packed vs scalar ref", want, got_fp4, 512);
    try expectI32Eq("fp4 unpacked vs scalar ref", want, got_unpacked, 512);

    // Also verify bmax + gather explicitly on FP4 packed vs unpacked
    const bmax_fp4 = try sc_mod.hostF32(try inax.indexScoresNaxBmaxFp4(&sc, try sc.astype(q_fp4, .bfloat16), k_pk.codes, k_pk.scales, .bfloat16, w, ws, nb - L, 1, 8, false));
    const bmax_plain = try sc_mod.hostF32(try inax.indexScoresNaxBmax(&sc, try sc.astype(q_fp4, .bfloat16), k_unpacked, w, ws, nb - L, 1, 8, false));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bmax_fp4), std.mem.sliceAsBytes(bmax_plain));

    // And verify that unquantized bf16 (without fp4 round trip) gives different scores!
    const raw_scores = try sc_mod.hostF32(try inax.indexScoresNax(&sc, q_raw, k_raw, w, ws, nb - L, 1, 0));
    const fp4_scores = try sc_mod.hostF32(ref_f32_s);
    try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(raw_scores), std.mem.sliceAsBytes(fp4_scores)));
}

test "sushi_ds41 gpu: small-M tail (1k/4k on 131k context) parity and decode M<=16 guard" {
    try requireM5Gpu();
    const gpa = testing.allocator;
    const gpu = mlx.gpuStream();
    try inax.probeDecTopk(gpa, gpu);

    var sc = Scope.plain(gpa, gpu);
    defer sc.deinit();
    const nb: usize = 131072;
    const ih: usize = 32;
    const ihd: usize = 128;
    const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd * ih)));
    const keys = try makeFp4Keys(&sc, nb, ihd, 1021, 0xca11, .bfloat16);
    const ks_fp4: inax.KeySource = .{ .fp4 = .{ .codes = keys.packed_k.codes, .scales = keys.packed_k.scales, .dt = .bfloat16 } };

    var cfg = cfg_mod.Config{};
    cfg.index_heads = ih;
    cfg.index_head_dim = ihd;
    cfg.index_topk = 512;
    cfg.cand_source = 0;
    cfg.cand_block_size = 8;
    cfg.cand_blocks = 2048;

    // 1. Decode M <= 16 MUST decline (null) so decode/verify path never changes or regresses.
    for ([_]usize{ 1, 4, 8, 16 }) |m_dec| {
        const qd = try fp4Qat(&sc, try randn(&sc, &.{ ci(m_dec), ci(ih), ci(ihd) }, 0xd0 + m_dec, .float32));
        const wd = try randn(&sc, &.{ ci(m_dec), ci(ih) }, 0xe0 + m_dec, .bfloat16);
        var cand_d = none;
        try testing.expect((try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 0, ks_fp4, wd, qd, ws, 262144 - m_dec, 2, &cand_d)) == null);
    }

    // 2. Small-M prefill tails (e.g. 1024-row and 4096-row chunks in 256-row spans, plus partial 77-row tail) on 131k cached context.
    for ([_]usize{ 77, 256 }) |span_l| {
        const p0 = nb * 2 - span_l;
        const q = try fp4Qat(&sc, try randn(&sc, &.{ ci(span_l), ci(ih), ci(ihd) }, 0xf00 + span_l, .float32));
        const w = try randn(&sc, &.{ ci(span_l), ci(ih) }, 0xf10 + span_l, .bfloat16);
        const lens_h = try gpa.alloc(i32, span_l);
        defer gpa.free(lens_h);
        for (lens_h, 0..) |*v, l| v.* = @intCast((p0 + l + 1) / 2);
        const lens = try sc.ints(lens_h, &.{ ci(span_l), 1 });

        const ref_a = try inax.indexScoresNax(&sc, q, keys.unpacked, w, ws, p0, 2, 0);
        const ref_s = try gpa.dupe(f32, try sc_mod.hostF32(ref_a));
        defer gpa.free(ref_s);
        const want = try gpa.dupe(i32, try sc_mod.hostI32(try maskScores(&sc, ref_a, lens)));
        defer gpa.free(want);
        const post_ref = try refCandMaskAndScores(gpa, ref_s, span_l, nb, p0, 2, 8, 2048);
        defer gpa.free(post_ref.cand_ids);
        defer gpa.free(post_ref.post_scores);
        const want_post = try gpa.dupe(i32, try sc_mod.hostI32(try maskScores(&sc, try sc.floats(post_ref.post_scores, &.{ ci(span_l), ci(nb) }), lens)));
        defer gpa.free(want_post);

        var sh_cand = none;
        defer if (sh_cand.ctx != null) {
            _ = mlx.mlx_array_free(sh_cand);
        };
        const got_csrc = (try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 0, ks_fp4, w, q, ws, p0, 2, &sh_cand)).?;
        try expectI32Eq("131k tail cand_src", want, try sc_mod.hostI32(got_csrc), 512);
        try expectI32Eq("131k tail cand_ids", post_ref.cand_ids, try sc_mod.hostI32(sh_cand), 2048);
        const got_post = (try inax.indexerOnlineTopk(&sc, &cfg, true, false, cfg_mod.index_stream_tile, 1, ks_fp4, w, q, ws, p0, 2, &sh_cand)).?;
        try expectI32Eq("131k tail post_cand", want_post, try sc_mod.hostI32(got_post), 512);
    }
}
