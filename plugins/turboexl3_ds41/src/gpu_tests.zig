//! Live Metal GPU parity tests for `turboexl3_ds41` (`zig build gpu-test -Dmlx-lib=<dir>`):
//! 1. Kernel registry validation (`manifest_sha256`, `lane_pins`, `source.sha256`) for the 4 refactored `b3_prep_*` kernels.
//! 2. Bit-exact GPU parity of the 4 `simd_shuffle_xor` `b3_prep_*` kernels across banks 0..2 vs stock `q3_*` twins.
//! 3. Bit-exact GPU parity of vectorized + TG-cached `tail_fuse` (`expand`, `expandDecode`, `expandInto`) vs scalar reference.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const tail_fuse = @import("bank/tail_fuse.zig");
const xk = @import("exl3/kernels.zig");

const testing = std.testing;

fn bf16FromF32(x: f32) u16 {
    // Round-to-nearest-even bf16 from f32
    const u: u32 = @bitCast(x);
    if ((u & 0x7FFF_FFFF) > 0x7F80_0000) return @truncate((u >> 16) | 0x0040);
    const lsb = (u >> 16) & 1;
    return @truncate((u + 0x7FFF + lsb) >> 16);
}

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn newKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn runKernel(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, ins: []const mlx.mlx_array, out_shapes: []const []const c_int, out_dts: []const mlx.mlx_dtype, grid: [3]c_int, tg: [3]c_int, outs: []mlx.mlx_array) !void {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (out_shapes, out_dts) |sh, dt| {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg[0], tg[1], tg[2]));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
    }
}

fn fillF16(buf: []f16, seed: usize) void {
    for (buf, 0..) |*v, i| {
        const val: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + seed * 7) % 29)) - 14)) * 0.05;
        v.* = @floatCast(val);
    }
}

fn fillBf16(buf: []u16, seed: usize) void {
    for (buf, 0..) |*v, i| {
        const val: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + seed * 5) % 31)) - 15)) * 0.06;
        v.* = bf16FromF32(val);
    }
}

fn fillF32(buf: []f32, seed: usize) void {
    for (buf, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + seed * 3) % 23)) - 11)) * 0.07;
    }
}

test "gpu: exl3 manifest pins validate with refactored b3_prep kernels" {
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, null);
    defer reg.deinit();
    try testing.expectEqual(@as(u32, 1280), reg.get(.dsv41_exl3_b3_prep_in_rin).launch.rule.grid[0].m);
    try testing.expectEqual(@as(u32, 32), reg.get(.dsv41_exl3_b3_prep_in_rin).launch.rule.threadgroup[0]);
    try testing.expectEqual(@as(u32, 576), reg.get(.dsv41_exl3_b3_prep_gu_epi).launch.rule.grid[0].m);
    try testing.expectEqual(@as(u32, 32), reg.get(.dsv41_exl3_b3_prep_gu_epi).launch.rule.threadgroup[0]);
    try testing.expectEqual(@as(u32, 576), reg.get(.dsv41_exl3_b3_prep_din_rin).launch.rule.grid[0].m);
    try testing.expectEqual(@as(u32, 32), reg.get(.dsv41_exl3_b3_prep_din_rin).launch.rule.threadgroup[0]);
    try testing.expectEqual(@as(u32, 1280), reg.get(.dsv41_exl3_b3_moeprep_dpost).launch.rule.grid[0].m);
    try testing.expectEqual(@as(u32, 32), reg.get(.dsv41_exl3_b3_moeprep_dpost).launch.rule.threadgroup[0]);
}

test "gpu: b3_prep simd_shuffle_xor kernels match stock q3_* twins bit-for-bit across all 3 banks" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var reg = try xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, null);
    defer reg.deinit();
    const s = mlx.gpuStream();
    const a = testing.allocator;

    const rows: c_int = 7;
    const cap: c_int = 4;
    const m_tokens: c_int = 8;
    const plain_slots = [_]u32{ 0, 2, 1, 3, 0, 1, 2 };
    const tok_vals = [_]i32{ 1, 5, 0, 7, 3, 2, 6 };

    const tok_arr = mlx.mlx_array_new_data(&tok_vals, &.{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(tok_arr);

    const x_buf = try a.alloc(u16, @intCast(m_tokens * 5120));
    defer a.free(x_buf);
    fillBf16(x_buf, 1);
    const x_arr = mlx.mlx_array_new_data(x_buf.ptr, &.{ m_tokens, 5120 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x_arr);

    const f32_5120 = try a.alloc(f32, @intCast(rows * 5120));
    defer a.free(f32_5120);
    fillF32(f32_5120, 2);
    const zd_arr = mlx.mlx_array_new_data(f32_5120.ptr, &.{ rows, 5120 }, 2, .float32);
    defer _ = mlx.mlx_array_free(zd_arr);

    const f32_2304_a = try a.alloc(f32, @intCast(rows * 2304));
    defer a.free(f32_2304_a);
    fillF32(f32_2304_a, 3);
    const zg_arr = mlx.mlx_array_new_data(f32_2304_a.ptr, &.{ rows, 2304 }, 2, .float32);
    defer _ = mlx.mlx_array_free(zg_arr);

    const f32_2304_b = try a.alloc(f32, @intCast(rows * 2304));
    defer a.free(f32_2304_b);
    fillF32(f32_2304_b, 4);
    const zu_arr = mlx.mlx_array_new_data(f32_2304_b.ptr, &.{ rows, 2304 }, 2, .float32);
    defer _ = mlx.mlx_array_free(zu_arr);

    const f16_5120 = try a.alloc(f16, @intCast(cap * 5120));
    defer a.free(f16_5120);
    const f16_2304 = try a.alloc(f16, @intCast(cap * 2304));
    defer a.free(f16_2304);

    var rg_b: [3]mlx.mlx_array = undefined;
    var ru_b: [3]mlx.mlx_array = undefined;
    var rgo_b: [3]mlx.mlx_array = undefined;
    var ruo_b: [3]mlx.mlx_array = undefined;
    var rn_b: [3]mlx.mlx_array = undefined;
    var rd_b: [3]mlx.mlx_array = undefined;
    for (0..3) |b| {
        fillF16(f16_5120, 10 + b);
        rg_b[b] = mlx.mlx_array_new_data(f16_5120.ptr, &.{ cap, 5120 }, 2, .float16);
        fillF16(f16_5120, 20 + b);
        ru_b[b] = mlx.mlx_array_new_data(f16_5120.ptr, &.{ cap, 5120 }, 2, .float16);
        fillF16(f16_2304, 30 + b);
        rgo_b[b] = mlx.mlx_array_new_data(f16_2304.ptr, &.{ cap, 2304 }, 2, .float16);
        fillF16(f16_2304, 40 + b);
        ruo_b[b] = mlx.mlx_array_new_data(f16_2304.ptr, &.{ cap, 2304 }, 2, .float16);
        fillF16(f16_2304, 50 + b);
        rn_b[b] = mlx.mlx_array_new_data(f16_2304.ptr, &.{ cap, 2304 }, 2, .float16);
        fillF16(f16_5120, 60 + b);
        rd_b[b] = mlx.mlx_array_new_data(f16_5120.ptr, &.{ cap, 5120 }, 2, .float16);
    }
    defer for (0..3) |b| {
        _ = mlx.mlx_array_free(rg_b[b]);
        _ = mlx.mlx_array_free(ru_b[b]);
        _ = mlx.mlx_array_free(rgo_b[b]);
        _ = mlx.mlx_array_free(ruo_b[b]);
        _ = mlx.mlx_array_free(rn_b[b]);
        _ = mlx.mlx_array_free(rd_b[b]);
    };

    const k_b3_in = try newKernel("b3_in", &.{ "x", "tok", "rg0", "rg1", "rg2", "ru0", "ru1", "ru2", "ids" }, &.{ "xg", "xu" }, reg.get(.dsv41_exl3_b3_prep_in_rin).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_b3_in);
    const k_q3_in = try newKernel("q3_in", &.{ "x", "tok", "rg", "ru", "ids" }, &.{ "xg", "xu" }, reg.get(.q3_exl3_prep_in_rin).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_q3_in);

    const k_b3_gu = try newKernel("b3_gu", &.{ "zg", "zu", "rg0", "rg1", "rg2", "ru0", "ru1", "ru2", "ids" }, &.{"out"}, reg.get(.dsv41_exl3_b3_prep_gu_epi).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_b3_gu);
    const k_q3_gu = try newKernel("q3_gu", &.{ "zg", "zu", "rg", "ru", "ids" }, &.{"out"}, reg.get(.q3_exl3_prep_gu_epi).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_q3_gu);

    const k_b3_din = try newKernel("b3_din", &.{ "hid", "rn0", "rn1", "rn2", "ids" }, &.{"out"}, reg.get(.dsv41_exl3_b3_prep_din_rin).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_b3_din);
    const k_q3_din = try newKernel("q3_din", &.{ "hid", "rn", "ids" }, &.{"out"}, reg.get(.q3_exl3_prep_din_rin).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_q3_din);

    const k_b3_dp = try newKernel("b3_dp", &.{ "zd", "rd0", "rd1", "rd2", "ids" }, &.{"out"}, reg.get(.dsv41_exl3_b3_moeprep_dpost).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_b3_dp);
    const k_q3_dp = try newKernel("q3_dp", &.{ "zd", "rd", "ids" }, &.{"out"}, reg.get(.q3_moeprep_dpost).source);
    defer _ = mlx.mlx_fast_metal_kernel_free(k_q3_dp);

    const s_ids = mlx.mlx_array_new_data(&plain_slots, &.{rows}, 1, .uint32);
    defer _ = mlx.mlx_array_free(s_ids);

    for (0..3) |b| {
        var packed_slots: [7]u32 = undefined;
        for (plain_slots, 0..) |sl, idx| packed_slots[idx] = (@as(u32, @intCast(b)) << 24) | sl;
        const b_ids = mlx.mlx_array_new_data(&packed_slots, &.{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(b_ids);

        // 1. in_rin
        var b3_in_out: [2]mlx.mlx_array = undefined;
        var q3_in_out: [2]mlx.mlx_array = undefined;
        try runKernel(s, k_b3_in, &.{ x_arr, tok_arr, rg_b[0], rg_b[1], rg_b[2], ru_b[0], ru_b[1], ru_b[2], b_ids }, &.{ &.{ rows, 5120 }, &.{ rows, 5120 } }, &.{ .float32, .float32 }, .{ 1280, rows, 1 }, .{ 32, 1, 1 }, &b3_in_out);
        defer for (b3_in_out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        try runKernel(s, k_q3_in, &.{ x_arr, tok_arr, rg_b[b], ru_b[b], s_ids }, &.{ &.{ rows, 5120 }, &.{ rows, 5120 } }, &.{ .float32, .float32 }, .{ 2560, rows, 1 }, .{ 64, 1, 1 }, &q3_in_out);
        defer for (q3_in_out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        for (b3_in_out, q3_in_out) |got, want| {
            try mlx.check(mlx.mlx_array_eval(got));
            try mlx.check(mlx.mlx_array_eval(want));
            const n: usize = @intCast(rows * 5120);
            try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(want).?[0..n], mlx.mlx_array_data_float32(got).?[0..n]);
        }

        // 2. gu_epi
        var b3_gu_out: [1]mlx.mlx_array = undefined;
        var q3_gu_out: [1]mlx.mlx_array = undefined;
        try runKernel(s, k_b3_gu, &.{ zg_arr, zu_arr, rgo_b[0], rgo_b[1], rgo_b[2], ruo_b[0], ruo_b[1], ruo_b[2], b_ids }, &.{&.{ rows, 2304 }}, &.{.float32}, .{ 576, rows, 1 }, .{ 32, 1, 1 }, &b3_gu_out);
        defer _ = mlx.mlx_array_free(b3_gu_out[0]);
        try runKernel(s, k_q3_gu, &.{ zg_arr, zu_arr, rgo_b[b], ruo_b[b], s_ids }, &.{&.{ rows, 2304 }}, &.{.float32}, .{ 1152, rows, 1 }, .{ 64, 1, 1 }, &q3_gu_out);
        defer _ = mlx.mlx_array_free(q3_gu_out[0]);
        try mlx.check(mlx.mlx_array_eval(b3_gu_out[0]));
        try mlx.check(mlx.mlx_array_eval(q3_gu_out[0]));
        {
            const n: usize = @intCast(rows * 2304);
            try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(q3_gu_out[0]).?[0..n], mlx.mlx_array_data_float32(b3_gu_out[0]).?[0..n]);
        }

        // 3. din_rin
        var b3_din_out: [1]mlx.mlx_array = undefined;
        var q3_din_out: [1]mlx.mlx_array = undefined;
        try runKernel(s, k_b3_din, &.{ zg_arr, rn_b[0], rn_b[1], rn_b[2], b_ids }, &.{&.{ rows, 2304 }}, &.{.float32}, .{ 576, rows, 1 }, .{ 32, 1, 1 }, &b3_din_out);
        defer _ = mlx.mlx_array_free(b3_din_out[0]);
        try runKernel(s, k_q3_din, &.{ zg_arr, rn_b[b], s_ids }, &.{&.{ rows, 2304 }}, &.{.float32}, .{ 1152, rows, 1 }, .{ 64, 1, 1 }, &q3_din_out);
        defer _ = mlx.mlx_array_free(q3_din_out[0]);
        try mlx.check(mlx.mlx_array_eval(b3_din_out[0]));
        try mlx.check(mlx.mlx_array_eval(q3_din_out[0]));
        {
            const n: usize = @intCast(rows * 2304);
            try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(q3_din_out[0]).?[0..n], mlx.mlx_array_data_float32(b3_din_out[0]).?[0..n]);
        }

        // 4. moeprep_dpost
        var b3_dp_out: [1]mlx.mlx_array = undefined;
        var q3_dp_out: [1]mlx.mlx_array = undefined;
        try runKernel(s, k_b3_dp, &.{ zd_arr, rd_b[0], rd_b[1], rd_b[2], b_ids }, &.{&.{ rows, 5120 }}, &.{.float32}, .{ 1280, rows, 1 }, .{ 32, 1, 1 }, &b3_dp_out);
        defer _ = mlx.mlx_array_free(b3_dp_out[0]);
        try runKernel(s, k_q3_dp, &.{ zd_arr, rd_b[b], s_ids }, &.{&.{ rows, 5120 }}, &.{.float32}, .{ 2560, rows, 1 }, .{ 64, 1, 1 }, &q3_dp_out);
        defer _ = mlx.mlx_array_free(q3_dp_out[0]);
        try mlx.check(mlx.mlx_array_eval(b3_dp_out[0]));
        try mlx.check(mlx.mlx_array_eval(q3_dp_out[0]));
        {
            const n: usize = @intCast(rows * 5120);
            try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(q3_dp_out[0]).?[0..n], mlx.mlx_array_data_float32(b3_dp_out[0]).?[0..n]);
        }
    }
}

test "gpu: vectorized tail_fuse expand, expandDecode, and expandInto match scalar reference bit-for-bit" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: usize = 3;
    const k: usize = 6;
    const hc: usize = 4;
    const d: usize = 256;

    const src_rows: usize = 5;
    var s0_buf: [src_rows * d]f32 = undefined;
    var s1_buf: [src_rows * d]f32 = undefined;
    fillF32(&s0_buf, 1);
    fillF32(&s1_buf, 2);
    const s0 = mlx.mlx_array_new_data(&s0_buf, &.{ src_rows, d }, 2, .float32);
    defer _ = mlx.mlx_array_free(s0);
    const s1 = mlx.mlx_array_new_data(&s1_buf, &.{ src_rows, d }, 2, .float32);
    defer _ = mlx.mlx_array_free(s1);

    var loc_buf: [n * k * 2]i32 = undefined;
    var w_buf: [n * k]f32 = undefined;
    for (0..n) |r| {
        for (0..k) |j| {
            loc_buf[(r * k + j) * 2] = @intCast((r + j) % 2);
            loc_buf[(r * k + j) * 2 + 1] = @intCast((r * 2 + j) % src_rows);
            w_buf[r * k + j] = @as(f32, @floatFromInt(@as(i32, @intCast(j)) - 2)) * 0.13 + 0.25;
        }
    }
    const loc = mlx.mlx_array_new_data(&loc_buf, &.{ n, k, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(loc);
    const w = mlx.mlx_array_new_data(&w_buf, &.{ n, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    var sh_buf: [n * d]u16 = undefined;
    fillBf16(&sh_buf, 3);
    const sh = mlx.mlx_array_new_data(&sh_buf, &.{ n, d }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(sh);

    var res_buf: [n * hc * d]u16 = undefined;
    fillBf16(&res_buf, 4);
    const res = mlx.mlx_array_new_data(&res_buf, &.{ n, hc, d }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(res);

    var pc_buf: [n * (hc + hc * hc)]f32 = undefined;
    fillF32(&pc_buf, 5);
    const pc = mlx.mlx_array_new_data(&pc_buf, &.{ n, hc + hc * hc }, 2, .float32);
    defer _ = mlx.mlx_array_free(pc);

    // Host reference matching tail_fuse.metal's exact rounding & FMA order.
    var want: [n * hc * d]u16 = undefined;
    for (0..n) |r| {
        const pr = pc_buf[r * (hc + hc * hc) ..];
        const cb = pr[hc..];
        for (0..d) |col| {
            var acc_t: f32 = 0;
            for (0..k) |ki| {
                const src_idx = loc_buf[(r * k + ki) * 2];
                const src_row: usize = @intCast(loc_buf[(r * k + ki) * 2 + 1]);
                const val = if (src_idx == 0) s0_buf[src_row * d + col] else s1_buf[src_row * d + col];
                const p = val * w_buf[r * k + ki] + 0.0;
                acc_t = if (ki == 0) p else p + acc_t;
            }
            const y = bf16ToF32(bf16FromF32(acc_t + bf16ToF32(sh_buf[r * d + col])));
            for (0..hc) |k2| {
                var acc: f32 = 0;
                for (0..hc) |j| {
                    acc = @mulAdd(f32, cb[j * hc + k2], bf16ToF32(res_buf[(r * hc + j) * d + col]), acc);
                }
                want[(r * hc + k2) * d + col] = bf16FromF32(@mulAdd(f32, pr[k2], y, acc));
            }
        }
    }

    const got_exp = try tail_fuse.expand(s, &.{ s0, s1 }, loc, w, sh, res, pc);
    defer _ = mlx.mlx_array_free(got_exp);
    try mlx.check(mlx.mlx_array_eval(got_exp));
    try testing.expectEqualSlices(u16, &want, mlx.mlx_array_data_bfloat16(got_exp).?[0 .. n * hc * d]);

    const got_dec = try tail_fuse.expandDecode(s, &.{ s0, s1 }, loc, w, sh, res, pc);
    defer _ = mlx.mlx_array_free(got_dec);
    try mlx.check(mlx.mlx_array_eval(got_dec));
    try testing.expectEqualSlices(u16, &want, mlx.mlx_array_data_bfloat16(got_dec).?[0 .. n * hc * d]);

    var res_copy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res_copy);
    try mlx.check(mlx.mlx_copy(&res_copy, res, s));
    try mlx.check(mlx.mlx_array_eval(res_copy));
    const carrier = try tail_fuse.expandInto(s, &.{ s0, s1 }, loc, w, sh, res_copy, pc);
    defer _ = mlx.mlx_array_free(carrier);
    try mlx.check(mlx.mlx_array_eval(carrier));
    try testing.expectEqualSlices(u16, &want, mlx.mlx_array_data_bfloat16(res_copy).?[0 .. n * hc * d]);
}
