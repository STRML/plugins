//! Cut edge: sushi's Cooler engine (`src/tensor/route.zig`) is absent in the plugin, which runs on the host's MLX.
//! Everything here keeps `enabled = false` and `active = false` so `mlx.zig` routes every call straight to mlx-c.

const std = @import("std");
const mlx = @import("../mlx.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const V = mlx.mlx_vector_array;

pub var enabled: bool = false;
pub var active: bool = false;
pub var own_weights: bool = false;
pub var mlx_ops: u64 = 0;

pub const MlxAtRest = struct {
    on: std.atomic.Value(bool) = .init(false),
    active: std.atomic.Value(usize) = .init(0),
    cache: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    external: std.atomic.Value(usize) = .init(0),
};
pub var mlx_at_rest: MlxAtRest = .{};

pub const attention = struct {
    pub const MaskKind = enum { none, causal, array };
    pub const Tally = struct {
        q_len: usize,
        k_len: usize,
        heads: usize,
        kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        dtype: mlx.mlx_dtype,
        mask: MaskKind,
        force_fused: bool,
    };
    pub fn tallyKey(buf: []u8, _: Tally) ![]const u8 {
        return buf[0..0];
    }
};

pub const chains = struct {
    pub const sigmoid_gated_add = 0;
};

pub fn fusedChain(_: anytype, _: S, _: []const A) !?A {
    return null;
}

pub const kernel = struct {
    pub const max_inputs = 32;
    pub const Targ = struct {
        name: [32]u8 = @splat(0),
        name_len: usize = 0,
        value: union(enum) { int: c_int, dtype: mlx.mlx_dtype, boolean: bool } = .{ .int = 0 },
    };
    pub const Config = struct {
        twin: ?*anyopaque = null,
        grid: [3]c_int = .{ 0, 0, 0 },
        group: [3]c_int = .{ 0, 0, 0 },
        behind: bool = false,
        order: [64]bool = @splat(false),
        n_out: usize = 0,
        n_targs: usize = 0,
        out_shapes: [8][8]c_int = @splat(@splat(0)),
        out_ndims: [8]usize = @splat(0),
        out_dtypes: [8]mlx.mlx_dtype = @splat(.float32),
        targs: [32]Targ = @splat(.{}),
    };
    pub const Def = struct {
        twin: ?*anyopaque = null,
        name: []const u8 = "",
        source: []const u8 = "",
        header: []const u8 = "",
        inputs: []const []u8 = &.{},
        outputs: []const []u8 = &.{},
        ensure_row_contiguous: bool = false,
        atomic_outputs: bool = false,
    };
};

pub const Makers = struct {
    was: bool = false,
    pub fn end(_: Makers) void {}
};
pub fn makers(_: bool) Makers {
    return .{};
}
pub fn openDispatches() usize {
    return 0;
}
pub fn submit() !void {}
pub fn close() void {}
pub fn poolBytes(bytes: u64) u64 {
    return @import("pool_class.zig").poolBytes(@intCast(bytes));
}
pub fn copyOwned(_: *A, _: A) c_int {
    return 1;
}
pub fn wrapMapped(_: ?*const anyopaque, _: []const c_int, _: mlx.mlx_dtype, _: ?*anyopaque, _: anytype) A {
    return .{ .ctx = null };
}

pub const Scope = struct {
    pub fn end(_: Scope) void {}
};
pub fn enter(_: S) Scope {
    return .{};
}

pub fn stream() S {
    return .{ .ctx = null };
}
pub fn isStream(_: S) bool {
    return false;
}
pub fn isOurs(_: A) bool {
    return false;
}
pub fn isVec(_: V) bool {
    return false;
}
pub fn isMap(_: ?*anyopaque) bool {
    return false;
}
pub fn isConfig(_: mlx.mlx_fast_metal_kernel_config) bool {
    return false;
}
pub fn isKernel(_: mlx.mlx_fast_metal_kernel) bool {
    return false;
}
pub fn isStrings(_: mlx.mlx_vector_string) bool {
    return false;
}
pub fn isMapped(_: A) bool {
    return false;
}
pub fn mappedTotal() u64 {
    return 0;
}
pub fn poolRefusals() u64 {
    return 0;
}
pub fn touch() void {}
pub fn traceRefusal() void {}
pub fn settleMlxAliases() void {}
pub fn refuse(_: []const u8, _: []const u8) c_int {
    return 1;
}
pub fn matrixUnits() bool {
    return false;
}
pub fn seal(_: bool) void {}
pub fn room(_: []const usize) void {}
pub fn budget(_: u64) !void {}
pub fn trim() void {}
pub fn roomFor(_: []const c_int, _: mlx.mlx_dtype) void {}
pub fn matmulRowsAs(_: *A, _: A, _: A, _: usize) c_int {
    return 1;
}

pub const Side = enum { mlx, engine, leak };
pub fn side(_: anytype) Side {
    return .mlx;
}

pub fn lendsOnce(_: A) bool {
    return false;
}
pub fn lend(_: A) ?A {
    return null;
}
pub fn lentVector(_: []const A) V {
    return .{ .ctx = null };
}

pub fn mlx_array_free(_: A) c_int {
    return 0;
}
pub fn mlx_array_set(_: *A, _: A) c_int {
    return 1;
}
pub fn mlx_array_eval(_: A) c_int {
    return 1;
}
pub fn mlx_array_new_int(_: c_int) A {
    return .{ .ctx = null };
}
pub fn mlx_array_new_float(_: f32) A {
    return .{ .ctx = null };
}
pub fn mlx_array_new_bool(_: bool) A {
    return .{ .ctx = null };
}
pub fn mlx_array_new_data(_: ?*const anyopaque, _: [*]const c_int, _: c_int, _: mlx.mlx_dtype) A {
    return .{ .ctx = null };
}
pub fn mlx_vector_array_new() V {
    return .{ .ctx = null };
}
pub fn mlx_vector_array_new_data(_: [*]const A, _: usize) V {
    return .{ .ctx = null };
}
pub fn mlx_vector_array_new_value(_: A) V {
    return .{ .ctx = null };
}
pub fn mlx_vector_array_append_value(_: V, _: A) c_int {
    return 1;
}
pub fn mlx_map_string_to_array_new() mlx.mlx_map_string_to_array {
    return .{ .ctx = null };
}
pub fn mlx_map_string_to_string_new() mlx.mlx_map_string_to_string {
    return .{ .ctx = null };
}
pub fn mlx_synchronize(_: S) c_int {
    return 0;
}
pub fn mlx_eval(_: V) c_int {
    return 1;
}
pub fn mlx_async_eval(_: V) c_int {
    return 1;
}
pub fn mlx_vector_array_size(_: V) usize {
    return 0;
}
pub fn mlx_array_data_float32(_: A) ?[*]const f32 {
    return null;
}
pub fn mlx_array_data_int32(_: A) ?[*]const i32 {
    return null;
}
pub fn mlx_array_data_uint32(_: A) ?[*]const u32 {
    return null;
}
pub fn mlx_array_data_uint8(_: A) ?[*]const u8 {
    return null;
}
pub fn mlx_array_data_bfloat16(_: A) ?[*]const u16 {
    return null;
}
pub fn mlx_array_data_float16(_: A) ?[*]const f16 {
    return null;
}
pub fn storePointer(_: A) ?[*]const u8 {
    return null;
}
pub fn mlx_array_data_uint16(_: A) ?[*]const u16 {
    return null;
}
pub fn mlx_array_strides(_: A) [*]const usize {
    unreachable;
}
pub fn mlx_array_itemsize(_: A) usize {
    return 0;
}
pub fn ndim(_: A) usize {
    return 0;
}
pub fn shape(_: A) [*]const c_int {
    unreachable;
}
pub fn size(_: A) usize {
    return 0;
}
pub fn dtypeOf(_: A) mlx.mlx_dtype {
    return .float32;
}
pub fn item(comptime T: type, _: *T, _: A) c_int {
    return 1;
}

pub fn stringsNew(_: ?[*]const [*:0]const u8, _: usize) mlx.mlx_vector_string {
    return .{ .ctx = null };
}
pub fn stringsFree(_: mlx.mlx_vector_string) void {}
pub fn stringsItems(_: mlx.mlx_vector_string) []const [:0]u8 {
    return &.{};
}
pub fn configNew() mlx.mlx_fast_metal_kernel_config {
    return .{ .ctx = null };
}
pub fn configFree(_: mlx.mlx_fast_metal_kernel_config) void {}
pub fn configOf(_: mlx.mlx_fast_metal_kernel_config) *kernel.Config {
    unreachable;
}
pub fn configOutput(_: mlx.mlx_fast_metal_kernel_config, _: [*]const c_int, _: usize, _: mlx.mlx_dtype) c_int {
    return 1;
}
pub fn configTemplate(_: mlx.mlx_fast_metal_kernel_config, _: [*:0]const u8, _: anytype) c_int {
    return 1;
}
pub fn forgetConfig(_: mlx.mlx_fast_metal_kernel_config) void {}
pub fn caughtUp(_: mlx.mlx_fast_metal_kernel_config) ?*kernel.Config {
    return null;
}
pub fn keepsBehind(_: mlx.mlx_fast_metal_kernel_config) bool {
    return false;
}
pub fn noteOutput(_: mlx.mlx_fast_metal_kernel_config, _: [*]const c_int, _: usize, _: mlx.mlx_dtype) void {}
pub fn noteGrid(_: mlx.mlx_fast_metal_kernel_config, _: c_int, _: c_int, _: c_int) void {}
pub fn noteGroup(_: mlx.mlx_fast_metal_kernel_config, _: c_int, _: c_int, _: c_int) void {}
pub fn noteTemplate(_: mlx.mlx_fast_metal_kernel_config, _: [*:0]const u8, _: anytype) void {}
pub fn kernelNative(_: [*:0]const u8, _: mlx.mlx_vector_string, _: mlx.mlx_vector_string, _: [*:0]const u8, _: [*:0]const u8, _: bool, _: bool) mlx.mlx_fast_metal_kernel {
    return .{ .ctx = null };
}
pub fn kernelNew(_: mlx.mlx_fast_metal_kernel, _: [*:0]const u8, _: mlx.mlx_vector_string, _: mlx.mlx_vector_string, _: [*:0]const u8, _: [*:0]const u8, _: bool, _: bool) void {}
pub fn kernelOf(_: mlx.mlx_fast_metal_kernel) *kernel.Def {
    unreachable;
}
pub fn kernelFree(_: mlx.mlx_fast_metal_kernel) void {}
pub fn kernelApplyInto(_: *V, _: mlx.mlx_fast_metal_kernel, _: V, _: mlx.mlx_fast_metal_kernel_config, _: A) c_int {
    return 1;
}
pub fn mlx_fast_metal_kernel_apply(_: *V, _: mlx.mlx_fast_metal_kernel, _: V, _: mlx.mlx_fast_metal_kernel_config, _: S) c_int {
    return 1;
}

pub const pipeline_registry = struct {
    pub fn takeMarker(_: []const u8) bool {
        return false;
    }
    pub fn recordMarker(_: []const u8, _: []const u8) void {}
};
pub fn pipelinesMade() struct { count: u64 = 0 } {
    return .{};
}
