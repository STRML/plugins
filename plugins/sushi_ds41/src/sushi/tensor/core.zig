//! Cut edge: `tensor/core.zig` is used by `dsv41_model` (`Rows.append`) and `mlx_pages` for element sizes.
const mlx = @import("../mlx.zig");

pub fn itemSize(dt: mlx.mlx_dtype) u64 {
    return switch (dt) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}
