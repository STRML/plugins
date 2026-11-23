//! Cut edge: Cooler pipeline-making guard (`dsv41_vision.zig`); always null on MLX's side.
pub const Making = struct {
    pub fn end(_: Making) void {}
};
pub fn making(_: bool) ?Making {
    return null;
}
