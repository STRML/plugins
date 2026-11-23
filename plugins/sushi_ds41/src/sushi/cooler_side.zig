//! Cut edge: the plugin runs on the host's MLX, never on sushi's Cooler engine.
pub fn onEngine(_: anytype) bool {
    return false;
}
