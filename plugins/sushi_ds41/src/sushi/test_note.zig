//! Cut edge: only sushi's tests reach this file, and they are not built here.
pub fn needArchs(_: []const []const u8) error{SkipZigTest}!void {
    return error.SkipZigTest;
}
