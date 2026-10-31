//! Experiment only (branch exp/ds41-conf-head): a replacement for DSpark's confidence head, retrained offline on the STQ
//! verifier's accepts (`bench/kld/conf-head-train.py` in local-llm). `SUSHI_DS41_CONF_HEAD=<file>` names it at load.
//! The logit of draft k is `concat(hout[k], me[k]) . w + bias[k] + extra[k] . (p1, margin, entropy)` of its draft row.
//! Pure data, no MLX.
const std = @import("std");

pub const magic = "DSCH";
pub const n_stats = 3;

/// File layout, little-endian: "DSCH", u32 version (1), u32 width (dim + rank), u32 block, u32 n_extra (0 or 3), then
/// f32 w[width], f32 bias[block], f32 extra[block * n_extra].
pub const Head = struct {
    w: []f32,
    bias: []f32,
    /// `[block][n_stats]`, empty when the head reads no row stats.
    extra: []f32,
    block: usize,

    pub fn deinit(self: *Head, gpa: std.mem.Allocator) void {
        gpa.free(self.w);
        gpa.free(self.bias);
        gpa.free(self.extra);
    }
};

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, width: usize, block: usize) !Head {
    if (bytes.len < 20 or !std.mem.eql(u8, bytes[0..4], magic)) return error.ConfHeadBadMagic;
    const word = struct {
        fn u(b: []const u8, i: usize) u32 {
            return std.mem.readInt(u32, b[4 + 4 * i ..][0..4], .little);
        }
    };
    if (word.u(bytes, 0) != 1) return error.ConfHeadBadVersion;
    if (word.u(bytes, 1) != width or word.u(bytes, 2) != block) return error.ConfHeadShape;
    const n_extra: usize = word.u(bytes, 3);
    if (n_extra != 0 and n_extra != n_stats) return error.ConfHeadShape;
    const floats = width + block + block * n_extra;
    if (bytes.len != 20 + 4 * floats) return error.ConfHeadShape;
    const all = try gpa.alloc(f32, floats);
    defer gpa.free(all);
    for (all, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[20 + 4 * i ..][0..4], .little));
    var h: Head = .{ .w = &.{}, .bias = &.{}, .extra = &.{}, .block = block };
    errdefer h.deinit(gpa);
    h.w = try gpa.dupe(f32, all[0..width]);
    h.bias = try gpa.dupe(f32, all[width..][0..block]);
    h.extra = try gpa.dupe(f32, all[width + block ..]);
    return h;
}

const testing = std.testing;

fn testFile(gpa: std.mem.Allocator, width: u32, block: u32, n_extra: u32, floats: []const f32) ![]u8 {
    var buf: std.Io.Writer.Allocating = .init(gpa);
    errdefer buf.deinit();
    try buf.writer.writeAll(magic);
    for ([_]u32{ 1, width, block, n_extra }) |v| try buf.writer.writeInt(u32, v, .little);
    for (floats) |v| try buf.writer.writeInt(u32, @bitCast(v), .little);
    return buf.toOwnedSlice();
}

test "sushi_ds41 conf head: a file parses into weights, per-offset biases and row-stat weights" {
    const f = try testFile(testing.allocator, 3, 2, 3, &.{ 1, 2, 3, -0.5, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 });
    defer testing.allocator.free(f);
    var h = try parse(testing.allocator, f, 3, 2);
    defer h.deinit(testing.allocator);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, h.w);
    try testing.expectEqualSlices(f32, &.{ -0.5, 0.5 }, h.bias);
    try testing.expectEqualSlices(f32, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }, h.extra);
}

test "sushi_ds41 conf head: no row stats, and shapes that do not match the model are refused" {
    const f = try testFile(testing.allocator, 3, 2, 0, &.{ 1, 2, 3, -0.5, 0.5 });
    defer testing.allocator.free(f);
    var h = try parse(testing.allocator, f, 3, 2);
    defer h.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 0), h.extra.len);
    try testing.expectError(error.ConfHeadShape, parse(testing.allocator, f, 4, 2));
    try testing.expectError(error.ConfHeadShape, parse(testing.allocator, f[0 .. f.len - 4], 3, 2));
    try testing.expectError(error.ConfHeadBadMagic, parse(testing.allocator, "XXXX0000000000000000", 3, 2));
}
