//! Experiment only (branch exp/ds41-draft-audit): the DSpark drafter's agreement with the STQ verifier, teacher-forced
//! along a reference text. With `SUSHI_DS41_AUDIT_OUT` set, every greedy round drafts the full block, verifies it, and
//! commits the forced text (`SUSHI_DS41_AUDIT_FORCE`, the reference continuation of the prompt being served) instead of
//! the verifier's picks, appending one JSON line per round. Pure data, no MLX.
const std = @import("std");

/// Token ids separated by whitespace or commas (the fixture's `generated_tokens.txt` as is).
pub fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,");
    while (it.next()) |tok| try out.append(gpa, try std.fmt.parseInt(u32, tok, 10));
    return out.toOwnedSlice(gpa);
}

pub const Commit = struct { accepted: usize, next: u32 };

/// The forced round: the leading drafts that equal the forced text stay, the forced token after them is next. `force`
/// is the forced text from the first draft's position on; null when it runs out before the next token (the round
/// falls back to the verifier's picks).
pub fn forcedCommit(drafts: []const u32, force: []const u32) ?Commit {
    var a: usize = 0;
    while (a < drafts.len and a < force.len and drafts[a] == force[a]) a += 1;
    if (a >= force.len) return null;
    return .{ .accepted = a, .next = force[a] };
}

pub const Line = struct {
    p: usize,
    pos: usize,
    t1: u32,
    t1_matches_force: bool,
    /// The committed ids from P to `pos` equal the forced text.
    history_ok: bool,
    drafts: []const u32,
    conf: []const f32,
    stq: []const u32,
    force: []const u32,
    accepted: usize,
    rows: usize,
    forced: bool,
};

pub fn writeLine(w: *std.Io.Writer, l: Line) !void {
    try w.print("{{\"p\":{d},\"pos\":{d},\"t1\":{d},\"t1_matches_force\":{},\"history_ok\":{},\"drafts\":", .{ l.p, l.pos, l.t1, l.t1_matches_force, l.history_ok });
    try writeInts(w, l.drafts);
    try w.writeAll(",\"conf\":[");
    for (l.conf, 0..) |c, i| try w.print("{s}{d:.4}", .{ if (i > 0) "," else "", c });
    try w.writeAll("],\"stq\":");
    try writeInts(w, l.stq);
    try w.writeAll(",\"force\":");
    try writeInts(w, l.force);
    try w.print(",\"accepted\":{d},\"rows\":{d},\"forced\":{}}}\n", .{ l.accepted, l.rows, l.forced });
}

fn writeInts(w: *std.Io.Writer, xs: []const u32) !void {
    try w.writeByte('[');
    for (xs, 0..) |x, i| try w.print("{s}{d}", .{ if (i > 0) "," else "", x });
    try w.writeByte(']');
}

const testing = std.testing;

test "sushi_ds41 audit: forced accept keeps the drafts that match the forced text, then the forced token" {
    // Two drafts match, the third does not: next is the forced token at the third draft's position.
    try testing.expectEqual(Commit{ .accepted = 2, .next = 9 }, forcedCommit(&.{ 5, 6, 7, 8 }, &.{ 5, 6, 9, 10, 11 }).?);
    // The first draft already differs.
    try testing.expectEqual(Commit{ .accepted = 0, .next = 4 }, forcedCommit(&.{ 5, 6 }, &.{ 4, 6, 7 }).?);
    // Every draft matches: next is the forced token after the last draft.
    try testing.expectEqual(Commit{ .accepted = 3, .next = 8 }, forcedCommit(&.{ 5, 6, 7 }, &.{ 5, 6, 7, 8 }).?);
    // No drafts (a one-row round): the forced token at t1's next position.
    try testing.expectEqual(Commit{ .accepted = 0, .next = 5 }, forcedCommit(&.{}, &.{5}).?);
}

test "sushi_ds41 audit: the forced text running out falls back to the verifier" {
    // Matches up to its end, so the token after the accepted drafts is not forced.
    try testing.expect(forcedCommit(&.{ 5, 6, 7 }, &.{ 5, 6 }) == null);
    try testing.expect(forcedCommit(&.{ 5, 6 }, &.{}) == null);
    // A mismatch inside the forced text still forces, though the rest of the block runs past it.
    try testing.expectEqual(Commit{ .accepted = 0, .next = 4 }, forcedCommit(&.{ 5, 6, 7 }, &.{4}).?);
}

test "sushi_ds41 audit: ids parse from commas or whitespace, and a line is one JSON object" {
    const ids = try parseIds(testing.allocator, "1,22, 333\n4\t5\n");
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 1, 22, 333, 4, 5 }, ids);
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    try writeLine(&buf.writer, .{ .p = 52, .pos = 53, .t1 = 7, .t1_matches_force = true, .history_ok = true, .drafts = &.{ 8, 9 }, .conf = &.{ 1.5, -0.25 }, .stq = &.{ 8, 3, 4 }, .force = &.{ 8, 3 }, .accepted = 1, .rows = 3, .forced = true });
    try testing.expectEqualStrings(
        "{\"p\":52,\"pos\":53,\"t1\":7,\"t1_matches_force\":true,\"history_ok\":true,\"drafts\":[8,9],\"conf\":[1.5000,-0.2500],\"stq\":[8,3,4],\"force\":[8,3],\"accepted\":1,\"rows\":3,\"forced\":true}\n",
        buf.written(),
    );
}
