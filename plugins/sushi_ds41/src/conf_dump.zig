//! Experiment only (branch exp/ds41-conf-dump): training data for DSpark's confidence head against the STQ verifier.
//! With `SUSHI_DS41_CONF_DUMP=<dir>` set at load, every greedy round drafts and verifies its whole block and appends one
//! binary record per draft position to `<dir>/records.bin` (the driver moves it between requests); `<dir>/manifest.json`
//! describes the layout. The round commits STQ's own greedy picks, so the text is the verifier's. Pure data, no MLX.
const std = @import("std");

/// Little-endian u32/f32 words before the features: pos, k, label, draft, stq, rows, conf, p1, margin, entropy.
pub const head_words = 10;

/// `label` of a record: the draft matched STQ's argmax and every earlier one did (1), the round's first mismatch (0),
/// or a position past it, which the verifier never judged on its own context (2: replay only, not a training sample).
pub const Label = enum(u32) { rejected = 0, accepted = 1, past = 2 };

pub fn label(k: usize, accepted: usize) Label {
    if (k < accepted) return .accepted;
    if (k == accepted) return .rejected;
    return .past;
}

pub const Stats = struct { p1: f32, margin: f32, entropy: f32 };

/// The softmax top-1 probability, the top-1 minus top-2 logit, and the entropy (nats) of one draft row.
pub fn rowStats(row: []const f32) Stats {
    var m1: f32 = -std.math.inf(f32);
    var m2: f32 = -std.math.inf(f32);
    for (row) |v| {
        if (v > m1) {
            m2 = m1;
            m1 = v;
        } else if (v > m2) m2 = v;
    }
    var z: f64 = 0;
    var ez: f64 = 0; // sum e^(v-m1) * (v-m1)
    for (row) |v| {
        const d: f64 = v - m1;
        const e = @exp(d);
        z += e;
        ez += e * d;
    }
    const lz = @log(z);
    return .{ .p1 = @floatCast(1.0 / z), .margin = m1 - m2, .entropy = @floatCast(lz - ez / z) };
}

pub const Round = struct {
    /// The position of the round's `t1` (the request's committed length).
    pos: usize,
    rows: usize,
    drafts: []const u32,
    /// STQ's argmax at each verified row.
    stq: []const u32,
    accepted: usize,
    conf: []const f32,
    /// The Markov-biased draft rows `[drafts.len, vocab]`.
    logits: []const f32,
    vocab: usize,
    /// `concat(hout[k], me[k])` per draft, `[drafts.len, dim + rank]`.
    feats: []const f32,
    width: usize,
};

/// One record per verified draft position (rows - 1 of them).
pub fn writeRound(w: *std.Io.Writer, r: Round) !void {
    const n = @min(r.drafts.len, r.rows -| 1);
    for (0..n) |k| {
        const s = rowStats(r.logits[k * r.vocab ..][0..r.vocab]);
        const ints = [_]u32{ @intCast(r.pos), @intCast(k), @backingInt(label(k, r.accepted)), r.drafts[k], r.stq[k], @intCast(r.rows) };
        for (ints) |v| try w.writeInt(u32, v, .little);
        for ([_]f32{ r.conf[k], s.p1, s.margin, s.entropy }) |v| try w.writeInt(u32, @bitCast(v), .little);
        for (r.feats[k * r.width ..][0..r.width]) |v| try w.writeInt(u32, @bitCast(v), .little);
    }
}

pub fn writeManifest(w: *std.Io.Writer, dim: usize, rank: usize, block: usize) !void {
    try w.print(
        "{{\"format\":\"sushi_ds41-conf-dump-v1\",\"dim\":{d},\"rank\":{d},\"block\":{d},\"record_words\":{d}," ++
            "\"head\":[\"pos:u32\",\"k:u32\",\"label:u32 (1 accepted, 0 first reject, 2 past it)\",\"draft:u32\",\"stq:u32\",\"rows:u32\"," ++
            "\"conf:f32\",\"p1:f32\",\"margin:f32\",\"entropy:f32\"],\"then\":[\"hout:f32[dim]\",\"me:f32[rank]\"]}}\n",
        .{ dim, rank, block, head_words + dim + rank },
    );
}

const testing = std.testing;

test "sushi_ds41 conf: labels stop at the first mismatch" {
    try testing.expectEqual(Label.accepted, label(0, 2));
    try testing.expectEqual(Label.accepted, label(1, 2));
    try testing.expectEqual(Label.rejected, label(2, 2));
    try testing.expectEqual(Label.past, label(3, 2));
    try testing.expectEqual(Label.rejected, label(0, 0));
}

test "sushi_ds41 conf: row stats of a known row" {
    // logits ln 2, 0, 0: softmax 0.5, 0.25, 0.25; entropy = 0.5 ln 2 + 2 * 0.25 ln 4 = 1.5 ln 2.
    const s = rowStats(&.{ @log(@as(f32, 2)), 0, 0 });
    try testing.expectApproxEqAbs(@as(f32, 0.5), s.p1, 1e-6);
    try testing.expectApproxEqAbs(@log(@as(f32, 2)), s.margin, 1e-6);
    try testing.expectApproxEqAbs(1.5 * @log(@as(f32, 2)), s.entropy, 1e-6);
}

test "sushi_ds41 conf: a round writes one fixed-size record per verified draft" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    const vocab = 3;
    const width = 2;
    try writeRound(&buf.writer, .{
        .pos = 40,
        .rows = 3, // two drafts verified of three
        .drafts = &.{ 7, 8, 9 },
        .stq = &.{ 7, 5, 6 },
        .accepted = 1,
        .conf = &.{ 1.5, -0.5, 0.25 },
        .logits = &.{ 0, 0, 0, 1, 0, 0, 2, 0, 0 },
        .vocab = vocab,
        .feats = &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 },
        .width = width,
    });
    const words = head_words + width;
    try testing.expectEqual(@as(usize, 2 * words * 4), buf.written().len);
    const at = struct {
        fn u(b: []const u8, rec: usize, i: usize) u32 {
            return std.mem.readInt(u32, b[(rec * words + i) * 4 ..][0..4], .little);
        }
        fn f(b: []const u8, rec: usize, i: usize) f32 {
            return @bitCast(u(b, rec, i));
        }
    };
    const b = buf.written();
    try testing.expectEqual(@as(u32, 40), at.u(b, 0, 0));
    try testing.expectEqual(@as(u32, 1), at.u(b, 1, 1)); // k
    try testing.expectEqual(@as(u32, 1), at.u(b, 0, 2)); // accepted
    try testing.expectEqual(@as(u32, 0), at.u(b, 1, 2)); // first reject
    try testing.expectEqual(@as(u32, 8), at.u(b, 1, 3));
    try testing.expectEqual(@as(u32, 5), at.u(b, 1, 4));
    try testing.expectEqual(@as(u32, 3), at.u(b, 1, 5));
    try testing.expectEqual(@as(f32, -0.5), at.f(b, 1, 6));
    try testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), at.f(b, 0, 7), 1e-6);
    try testing.expectEqual(@as(f32, 0.3), at.f(b, 1, head_words));
    try testing.expectEqual(@as(f32, 0.4), at.f(b, 1, head_words + 1));
}

test "sushi_ds41 conf: the manifest names the record size" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    try writeManifest(&buf.writer, 4096, 256, 5);
    try testing.expect(std.mem.indexOf(u8, buf.written(), "\"record_words\":4362") != null);
}
