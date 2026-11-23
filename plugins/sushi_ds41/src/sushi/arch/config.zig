//! Cut edge: `expert_store.zig` asks `factsOfType` for `expert_source` and `streams_rate_groups`.
const std = @import("std");

pub const ExpertSource = enum { none, language_model_banks, flat_banks, individual_source, mxfp4_w123 };

pub const Facts = struct {
    expert_source: ExpertSource = .none,
    streams_rate_groups: bool = false,
};

pub fn factsOfType(model_type: []const u8) ?Facts {
    if (std.mem.eql(u8, model_type, "deepseek_v41")) {
        return .{ .expert_source = .mxfp4_w123, .streams_rate_groups = true };
    }
    return null;
}

pub fn isOf(_: anytype, _: []const u8) bool {
    return false;
}
