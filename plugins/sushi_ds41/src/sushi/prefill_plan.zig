//! Cut edge: the pure tail-merge and widest-forward arithmetic of sushi's `prefill_plan.zig` (used by `dsv41_config.chunkCap` and `plugin.zig`).

pub const TAIL_MERGE_MAX: usize = 512;
pub const SSM_SNAPSHOT_BACKOFF: usize = 30;

pub fn tailMergeMax(default_chunk: usize) usize {
    return @min(TAIL_MERGE_MAX, @max(default_chunk / 8, 1));
}

pub fn tailMergeMaxFor(default_chunk: usize, adaptive_width: bool) usize {
    return if (adaptive_width) tailMergeMax(default_chunk) else TAIL_MERGE_MAX;
}

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    adaptive_width: bool,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (want_ssm_cp and ssm_cp_stride > 0) {
        const abs_pos = pos + ssm_cp_offset;
        const abs_end = end + ssm_cp_offset;
        const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
        if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) {
            return next_boundary_abs - ssm_cp_offset;
        }
    }
    if (end < prefix_len and prefix_len - end < tailMergeMaxFor(default_chunk, adaptive_width)) {
        end = prefix_len;
    }
    return end;
}

pub fn widestForward(prefix_len: usize, default_chunk: usize, adaptive_width: bool) usize {
    if (prefix_len <= default_chunk) return prefix_len;
    var lo: usize = default_chunk;
    var hi: usize = prefix_len;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (nextChunkEnd(0, mid, default_chunk, false, 0, 0, adaptive_width) == mid) lo = mid else hi = mid - 1;
    }
    return lo;
}
