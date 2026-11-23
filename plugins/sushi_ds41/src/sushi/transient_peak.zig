//! Cut edge: sushi's `SUSHI_TRANSIENT_PEAK` diagnostic reads counters of its patched MLX and engine that the host's
//! MLX does not export, so here the phase marks the model and the vision tower place are no-ops. Ours; the signatures
//! are sushi's.

pub const Phase = enum { decode, verify, draft, prefill_chunk, vision_encode };

pub fn begin() void {}

/// `rows` is the phase's width: tokens of the pass, or patches of an image; `bill` is ignored here.
pub fn end(phase: Phase, rows: usize, bill: ?u64) void {
    _ = phase;
    _ = rows;
    _ = bill;
}
