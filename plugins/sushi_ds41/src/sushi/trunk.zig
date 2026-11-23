//! Cut edge: `nax_gate.zig` reaches `verifyQmmEnabled` here; in the plugin QMM NAX is always allowed when hardware supports it.
pub fn verifyQmmEnabled() bool {
    return true;
}
