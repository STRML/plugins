//! Bounded LRU cache for grouped `wo_a` decoded `W_hat` matrices (`woa_join.Hats`), keyed by the weight's trellis
//! pointer (`w.w.ctx`). Prevents prefill chunk thrashing across layers while capping RAM usage and releasing entries
//! explicitly on weight release, decode phase transition, or bank close.

const std = @import("std");
const builtin = @import("builtin");

/// Physical RAM bytes via `hw.memsize` (0 if unavailable). MLX-free so both `config.bill` and `plugin.zig` agree.
pub fn hostMemBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    _ = std.c.sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    return mem;
}

/// Sensible RAM cap for cached `wo_a` `W_hat` buffers given physical RAM bytes (0 = unknown).
/// DS4.1 has 40 trunk layers (+3 MTP) at ~64 MiB/layer (~2.7 GiB total): 128 GB gets 3 GiB (fits all layers with
/// <2.5% RAM), smaller boxes scale down cleanly, and 512 GB boxes stay capped at 3 GiB.
pub fn defaultCapBytes(total_mem: u64) u64 {
    const cap: u64 = 3 << 30;
    const floor: u64 = 512 << 20;
    if (total_mem == 0) return 1 << 30;
    return @min(cap, @max(floor, total_mem / 42));
}

pub fn defaultHostCapBytes() u64 {
    return defaultCapBytes(hostMemBytes());
}

/// Saturating deduction of `hats_cap` from the streamed bank's prompt budget: the host bills only a single layer's
/// `wo_a` `W_hat` during prompt while `hats_cache` can hold up to `hats_cap` bytes until `Bank.phase(..., false)` clears it.
pub fn deductPromptCap(prompt_budget: u64, hats_cap: u64) u64 {
    return prompt_budget -| hats_cap;
}

pub fn LruCache(comptime V: type, comptime freeFn: fn (*V) void, comptime max_entries: usize) type {
    return struct {
        const Self = @This();

        const Slot = struct {
            key: ?*anyopaque = null,
            val: V = undefined,
            bytes: u64 = 0,
            last_used: u64 = 0,
            active: bool = false,
        };

        slots: [max_entries]Slot = @splat(.{}),
        max_bytes: u64,
        used_bytes: u64 = 0,
        count: usize = 0,
        tick: u64 = 0,

        pub fn init(max_bytes: u64) Self {
            return .{ .max_bytes = max_bytes };
        }

        fn nextTick(self: *Self) u64 {
            self.tick +%= 1;
            return self.tick;
        }

        /// Look up `key` and mark it most-recently-used.
        pub fn get(self: *Self, key: ?*anyopaque) ?*V {
            const k = key orelse return null;
            for (&self.slots) |*s| {
                if (s.active and s.key == k) {
                    s.last_used = self.nextTick();
                    return &s.val;
                }
            }
            return null;
        }

        fn evictUntil(self: *Self, limit_count: usize, limit_bytes: u64) void {
            while (self.count > limit_count or (self.count > 0 and self.used_bytes > limit_bytes)) {
                var victim_idx: ?usize = null;
                var oldest: u64 = std.math.maxInt(u64);
                for (self.slots, 0..) |s, i| {
                    if (s.active and s.last_used <= oldest) {
                        oldest = s.last_used;
                        victim_idx = i;
                    }
                }
                const vi = victim_idx orelse break;
                freeFn(&self.slots[vi].val);
                self.used_bytes -= self.slots[vi].bytes;
                self.slots[vi] = .{};
                self.count -= 1;
            }
        }

        /// Evict LRU entries *before* allocating `incoming_bytes` so live cache + pending entry never exceeds `max_bytes`.
        /// Returns false without evicting if `incoming_bytes > max_bytes`.
        pub fn reserve(self: *Self, incoming_bytes: u64) bool {
            if (incoming_bytes > self.max_bytes or max_entries == 0) return false;
            self.evictUntil(max_entries - 1, self.max_bytes - incoming_bytes);
            return true;
        }

        /// Insert `(key, val)` taking `bytes`, evicting LRU entries to stay within `max_entries` and `max_bytes`.
        /// Returns `null` (leaving `val` untouched for caller cleanup) if `key == null` or `bytes > max_bytes`.
        pub fn put(self: *Self, key: ?*anyopaque, val: V, bytes: u64) ?*V {
            const k = key orelse return null;
            if (bytes > self.max_bytes or max_entries == 0) return null;

            for (&self.slots) |*s| {
                if (s.active and s.key == k) {
                    freeFn(&s.val);
                    self.used_bytes -= s.bytes;
                    s.active = false;
                    self.count -= 1;
                    break;
                }
            }

            self.evictUntil(max_entries - 1, self.max_bytes - bytes);

            for (&self.slots) |*s| {
                if (!s.active) {
                    s.* = .{
                        .key = k,
                        .val = val,
                        .bytes = bytes,
                        .last_used = self.nextTick(),
                        .active = true,
                    };
                    self.used_bytes += bytes;
                    self.count += 1;
                    return &s.val;
                }
            }
            return null;
        }

        /// Explicitly free and remove `key` (e.g. on weight `release`).
        pub fn remove(self: *Self, key: ?*anyopaque) void {
            const k = key orelse return;
            for (&self.slots) |*s| {
                if (s.active and s.key == k) {
                    freeFn(&s.val);
                    self.used_bytes -= s.bytes;
                    s.* = .{};
                    self.count -= 1;
                    return;
                }
            }
        }

        /// Free all cached entries (e.g. on decode phase switch or bank close / unload).
        pub fn clear(self: *Self) void {
            for (&self.slots) |*s| {
                if (s.active) {
                    freeFn(&s.val);
                    s.* = .{};
                }
            }
            self.used_bytes = 0;
            self.count = 0;
        }
    };
}

// ── tests ──

const testing = std.testing;

test "hats cache: LRU eviction, byte cap, explicit remove, and clear" {
    const State = struct {
        var freed: [16]u32 = @splat(0);
        var freed_n: usize = 0;
        const Item = struct { id: u32 };
        fn free(it: *Item) void {
            freed[freed_n] = it.id;
            freed_n += 1;
        }
    };
    State.freed_n = 0;

    var k1: u8 = 1;
    var k2: u8 = 2;
    var k3: u8 = 3;
    var k4: u8 = 4;

    var cache = LruCache(State.Item, State.free, 4).init(200);
    _ = cache.put(&k1, .{ .id = 1 }, 80).?;
    _ = cache.put(&k2, .{ .id = 2 }, 80).?;
    try testing.expectEqual(@as(usize, 2), cache.count);
    try testing.expectEqual(@as(u64, 160), cache.used_bytes);

    // Touch k1 so k2 becomes LRU.
    try testing.expectEqual(@as(u32, 1), cache.get(&k1).?.id);

    // Pre-alloc reserve(80) evicts k2 BEFORE caller allocates k3, keeping live total <= 200.
    try testing.expect(cache.reserve(80));
    try testing.expectEqual(@as(usize, 1), cache.count);
    try testing.expectEqual(@as(u64, 80), cache.used_bytes);
    try testing.expectEqual(@as(u32, 2), State.freed[0]);
    try testing.expect(!cache.reserve(201));
    try testing.expectEqual(@as(usize, 1), cache.count);

    // Inserting k3 (80 bytes) fits within cap alongside k1.
    _ = cache.put(&k3, .{ .id = 3 }, 80).?;
    try testing.expectEqual(@as(usize, 2), cache.count);
    try testing.expectEqual(@as(u64, 160), cache.used_bytes);
    try testing.expect(cache.get(&k2) == null);
    try testing.expect(cache.get(&k1) != null);
    try testing.expectEqual(@as(usize, 1), State.freed_n);

    // Oversize item is rejected without evicting existing entries.
    try testing.expect(cache.put(&k4, .{ .id = 4 }, 201) == null);
    try testing.expectEqual(@as(usize, 2), cache.count);

    // Explicit remove frees only the removed key.
    cache.remove(&k1);
    try testing.expectEqual(@as(usize, 1), cache.count);
    try testing.expectEqual(@as(u64, 80), cache.used_bytes);
    try testing.expectEqual(@as(u32, 1), State.freed[1]);

    // Clear frees remaining entries.
    cache.clear();
    try testing.expectEqual(@as(usize, 0), cache.count);
    try testing.expectEqual(@as(u64, 0), cache.used_bytes);
    try testing.expectEqual(@as(usize, 3), State.freed_n);
    try testing.expectEqual(@as(u32, 3), State.freed[2]);
}

test "hats cache: defaultCapBytes scales cleanly for 64G, 128G, 512G, and unknown RAM" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(1 * gib, defaultCapBytes(0));
    try testing.expect(defaultCapBytes(64 * gib) >= 1 * gib and defaultCapBytes(64 * gib) <= 2 * gib);
    // 128 GB fits all 43 DS4.1 layers (~2.75 GiB) with cap at 3 GiB.
    try testing.expectEqual(3 * gib, defaultCapBytes(128 * gib));
    try testing.expectEqual(3 * gib, defaultCapBytes(512 * gib));
    try testing.expectEqual(12 * gib, deductPromptCap(15 * gib, defaultCapBytes(128 * gib)));
    try testing.expectEqual(@as(u64, 0), deductPromptCap(2 * gib, defaultCapBytes(128 * gib)));
}
