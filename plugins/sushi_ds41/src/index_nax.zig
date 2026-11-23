//! DeepSeek-V4.1's prefill index scorer and online block-max + register-staged NAX gather top-k on the M5 GPU's
//! matrix units (`matmul2d`):
//! - `indexScoresNax` / `indexScoresNaxFp4`: exact sequential `h = 0..IH-1` head accumulation over `matmul2d(16, 32, DK)`.
//! - `indexScoresNaxBmax` / `indexScoresNaxBmaxFp4`: online block-max reduction over `BS` (`8, 16, 32`) keys without
//!   materializing `[L, nb]` f32 scores.
//! - `indexScoresNaxGather` / `indexScoresNaxGatherFp4`: register-staged `q` + direct key gather for candidate blocks.
//! - `pickRows`: register-resident radix top-k (`dec_topk`), optionally mapping gathered slots directly to key ids.
//! - `indexerOnlineTopk`: zero-`[L, nb]` prefill indexer for wide spans (`L > NARROW_ROWS`), supporting both packed
//!   `.fp4_ue8m0` (`group = 32`) and unpacked `bf16`/`f16` keys.
const std = @import("std");
const mlx = @import("sushi/mlx.zig");
const sc_mod = @import("sushi/dsv41/dsv41_scope.zig");
const kernels = @import("sushi/dsv41/dsv41_kernels.zig");
const qat_mod = @import("sushi/dsv41/dsv41_qat.zig");
const cfg_mod = @import("sushi/dsv41/dsv41_config.zig");
const nax_gate = @import("sushi/nax_gate.zig");

const A = sc_mod.A;
const S = sc_mod.S;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;
const none = sc_mod.none;

const IDX_NAX_HEADER: [:0]const u8 =
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
    \\constant float DSV41_IDX_E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
;

const IDX_NAX_SHARED_TOP: [:0]const u8 =
    \\constexpr int WM = BM / 16;
    \\constexpr int WN = BN / 32;
    \\constexpr int NSG = WM * WN;
    \\constexpr int TGN = NSG * 32;
    \\const int L = q_shape[0];
    \\const int l0 = int(threadgroup_position_in_grid.x) * BM;
    \\const int n0 = int(threadgroup_position_in_grid.y) * BN;
    \\const uint tid = thread_index_in_threadgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const int p0 = meta[0], ratio = meta[1], key0 = meta[2];
    \\const int rows_left = metal::min(BM, L - l0);
    \\const int max_seen = (p0 + l0 + rows_left) / ratio - key0;
    \\const int end = metal::min(nb, n0 + BN);
;

const IDX_NAX_EARLY_SCORES: [:0]const u8 =
    \\if (n0 >= max_seen) {
    \\  for (int r = 0; r < rows_left; ++r)
    \\    for (int n = n0 + int(tid); n < end; n += TGN) scores[size_t(l0 + r) * nb + n] = -INFINITY;
    \\  return;
    \\}
;

const IDX_NAX_EARLY_BMAX: [:0]const u8 =
    \\const int n_blocks = (nb + BS - 1) / BS;
    \\if (n0 >= max_seen) {
    \\  const int b0 = n0 / BS;
    \\  const int bend = (end + BS - 1) / BS;
    \\  for (int r = 0; r < rows_left; ++r)
    \\    for (int b = b0 + int(tid); b < bend; b += TGN) bmax[size_t(l0 + r) * n_blocks + b] = -INFINITY;
    \\  return;
    \\}
;

const IDX_NAX_KT_PLAIN: [:0]const u8 =
    \\constexpr int C4 = IHD / 4;
    \\using T4 = metal::vec<T, 4>;
    \\threadgroup metal::vec<T, 8> kt_v[BN * (IHD / 8)];
    \\threadgroup T* kt = (threadgroup T*)kt_v;
    \\for (uint i = tid; i < uint(BN * C4); i += uint(TGN)) {
    \\  const int n = n0 + int(i / C4);
    \\  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
;

const IDX_NAX_KT_FP4: [:0]const u8 =
    \\constexpr int C8 = IHD / 8;
    \\constexpr int NG = IHD / 32;
    \\threadgroup metal::vec<T, 8> kt_v[BN * C8];
    \\threadgroup T* kt = (threadgroup T*)kt_v;
    \\for (uint i = tid; i < uint(BN * C8); i += uint(TGN)) {
    \\  const int n = n0 + int(i / C8);
    \\  const int c8 = int(i % C8);
    \\  metal::vec<T, 8> v = metal::vec<T, 8>(T(0));
    \\  if (n < nb) {
    \\    const uint packed = ((const device uint*)codes)[size_t(n) * C8 + c8];
    \\    const uint sb = uint(scales[size_t(n) * NG + (c8 >> 2)]);
    \\    const float sc = as_type<float>(metal::clamp(sb, 1u, 254u) << 23);
    \\    #pragma clang loop unroll(full)
    \\    for (uint j = 0; j < 8u; ++j) {
    \\      const uint code = (packed >> (j * 4u)) & 15u;
    \\      const float fv = DSV41_IDX_E2M1[code & 7u] * sc;
    \\      v[j] = T((code & 8u) ? -fv : fv);
    \\    }
    \\  }
    \\  kt_v[i] = v;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
;

const IDX_NAX_MATMUL: [:0]const u8 =
    \\const int sg_m = int(sg) / WN;
    \\const int sg_n = int(sg) % WN;
    \\const int sg_l0 = l0 + sg_m * 16;
    \\if (sg_l0 >= L) return;
    \\const int sg_n0 = n0 + sg_n * 32;
    \\const int sg_rows = metal::min(16, L - sg_l0);
    \\const ushort qid = ushort(lane >> 2);
    \\const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
    \\const ushort fn = ((qid & 2) | (ushort(lane) & 1)) * 4;
    \\const int r0 = sg_l0 + int(fm);
    \\const int r1 = r0 + 8;
    \\const int r0_safe = metal::min(r0, L - 1);
    \\const int r1_safe = metal::min(r1, L - 1);
    \\const float ws = wsc[0];
    \\using ext_t = metal::dextents<int, 2>;
    \\using str_t = metal::array<int, 2>;
    \\constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    \\    16, 32, DK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    \\mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\tensor<threadgroup T, ext_t, metal::tensor_inline> B(kt + sg_n * 32 * IHD, ext_t(IHD, 32));
    \\auto b0_t = B.template slice<DK, 32>(0, 0);
    \\tensor<device T, ext_t, metal::tensor_inline> A0(
    \\    (device T*)q + size_t(sg_l0) * IH * IHD, ext_t(IHD, sg_rows), str_t{1, IH * IHD});
    \\auto a0_t = A0.template slice<DK, 16>(0, 0);
    \\auto ct_c = op.template get_destination_cooperative_tensor<decltype(a0_t), decltype(b0_t), float>();
    \\float acc[16];
    \\#pragma clang loop unroll(full)
    \\for (short e = 0; e < 16; ++e) acc[e] = 0.0f;
    \\for (int h = 0; h < IH; ++h) {
    \\  const float w0 = float(w[size_t(r0_safe) * IH + h]) * ws;
    \\  const float w1 = float(w[size_t(r1_safe) * IH + h]) * ws;
    \\  tensor<device T, ext_t, metal::tensor_inline> A(
    \\      (device T*)q + (size_t(sg_l0) * IH + h) * IHD, ext_t(IHD, sg_rows), str_t{1, IH * IHD});
    \\  #pragma clang loop unroll(full)
    \\  for (short e = 0; e < 16; ++e) ct_c[e] = 0.0f;
    \\  for (int k0 = 0; k0 < IHD; k0 += DK) {
    \\    auto a = A.template slice<DK, 16>(k0, 0);
    \\    auto b = B.template slice<DK, 32>(k0, 0);
    \\    op.run(a, b, ct_c);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (short j = 0; j < 4; ++j) {
    \\    const float d0 = ct_c[j], d1 = ct_c[4 + j], d2 = ct_c[8 + j], d3 = ct_c[12 + j];
    \\    acc[j]      += w0 * ((d0 < 0.0f) ? 0.0f : d0);
    \\    acc[4 + j]  += w1 * ((d1 < 0.0f) ? 0.0f : d1);
    \\    acc[8 + j]  += w0 * ((d2 < 0.0f) ? 0.0f : d2);
    \\    acc[12 + j] += w1 * ((d3 < 0.0f) ? 0.0f : d3);
    \\  }
    \\}
    \\const int seen0 = metal::min((p0 + r0 + 1) / ratio - key0, nb);
    \\const int seen1 = metal::min((p0 + r1 + 1) / ratio - key0, nb);
;

const IDX_NAX_TAIL_SCORES: [:0]const u8 =
    \\#pragma clang loop unroll(full)
    \\for (short j = 0; j < 4; ++j) {
    \\  const int c0 = sg_n0 + int(fn) + int(j);
    \\  const int c1 = c0 + 16;
    \\  if (r0 < L) {
    \\    if (c0 < nb) scores[size_t(r0) * nb + c0] = c0 < seen0 ? acc[j] : -INFINITY;
    \\    if (c1 < nb) scores[size_t(r0) * nb + c1] = c1 < seen0 ? acc[8 + j] : -INFINITY;
    \\  }
    \\  if (r1 < L) {
    \\    if (c0 < nb) scores[size_t(r1) * nb + c0] = c0 < seen1 ? acc[4 + j] : -INFINITY;
    \\    if (c1 < nb) scores[size_t(r1) * nb + c1] = c1 < seen1 ? acc[12 + j] : -INFINITY;
    \\  }
    \\}
;

const IDX_NAX_TAIL_BMAX: [:0]const u8 =
    \\auto idx_bmax2 = [](float a, float b) -> float {
    \\  return (metal::isnan(a) || metal::isnan(b)) ? NAN : metal::max(a, b);
    \\};
    \\float m0a = -INFINITY, m1a = -INFINITY, m0b = -INFINITY, m1b = -INFINITY;
    \\#pragma clang loop unroll(full)
    \\for (short j = 0; j < 4; ++j) {
    \\  const int c0 = sg_n0 + int(fn) + int(j);
    \\  const int c1 = c0 + 16;
    \\  if (c0 < nb && c0 < seen0) m0a = idx_bmax2(m0a, acc[j]);
    \\  if (c0 < nb && c0 < seen1) m1a = idx_bmax2(m1a, acc[4 + j]);
    \\  if (c1 < nb && c1 < seen0) m0b = idx_bmax2(m0b, acc[8 + j]);
    \\  if (c1 < nb && c1 < seen1) m1b = idx_bmax2(m1b, acc[12 + j]);
    \\}
    \\m0a = idx_bmax2(m0a, metal::simd_shuffle_xor(m0a, 1));
    \\m1a = idx_bmax2(m1a, metal::simd_shuffle_xor(m1a, 1));
    \\m0b = idx_bmax2(m0b, metal::simd_shuffle_xor(m0b, 1));
    \\m1b = idx_bmax2(m1b, metal::simd_shuffle_xor(m1b, 1));
    \\if (BS >= 16) {
    \\  m0a = idx_bmax2(m0a, metal::simd_shuffle_xor(m0a, 8));
    \\  m1a = idx_bmax2(m1a, metal::simd_shuffle_xor(m1a, 8));
    \\  m0b = idx_bmax2(m0b, metal::simd_shuffle_xor(m0b, 8));
    \\  m1b = idx_bmax2(m1b, metal::simd_shuffle_xor(m1b, 8));
    \\}
    \\if (BS == 32) {
    \\  m0a = idx_bmax2(m0a, m0b);
    \\  m1a = idx_bmax2(m1a, m1b);
    \\}
    \\if (BS == 8) {
    \\  if ((lane & 1) == 0) {
    \\    const int bk0 = (sg_n0 + int(fn)) / 8;
    \\    const int bk1 = bk0 + 2;
    \\    if (r0 < L) {
    \\      if (bk0 < n_blocks) bmax[size_t(r0) * n_blocks + bk0] = (PIN_LAST && seen0 > 0 && bk0 == (seen0 - 1) / 8) ? INFINITY : m0a;
    \\      if (bk1 < n_blocks) bmax[size_t(r0) * n_blocks + bk1] = (PIN_LAST && seen0 > 0 && bk1 == (seen0 - 1) / 8) ? INFINITY : m0b;
    \\    }
    \\    if (r1 < L) {
    \\      if (bk0 < n_blocks) bmax[size_t(r1) * n_blocks + bk0] = (PIN_LAST && seen1 > 0 && bk0 == (seen1 - 1) / 8) ? INFINITY : m1a;
    \\      if (bk1 < n_blocks) bmax[size_t(r1) * n_blocks + bk1] = (PIN_LAST && seen1 > 0 && bk1 == (seen1 - 1) / 8) ? INFINITY : m1b;
    \\    }
    \\  }
    \\} else if (BS == 16) {
    \\  if ((lane & 9) == 0) {
    \\    const int bk0 = sg_n0 / 16;
    \\    const int bk1 = bk0 + 1;
    \\    if (r0 < L) {
    \\      if (bk0 < n_blocks) bmax[size_t(r0) * n_blocks + bk0] = (PIN_LAST && seen0 > 0 && bk0 == (seen0 - 1) / 16) ? INFINITY : m0a;
    \\      if (bk1 < n_blocks) bmax[size_t(r0) * n_blocks + bk1] = (PIN_LAST && seen0 > 0 && bk1 == (seen0 - 1) / 16) ? INFINITY : m0b;
    \\    }
    \\    if (r1 < L) {
    \\      if (bk0 < n_blocks) bmax[size_t(r1) * n_blocks + bk0] = (PIN_LAST && seen1 > 0 && bk0 == (seen1 - 1) / 16) ? INFINITY : m1a;
    \\      if (bk1 < n_blocks) bmax[size_t(r1) * n_blocks + bk1] = (PIN_LAST && seen1 > 0 && bk1 == (seen1 - 1) / 16) ? INFINITY : m1b;
    \\    }
    \\  }
    \\} else {
    \\  if ((lane & 9) == 0) {
    \\    const int bk0 = sg_n0 / 32;
    \\    if (r0 < L && bk0 < n_blocks) bmax[size_t(r0) * n_blocks + bk0] = (PIN_LAST && seen0 > 0 && bk0 == (seen0 - 1) / 32) ? INFINITY : m0a;
    \\    if (r1 < L && bk0 < n_blocks) bmax[size_t(r1) * n_blocks + bk0] = (PIN_LAST && seen1 > 0 && bk0 == (seen1 - 1) / 32) ? INFINITY : m1a;
    \\  }
    \\}
;

const IDX_NAX_GATHER_TOP: [:0]const u8 =
    \\constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    \\    16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    \\mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\const uint lane = thread_index_in_simdgroup;
    \\const int R = w_shape[0];
    \\const int ratio = meta[1];
    \\const int kb = cand_shape[1];
    \\const int ncols = kb * BS;
    \\const int row = int(threadgroup_position_in_grid.x) % R;
    \\const int nranges = int(threadgroups_per_grid.x) / R;
    \\const int nstreams = int(threads_per_threadgroup.x / 32);
    \\const ushort qid = ushort(lane >> 2);
    \\const ushort fm = (qid & 4) | ((ushort(lane) >> 1) & 3);
    \\const ushort f4 = (qid & 2) | (ushort(lane) & 1);
    \\constexpr int KSTEPS = IHD / 16;
    \\constexpr int C8 = IHD / 32;
    \\auto ct_a = op.template get_left_input_cooperative_tensor<T, T, float>();
    \\auto ct_b0 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b1 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b2 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b3 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b4 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b5 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b6 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_b7 = op.template get_right_input_cooperative_tensor<T, T, float>();
    \\auto ct_c = op.template get_destination_cooperative_tensor<
    \\    metal::remove_addrspace_t<decltype(ct_a)>,
    \\    metal::remove_addrspace_t<decltype(ct_b0)>, float>();
    \\#define LOAD_B(KS, CT) if ((KS) < KSTEPS) { \
    \\  const int d0_ = ((KS) >> 1) * 32 + int(f4) * 8 + ((KS) & 1) * 4; \
    \\  _Pragma("unroll") for (ushort tn = 0; tn < 2; ++tn) _Pragma("unroll") for (ushort i = 0; i < 2; ++i) { \
    \\    const int h_ = int(tn) * 16 + int(fm) + int(i) * 8; \
    \\    const metal::vec<float, 4> v = h_ < IH ? metal::vec<float, 4>(*(const device metal::vec<QT, 4>*)(q + (size_t(row) * IH + h_) * IHD + d0_)) : metal::vec<float, 4>(0.0f); \
    \\    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) CT[tn * 8 + i * 4 + j] = T(v[j]); \
    \\  } \
    \\}
    \\LOAD_B(0, ct_b0) LOAD_B(1, ct_b1) LOAD_B(2, ct_b2) LOAD_B(3, ct_b3)
    \\LOAD_B(4, ct_b4) LOAD_B(5, ct_b5) LOAD_B(6, ct_b6) LOAD_B(7, ct_b7)
    \\#undef LOAD_B
    \\float wv[8];
    \\#pragma clang loop unroll(full)
    \\for (ushort tn = 0; tn < 2; ++tn)
    \\  #pragma clang loop unroll(full)
    \\  for (ushort j = 0; j < 4; ++j) {
    \\    const int h_ = int(tn) * 16 + int(f4) * 4 + int(j);
    \\    wv[tn * 4 + j] = h_ < IH ? float(w[size_t(row) * IH + h_]) * wsc[0] : 0.0f;
    \\  }
    \\const int seen = metal::min((meta[0] + row + 1) / ratio, nkeys);
    \\device float* out_row = scores + size_t(row) * ncols;
    \\const int ntiles = (ncols + 15) / 16;
    \\const int stride = nranges * nstreams;
    \\for (int tile = (int(threadgroup_position_in_grid.x) / R) * nstreams + int(simdgroup_index_in_threadgroup); tile < ntiles; tile += stride) {
    \\  int n0, n1;
    \\  bool v0, v1;
    \\  if (BS == 8) {
    \\    const int c0 = cand[size_t(row) * kb + tile * 2];
    \\    const int c1 = (tile * 2 + 1 < kb) ? cand[size_t(row) * kb + tile * 2 + 1] : -1;
    \\    n0 = c0 * 8 + int(fm);
    \\    n1 = c1 * 8 + int(fm);
    \\    v0 = c0 >= 0;
    \\    v1 = c1 >= 0;
    \\  } else if (BS == 16) {
    \\    const int c0 = cand[size_t(row) * kb + tile];
    \\    n0 = c0 * 16 + int(fm);
    \\    n1 = n0 + 8;
    \\    v0 = c0 >= 0;
    \\    v1 = v0;
    \\  } else {
    \\    const int c0 = cand[size_t(row) * kb + (tile >> 1)];
    \\    const int sub = (tile & 1) * 16;
    \\    n0 = c0 * BS + sub + int(fm);
    \\    n1 = n0 + 8;
    \\    v0 = c0 >= 0;
    \\    v1 = v0;
    \\  }
    \\  const int safe_n0 = metal::max(0, metal::min(n0, nkeys - 1));
    \\  const int safe_n1 = metal::max(0, metal::min(n1, nkeys - 1));
;

const IDX_NAX_GATHER_LOAD_PLAIN: [:0]const u8 =
    \\  const device metal::vec<T, 8>* k0 = (const device metal::vec<T, 8>*)(keys + size_t(safe_n0) * IHD + int(f4) * 8);
    \\  const device metal::vec<T, 8>* k1 = (const device metal::vec<T, 8>*)(keys + size_t(safe_n1) * IHD + int(f4) * 8);
    \\  metal::vec<T, 8> a0[C8], a1[C8];
    \\  #pragma clang loop unroll(full)
    \\  for (ushort c = 0; c < C8; ++c) {
    \\    a0[c] = v0 ? k0[c * 4] : metal::vec<T, 8>(T(0));
    \\    a1[c] = v1 ? k1[c * 4] : metal::vec<T, 8>(T(0));
    \\  }
;

const IDX_NAX_GATHER_LOAD_FP4: [:0]const u8 =
    \\  const device uint* kc0 = (const device uint*)codes + size_t(safe_n0) * (IHD / 8) + int(f4);
    \\  const device uint* kc1 = (const device uint*)codes + size_t(safe_n1) * (IHD / 8) + int(f4);
    \\  const device uchar* ks0 = scales + size_t(safe_n0) * C8;
    \\  const device uchar* ks1 = scales + size_t(safe_n1) * C8;
    \\  metal::vec<T, 8> a0[C8], a1[C8];
    \\  #pragma clang loop unroll(full)
    \\  for (ushort c = 0; c < C8; ++c) {
    \\    if (v0) {
    \\      const uint p = kc0[c * 4];
    \\      const float sc = as_type<float>(metal::clamp(uint(ks0[c]), 1u, 254u) << 23);
    \\      #pragma clang loop unroll(full)
    \\      for (uint j = 0; j < 8u; ++j) {
    \\        const uint code = (p >> (j * 4u)) & 15u;
    \\        const float fv = DSV41_IDX_E2M1[code & 7u] * sc;
    \\        a0[c][j] = T((code & 8u) ? -fv : fv);
    \\      }
    \\    } else {
    \\      a0[c] = metal::vec<T, 8>(T(0));
    \\    }
    \\    if (v1) {
    \\      const uint p = kc1[c * 4];
    \\      const float sc = as_type<float>(metal::clamp(uint(ks1[c]), 1u, 254u) << 23);
    \\      #pragma clang loop unroll(full)
    \\      for (uint j = 0; j < 8u; ++j) {
    \\        const uint code = (p >> (j * 4u)) & 15u;
    \\        const float fv = DSV41_IDX_E2M1[code & 7u] * sc;
    \\        a1[c][j] = T((code & 8u) ? -fv : fv);
    \\      }
    \\    } else {
    \\      a1[c] = metal::vec<T, 8>(T(0));
    \\    }
    \\  }
;

const IDX_NAX_GATHER_BOT: [:0]const u8 =
    \\  #pragma clang loop unroll(full)
    \\  for (short e = 0; e < 16; ++e) ct_c[e] = 0.0f;
    \\#define STEP(KS, CT_B) if ((KS) < KSTEPS) { \
    \\    _Pragma("unroll") for (ushort j = 0; j < 4; ++j) { \
    \\      ct_a[j] = a0[(KS) >> 1][((KS) & 1) * 4 + j]; \
    \\      ct_a[4 + j] = a1[(KS) >> 1][((KS) & 1) * 4 + j]; \
    \\    } \
    \\    op.run(ct_a, CT_B, ct_c); \
    \\  }
    \\  STEP(0, ct_b0) STEP(1, ct_b1) STEP(2, ct_b2) STEP(3, ct_b3)
    \\  STEP(4, ct_b4) STEP(5, ct_b5) STEP(6, ct_b6) STEP(7, ct_b7)
    \\#undef STEP
    \\  float p0[8], p1[8];
    \\  #pragma clang loop unroll(full)
    \\  for (ushort tn = 0; tn < 2; ++tn)
    \\    #pragma clang loop unroll(full)
    \\    for (ushort j = 0; j < 4; ++j) {
    \\      const float d0 = ct_c[tn * 8 + j], d1 = ct_c[tn * 8 + 4 + j];
    \\      const float wt = wv[tn * 4 + j];
    \\      p0[tn * 4 + j] = wt * ((d0 < 0.0f) ? 0.0f : d0);
    \\      p1[tn * 4 + j] = wt * ((d1 < 0.0f) ? 0.0f : d1);
    \\    }
    \\  float s0 = 0.0f, s1 = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (ushort tn = 0; tn < 2; ++tn) {
    \\    #pragma clang loop unroll(full)
    \\    for (ushort src = 0; src < 4; ++src) {
    \\      const ushort xmask = (src & 1) | ((src & 2) << 2);
    \\      #pragma clang loop unroll(full)
    \\      for (ushort j = 0; j < 4; ++j) {
    \\        const float q0 = xmask ? metal::simd_shuffle_xor(p0[tn * 4 + j], xmask) : p0[tn * 4 + j];
    \\        const float q1 = xmask ? metal::simd_shuffle_xor(p1[tn * 4 + j], xmask) : p1[tn * 4 + j];
    \\        if (int(tn) * 16 + int(src) * 4 + int(j) < IH) {
    \\          s0 += q0;
    \\          s1 += q1;
    \\        }
    \\      }
    \\    }
    \\  }
    \\  if (f4 == 0) {
    \\    const int m0 = tile * 16 + int(fm), m1 = m0 + 8;
    \\    if (m0 < ncols) out_row[m0] = (v0 && n0 >= 0 && n0 < seen) ? s0 : -INFINITY;
    \\    if (m1 < ncols) out_row[m1] = (v1 && n1 >= 0 && n1 < seen) ? s1 : -INFINITY;
    \\  }
    \\}
;

const DEC_TOPK_HEADER: [:0]const u8 =
    \\inline uint dec_key(float f) {
    \\  const uint u = as_type<uint>(f);
    \\  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
    \\template <int TG, int N>
    \\inline void dec_radix(const thread uint (&kk)[N], uint krem, threadgroup atomic_uint* hist, uint tid, uint lane, thread uint& thr, thread uint& rem) {
    \\  threadgroup atomic_uint* sel = hist + 512;
    \\  for (uint b = tid; b < 256; b += TG) metal::atomic_store_explicit(hist + b, 0u, metal::memory_order_relaxed);
    \\  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  uint prefix = 0;
    \\  for (uint pass = 0; pass < 4; ++pass) {
    \\    const uint shift = 24 - 8 * pass;
    \\    const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    \\    threadgroup atomic_uint* h = hist + (pass & 1) * 256;
    \\    threadgroup atomic_uint* next = hist + ((pass + 1) & 1) * 256;
    \\    for (uint b = tid; b < 256; b += TG) metal::atomic_store_explicit(next + b, 0u, metal::memory_order_relaxed);
    \\    #pragma clang loop unroll(full)
    \\    for (int e = 0; e < N; ++e)
    \\      if ((kk[e] & hmask) == prefix) metal::atomic_fetch_add_explicit(h + ((kk[e] >> shift) & 255u), 1u, metal::memory_order_relaxed);
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (tid < 32) {
    \\      uint c[8], tot = 0;
    \\      #pragma clang loop unroll(full)
    \\      for (uint j = 0; j < 8; ++j) { c[j] = metal::atomic_load_explicit(h + (255u - (lane * 8 + j)), metal::memory_order_relaxed); tot += c[j]; }
    \\      const uint before = metal::simd_prefix_exclusive_sum(tot);
    \\      uint pk = 0;
    \\      if (before < krem && krem <= before + tot) {
    \\        uint acc = before, j = 0;
    \\        for (; j < 7; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
    \\        pk = (acc << 8) | (255u - (lane * 8 + j));
    \\      }
    \\      pk = metal::simd_max(pk);
    \\      if (lane == 0) metal::atomic_store_explicit(sel + (pass & 1), pk, metal::memory_order_relaxed);
    \\    }
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    const uint pk = metal::atomic_load_explicit(sel + (pass & 1), metal::memory_order_relaxed);
    \\    prefix |= (pk & 255u) << shift;
    \\    krem -= pk >> 8;
    \\  }
    \\  thr = prefix;
    \\  rem = krem;
    \\}
    \\template <int TG, int N>
    \\inline void dec_scan(const thread uint (&kk)[N], uint above, uint eq, bool ties, uint rem, threadgroup uint* part, uint sg, uint lane,
    \\                     thread uint (&bef)[N], thread uint& kept) {
    \\  uint run = 0;
    \\  #pragma clang loop unroll(full)
    \\  for (int e = 0; e < N; ++e) {
    \\    const uint v = (uint(kk[e] > above) << 16) | uint(ties && kk[e] == eq);
    \\    bef[e] = run + metal::simd_prefix_exclusive_sum(v);
    \\    run += metal::simd_sum(v);
    \\  }
    \\  if (lane == 0) part[sg] = run;
    \\  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  const uint pv = lane < TG / 32 ? part[lane] : 0u;
    \\  const uint off = metal::simd_shuffle(metal::simd_prefix_exclusive_sum(pv), sg), all = metal::simd_sum(pv);
    \\  #pragma clang loop unroll(full)
    \\  for (int e = 0; e < N; ++e) bef[e] += off;
    \\  kept = (all >> 16) + metal::min(all & 0xFFFFu, rem);
    \\}
;

const DEC_TOPK_SOURCE: [:0]const u8 =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int row = threadgroup_position_in_grid.y, n = scores_shape[1];
    \\const int K = meta[2];
    \\const int lens = (meta[0] + row + 1) / meta[1];
    \\const auto sr = scores + size_t(row) * n;
    \\const auto cr = cand + size_t(row) * KB;
    \\device int* out = picks + size_t(row) * K;
    \\threadgroup atomic_uint hist[2 * 256 + 2];
    \\threadgroup uint part[TG / 32];
    \\uint kk[EPT];
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < EPT; ++e) {
    \\  const int s = (int(sg) * EPT + e) * 32 + int(lane);
    \\  kk[e] = s < n && (!GATHER || cr[s / BS] >= 0) ? dec_key(sr[s]) : 0u;
    \\}
    \\const uint want = metal::min(uint(K), uint(n));
    \\uint thr, rem, kept;
    \\dec_radix<TG, EPT>(kk, want, hist, tid, lane, thr, rem);
    \\uint bef[EPT];
    \\dec_scan<TG, EPT>(kk, thr, thr, true, rem, part, sg, lane, bef, kept);
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < EPT; ++e) {
    \\  const bool gt = kk[e] > thr, eq = kk[e] == thr;
    \\  const uint g = bef[e] >> 16, q = bef[e] & 0xFFFFu;
    \\  if (gt || (eq && q < rem)) {
    \\    const int s = (int(sg) * EPT + e) * 32 + int(lane);
    \\    int id = s;
    \\    if (GATHER) { const int b = cr[s / BS]; id = b >= 0 ? b * BS + s % BS : -1; }
    \\    out[g + (gt ? metal::min(q, rem) : q)] = id >= 0 && id < lens ? id : -1;
    \\  }
    \\}
    \\for (uint p = want + tid; p < uint(K); p += TG) out[p] = -1;
;

const SOURCE_SCORES_PLAIN: [:0]const u8 = "const int nb = keys_shape[0];\n" ++ IDX_NAX_SHARED_TOP ++ "\n" ++ IDX_NAX_EARLY_SCORES ++ "\n" ++ IDX_NAX_KT_PLAIN ++ "\n" ++ IDX_NAX_MATMUL ++ "\n" ++ IDX_NAX_TAIL_SCORES ++ "\n";
const SOURCE_SCORES_FP4: [:0]const u8 = "const int nb = codes_shape[0];\n" ++ IDX_NAX_SHARED_TOP ++ "\n" ++ IDX_NAX_EARLY_SCORES ++ "\n" ++ IDX_NAX_KT_FP4 ++ "\n" ++ IDX_NAX_MATMUL ++ "\n" ++ IDX_NAX_TAIL_SCORES ++ "\n";
const SOURCE_BMAX_PLAIN: [:0]const u8 = "const int nb = keys_shape[0];\n" ++ IDX_NAX_SHARED_TOP ++ "\n" ++ IDX_NAX_EARLY_BMAX ++ "\n" ++ IDX_NAX_KT_PLAIN ++ "\n" ++ IDX_NAX_MATMUL ++ "\n" ++ IDX_NAX_TAIL_BMAX ++ "\n";
const SOURCE_BMAX_FP4: [:0]const u8 = "const int nb = codes_shape[0];\n" ++ IDX_NAX_SHARED_TOP ++ "\n" ++ IDX_NAX_EARLY_BMAX ++ "\n" ++ IDX_NAX_KT_FP4 ++ "\n" ++ IDX_NAX_MATMUL ++ "\n" ++ IDX_NAX_TAIL_BMAX ++ "\n";
const SOURCE_GATHER_PLAIN: [:0]const u8 = "const int nkeys = keys_shape[0];\n" ++ IDX_NAX_GATHER_TOP ++ "\n" ++ IDX_NAX_GATHER_LOAD_PLAIN ++ "\n" ++ IDX_NAX_GATHER_BOT ++ "\n";
const SOURCE_GATHER_FP4: [:0]const u8 = "const int nkeys = codes_shape[0];\n" ++ IDX_NAX_GATHER_TOP ++ "\n" ++ IDX_NAX_GATHER_LOAD_FP4 ++ "\n" ++ IDX_NAX_GATHER_BOT ++ "\n";

var def_scores_plain = kernels.Def{ .name = "sushi_ds41_index_nax", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = SOURCE_SCORES_PLAIN, .header = IDX_NAX_HEADER };
var def_scores_fp4 = kernels.Def{ .name = "sushi_ds41_index_nax_fp4", .ins = &.{ "q", "codes", "scales", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = SOURCE_SCORES_FP4, .header = IDX_NAX_HEADER };
var def_bmax_plain = kernels.Def{ .name = "sushi_ds41_idx_nax_bmax", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"bmax"}, .source = SOURCE_BMAX_PLAIN, .header = IDX_NAX_HEADER };
var def_bmax_fp4 = kernels.Def{ .name = "sushi_ds41_idx_nax_bmax_fp4", .ins = &.{ "q", "codes", "scales", "w", "meta", "wsc" }, .outs = &.{"bmax"}, .source = SOURCE_BMAX_FP4, .header = IDX_NAX_HEADER };
var def_gather_plain = kernels.Def{ .name = "sushi_ds41_idx_nax_gather", .ins = &.{ "q", "keys", "w", "meta", "wsc", "cand" }, .outs = &.{"scores"}, .source = SOURCE_GATHER_PLAIN, .header = IDX_NAX_HEADER };
var def_gather_fp4 = kernels.Def{ .name = "sushi_ds41_idx_nax_gather_fp4", .ins = &.{ "q", "codes", "scales", "w", "meta", "wsc", "cand" }, .outs = &.{"scores"}, .source = SOURCE_GATHER_FP4, .header = IDX_NAX_HEADER };
var def_dec_topk = kernels.Def{ .name = "sushi_ds41_dec_topk", .ins = &.{ "scores", "meta", "cand" }, .outs = &.{"picks"}, .source = DEC_TOPK_SOURCE, .header = DEC_TOPK_HEADER };

/// The most elements `dec_topk` selects among: the scan counts ties and keys above in 16 bits each.
pub const max_elements: usize = 65535;
const max_ept: usize = 64;
const narrow_rows: usize = @intCast(cfg_mod.narrow_rows);
pub var dec_topk_tg: usize = 1024;
var dec_topk_probed = false;

/// Tile and dispatch geometry for `indexScoresNax` and `indexScoresNaxBmax`.
pub fn idxNaxGeom(L: usize, nb: usize, ihd: usize) struct { bm: usize, bn: usize, tgn: usize, dk: usize, gx: usize, gy: usize } {
    const bm: usize = if (L <= 16) 16 else 32;
    const bn: usize = 32;
    const tgn: usize = (bm / 16) * (bn / 32) * 32;
    const dk: usize = if (ihd % 64 == 0) 64 else if (ihd % 32 == 0) 32 else 16;
    return .{ .bm = bm, .bn = bn, .tgn = tgn, .dk = dk, .gx = (L + bm - 1) / bm * tgn, .gy = (nb + bn - 1) / bn };
}

pub fn idxNaxGatherRanges(L: usize, ncols: usize) usize {
    const ntiles = (ncols + 15) / 16;
    const max_ranges: usize = if (L <= 32) 32 else 8;
    return @min(max_ranges, @max(@as(usize, 1), (ntiles + 3) / 4));
}

/// Whether `indexScoresNax` serves `ih` heads of `ihd` channels in `dt` on this GPU (up to 128 heads, 16-channel steps,
/// and the M5 matrix units present).
pub fn fits(ih: usize, ihd: usize, dt: mlx.mlx_dtype) bool {
    if (dt != .bfloat16 and dt != .float16) return false;
    if (ih == 0 or ih > 128 or ihd % 16 != 0 or ihd > 128) return false;
    return nax_gate.verifyQmmNaxAvailable();
}

/// Ensures `dec_topk_tg` fits this GPU (halving on threadgroup size refusal), once per process.
pub fn probeDecTopk(gpa: std.mem.Allocator, s: S) !void {
    if (dec_topk_probed or !mlx.streamIsGpu(s)) return;
    defer mlx.expected_refusal = null;
    while (true) {
        mlx.expected_refusal = if (dec_topk_tg > 64) "Thread group size" else null;
        probeDecTopkOnce(gpa, s) catch |e| {
            if (e != error.MlxError or dec_topk_tg <= 64 or !mlx.takeErrorIf("Thread group size")) return e;
            dec_topk_tg /= 2;
            continue;
        };
        dec_topk_probed = true;
        return;
    }
}

fn probeDecTopkOnce(gpa: std.mem.Allocator, s: S) !void {
    var sc = Scope.plain(gpa, s);
    defer sc.deinit();
    const x = try sc.zeros(&.{ 1, ci(dec_topk_tg * max_ept) }, .float32);
    const meta = try sc.ints(&.{ ci(dec_topk_tg * max_ept - 1), 1, 512 }, &.{3});
    const dummy = try sc.ints(&.{0}, &.{1});
    var o: [1]A = undefined;
    try kernels.launch(&sc, &def_dec_topk, &.{ x, meta, dummy }, &.{.{ .shape = &.{ 1, 512 }, .dt = .int32 }}, .{ dec_topk_tg, 1, 1 }, .{ dec_topk_tg, 1, 1 }, &.{
        .{ .name = "TG", .v = ci(dec_topk_tg) }, .{ .name = "EPT", .v = max_ept }, .{ .name = "BS", .v = 8 }, .{ .name = "KB", .v = 1 }, .{ .name = "GATHER", .v = 0 },
    }, &.{}, &o);
    try sc_mod.evalAll(&o);
}

fn launchIdxNax2d(
    sc: *Scope,
    def: *kernels.Def,
    q: A,
    keys_or_codes: A,
    scales: ?A,
    dt: mlx.mlx_dtype,
    w: A,
    wscale: f32,
    p0: usize,
    ratio: usize,
    key0: usize,
    out_cols: usize,
    bs: ?usize,
    pin_last: bool,
) !A {
    const L = dim(q, 0);
    const nb = dim(keys_or_codes, 0);
    const ihd = dim(q, 2);
    const geom = idxNaxGeom(L, nb, ihd);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio), ci(key0) }, &.{3});
    const q_in = try sc.contiguous(if (mlx.mlx_array_dtype(q) == dt) q else try sc.astype(q, dt));
    const k_c = try sc.contiguous(keys_or_codes);
    const w_c = try sc.contiguous(w);
    const wsc = try kernels.f32Const(sc, &.{wscale});
    var o: [1]A = undefined;
    if (bs) |b| {
        if (scales) |sc_a| {
            const s_c = try sc.contiguous(sc_a);
            try kernels.launch(sc, def, &.{ q_in, k_c, s_c, w_c, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(out_cols) }, .dt = .float32 }}, .{ geom.gx, geom.gy, 1 }, .{ geom.tgn, 1, 1 }, &.{
                .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(ihd) }, .{ .name = "BM", .v = ci(geom.bm) }, .{ .name = "BN", .v = ci(geom.bn) }, .{ .name = "DK", .v = ci(geom.dk) }, .{ .name = "BS", .v = ci(b) }, .{ .name = "PIN_LAST", .v = @intFromBool(pin_last) },
            }, &.{.{ .name = "T", .dt = dt }}, &o);
        } else {
            try kernels.launch(sc, def, &.{ q_in, k_c, w_c, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(out_cols) }, .dt = .float32 }}, .{ geom.gx, geom.gy, 1 }, .{ geom.tgn, 1, 1 }, &.{
                .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(ihd) }, .{ .name = "BM", .v = ci(geom.bm) }, .{ .name = "BN", .v = ci(geom.bn) }, .{ .name = "DK", .v = ci(geom.dk) }, .{ .name = "BS", .v = ci(b) }, .{ .name = "PIN_LAST", .v = @intFromBool(pin_last) },
            }, &.{.{ .name = "T", .dt = dt }}, &o);
        }
    } else if (scales) |sc_a| {
        const s_c = try sc.contiguous(sc_a);
        try kernels.launch(sc, def, &.{ q_in, k_c, s_c, w_c, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(out_cols) }, .dt = .float32 }}, .{ geom.gx, geom.gy, 1 }, .{ geom.tgn, 1, 1 }, &.{
            .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(ihd) }, .{ .name = "BM", .v = ci(geom.bm) }, .{ .name = "BN", .v = ci(geom.bn) }, .{ .name = "DK", .v = ci(geom.dk) },
        }, &.{.{ .name = "T", .dt = dt }}, &o);
    } else {
        try kernels.launch(sc, def, &.{ q_in, k_c, w_c, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(out_cols) }, .dt = .float32 }}, .{ geom.gx, geom.gy, 1 }, .{ geom.tgn, 1, 1 }, &.{
            .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(ihd) }, .{ .name = "BM", .v = ci(geom.bm) }, .{ .name = "BN", .v = ci(geom.bn) }, .{ .name = "DK", .v = ci(geom.dk) },
        }, &.{.{ .name = "T", .dt = dt }}, &o);
    }
    return o[0];
}

/// `indexScoresTile`'s scores `[L, nb]` (f32, -inf past what a row sees) of `q` `[L, IH, IHD]` against `keys`
/// `[nb, IHD]` from `key0` on, with head weights `w` `[L, IH]` times `wscale`; `q` is cast to the keys' dtype.
pub fn indexScoresNax(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize, key0: usize) !A {
    const dt = mlx.mlx_array_dtype(keys);
    std.debug.assert(key0 % 32 == 0 and fits(dim(q, 1), dim(q, 2), dt));
    return launchIdxNax2d(sc, &def_scores_plain, q, keys, null, dt, w, wscale, p0, ratio, key0, dim(keys, 0), null, false);
}

/// `indexScoresNax` reading packed FP4 keys (`codes` `[nb, IHD/2]`, `scales` `[nb, IHD/32]`, `.fp4_ue8m0`), unpacked
/// to `dt` (`bfloat16`/`float16`) in threadgroup memory.
pub fn indexScoresNaxFp4(sc: *Scope, q: A, codes: A, scales: A, dt: mlx.mlx_dtype, w: A, wscale: f32, p0: usize, ratio: usize, key0: usize) !A {
    std.debug.assert(key0 % 32 == 0 and fits(dim(q, 1), dim(q, 2), dt));
    return launchIdxNax2d(sc, &def_scores_fp4, q, codes, scales, dt, w, wscale, p0, ratio, key0, dim(codes, 0), null, false);
}

/// Online block-max reduction of index scores over blocks of `bs` (`8, 16, 32`) keys: `[L, ceil(nb / bs)]` f32.
/// With `pin_last`, each row's newest visible block `(seen - 1) / bs` is pinned to `+inf`.
pub fn indexScoresNaxBmax(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize, bs: usize, pin_last: bool) !A {
    const dt = mlx.mlx_array_dtype(keys);
    const n_blocks = (dim(keys, 0) + bs - 1) / bs;
    return launchIdxNax2d(sc, &def_bmax_plain, q, keys, null, dt, w, wscale, p0, ratio, 0, n_blocks, bs, pin_last);
}

/// `indexScoresNaxBmax` reading packed FP4 keys (`codes`, `scales`, `.fp4_ue8m0`, group 32).
pub fn indexScoresNaxBmaxFp4(sc: *Scope, q: A, codes: A, scales: A, dt: mlx.mlx_dtype, w: A, wscale: f32, p0: usize, ratio: usize, bs: usize, pin_last: bool) !A {
    const n_blocks = (dim(codes, 0) + bs - 1) / bs;
    return launchIdxNax2d(sc, &def_bmax_fp4, q, codes, scales, dt, w, wscale, p0, ratio, 0, n_blocks, bs, pin_last);
}

fn launchGather(
    sc: *Scope,
    def: *kernels.Def,
    q: A,
    keys_or_codes: A,
    scales: ?A,
    dt: mlx.mlx_dtype,
    w: A,
    cand: A,
    wscale: f32,
    p0: usize,
    ratio: usize,
    bs: usize,
) !A {
    const L = dim(q, 0);
    const kb = dim(cand, 1);
    const ncols = kb * bs;
    const nranges = idxNaxGatherRanges(L, ncols);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio) }, &.{2});
    const q_c = try sc.contiguous(q);
    const k_c = try sc.contiguous(keys_or_codes);
    const w_c = try sc.contiguous(w);
    const cand_c = try sc.contiguous(cand);
    const wsc = try kernels.f32Const(sc, &.{wscale});
    const q_dt = mlx.mlx_array_dtype(q_c);
    var o: [1]A = undefined;
    if (scales) |sc_a| {
        const s_c = try sc.contiguous(sc_a);
        try kernels.launch(sc, def, &.{ q_c, k_c, s_c, w_c, meta, wsc, cand_c }, &.{.{ .shape = &.{ ci(L), ci(ncols) }, .dt = .float32 }}, .{ nranges * L * 128, 1, 1 }, .{ 128, 1, 1 }, &.{
            .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) }, .{ .name = "BS", .v = ci(bs) },
        }, &.{ .{ .name = "T", .dt = dt }, .{ .name = "QT", .dt = q_dt } }, &o);
    } else {
        try kernels.launch(sc, def, &.{ q_c, k_c, w_c, meta, wsc, cand_c }, &.{.{ .shape = &.{ ci(L), ci(ncols) }, .dt = .float32 }}, .{ nranges * L * 128, 1, 1 }, .{ 128, 1, 1 }, &.{
            .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) }, .{ .name = "BS", .v = ci(bs) },
        }, &.{ .{ .name = "T", .dt = dt }, .{ .name = "QT", .dt = q_dt } }, &o);
    }
    return o[0];
}

/// Register-staged NAX gather scorer over candidate blocks `cand` (`[L, kb]` int32): `[L, kb * bs]` f32.
pub fn indexScoresNaxGather(sc: *Scope, q: A, keys: A, w: A, cand: A, wscale: f32, p0: usize, ratio: usize, bs: usize) !A {
    return launchGather(sc, &def_gather_plain, q, keys, null, mlx.mlx_array_dtype(keys), w, cand, wscale, p0, ratio, bs);
}

/// `indexScoresNaxGather` reading packed FP4 keys (`codes`, `scales`, `.fp4_ue8m0`, group 32).
pub fn indexScoresNaxGatherFp4(sc: *Scope, q: A, codes: A, scales: A, dt: mlx.mlx_dtype, w: A, cand: A, wscale: f32, p0: usize, ratio: usize, bs: usize) !A {
    return launchGather(sc, &def_gather_fp4, q, codes, scales, dt, w, cand, wscale, p0, ratio, bs);
}

fn pow2Ceil(x: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(x, 1)) catch unreachable;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Register-resident top-`k` picks of `[R, n]` f32 `scores` (`dec_topk`), optionally mapping column positions through
/// `cand` (`[R, kb]` int32, `bs` keys per block) and masking picks `>= (p0 + r + 1) / ratio` to `-1`. Returns `null`
/// when `n` exceeds `max_elements` or `max_ept * dec_topk_tg`.
pub fn pickRows(sc: *Scope, scores: A, cand: ?A, p0: usize, ratio: usize, k: usize, bs: usize, kb: usize) !?A {
    if (!mlx.streamIsGpu(sc.s) or mlx.mlx_array_dtype(scores) != .float32) return null;
    const sh = mlx.getShape(scores);
    if (sh.len != 2 or sh[0] < 1 or k == 0 or k > 1024 or bs == 0 or kb == 0) return null;
    const rows: usize = @intCast(sh[0]);
    const n: usize = @intCast(sh[1]);
    if (n == 0 or n > max_elements) return null;
    if (cand) |c| {
        const csh = mlx.getShape(c);
        if (csh.len != 2 or csh[0] != sh[0] or csh[1] != ci(kb) or n != kb * bs) return null;
    }
    try probeDecTopk(sc.a, sc.s);
    const tg = dec_topk_tg;
    const ept = pow2Ceil(ceilDiv(n, tg));
    if (ept > max_ept) return null;
    const meta = try sc.ints(&.{ ci(p0), ci(ratio), ci(k) }, &.{3});
    const scores_c = try sc.contiguous(scores);
    const cand_c = if (cand) |c| try sc.contiguous(c) else try sc.ints(&.{0}, &.{1});
    var o: [1]A = undefined;
    try kernels.launch(sc, &def_dec_topk, &.{ scores_c, meta, cand_c }, &.{.{ .shape = &.{ ci(rows), ci(k) }, .dt = .int32 }}, .{ tg, rows, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "TG", .v = ci(tg) },
        .{ .name = "EPT", .v = ci(ept) },
        .{ .name = "BS", .v = ci(bs) },
        .{ .name = "KB", .v = ci(kb) },
        .{ .name = "GATHER", .v = @intFromBool(cand != null) },
    }, &.{}, &o);
    return o[0];
}

pub const KeySource = union(enum) {
    plain: A,
    fp4: struct { codes: A, scales: A, dt: mlx.mlx_dtype },

    fn dtype(self: KeySource) mlx.mlx_dtype {
        return switch (self) {
            .plain => |k| mlx.mlx_array_dtype(k),
            .fp4 => |f| f.dt,
        };
    }

    fn count(self: KeySource) usize {
        return switch (self) {
            .plain => |k| dim(k, 0),
            .fp4 => |f| dim(f.codes, 0),
        };
    }
};

fn onlinePlainBs(nb: usize, max_pick: usize) usize {
    if ((nb + 7) / 8 <= max_pick) return 8;
    if ((nb + 15) / 16 <= max_pick) return 16;
    return 32;
}

pub const OnlinePlan = union(enum) {
    post_cand: struct { kb: usize, bs: usize },
    cand_src: struct { kb: usize, bs: usize, n_blocks: usize },
    dense_nax,
    plain: struct { bs: usize, n_blocks: usize, k_blk: usize },
};

/// Whether `indexerOnlineTopk` can serve `(fused, tracing, stream_tile, gpu_stream, L, nb, ih, ihd, k_dt, w_dt, q_dt)`.
pub fn onlineTopkEligible(
    fused: bool,
    tracing: bool,
    stream_tile: usize,
    gpu_stream: bool,
    L: usize,
    nb: usize,
    ih: usize,
    ihd: usize,
    k_dt: mlx.mlx_dtype,
    w_dt: mlx.mlx_dtype,
    q_dt: mlx.mlx_dtype,
) bool {
    if (!fused or tracing or stream_tile != cfg_mod.index_stream_tile) return false;
    if (!gpu_stream or L <= narrow_rows or nb == 0) return false;
    if (!nax_gate.verifyQmmNaxAvailable()) return false;
    if (k_dt != .bfloat16 and k_dt != .float16) return false;
    if (w_dt != .bfloat16 and w_dt != .float16 and w_dt != .float32) return false;
    if (q_dt != .float32 and q_dt != .bfloat16 and q_dt != .float16) return false;
    return ih > 0 and ih <= 32 and ihd % 32 == 0 and ihd <= 128;
}

/// Exact indexer byte bill inside `dsv41_wide_bytes.spanBytes` (`keys = min(nb, index_stream_tile)`), so `onlinePlan`
/// never allocates more than admission billed (`dsv41_config.zig` keeps `forwardCap(2048, 9k..1M)` invariant).
pub fn spanBilledIndexerBytes(L: usize, nb: usize, ih: usize, ihd: usize, k: usize) u64 {
    const keys: u64 = @min(nb, cfg_mod.index_stream_tile);
    var bytes: u64 = @as(u64, L) * (@as(u64, ih) * ihd * 2 * 2 + @as(u64, ih) * ihd * 4 + @as(u64, ih) * 2 + keys * 4);
    bytes += keys * @as(u64, ihd) * 2;
    if (nb > cfg_mod.index_stream_tile) {
        bytes += @as(u64, L) * 12 * @as(u64, k) * 4;
    }
    return bytes;
}

/// Peak byte allocation inside `indexerOnlineTopk`'s span `Scope` for `plan`.
pub fn onlinePlanBytes(plan: OnlinePlan, L: usize, nb: usize, ih: usize, ihd: usize, k: usize) u64 {
    const l64: u64 = L;
    const base: u64 = l64 * (@as(u64, ih) * ihd * 4 + @as(u64, ih) * 2);
    return switch (plan) {
        .dense_nax => base + l64 * (@as(u64, nb) * 4 + @as(u64, k) * 4),
        .post_cand => |p| base + l64 * (@as(u64, p.kb) * p.bs * 4 + @as(u64, k) * 4),
        .cand_src => |p| base + l64 * (@as(u64, p.n_blocks) * 4 + @as(u64, p.kb) * 4 + @as(u64, p.kb) * 5 + @as(u64, p.kb) * p.bs * 4 + @as(u64, k) * 4) + @as(u64, p.kb) * 4 + l64 * 8,
        .plain => |p| base + l64 * (@as(u64, p.n_blocks) * 4 + @as(u64, p.k_blk) * 4 + @as(u64, p.k_blk) * p.bs * 4 + @as(u64, k) * 4),
    };
}

pub fn onlinePlan(
    c: *const cfg_mod.Config,
    fused: bool,
    tracing: bool,
    stream_tile: usize,
    li: usize,
    gpu_stream: bool,
    L: usize,
    nb: usize,
    ih: usize,
    ihd: usize,
    k_dt: mlx.mlx_dtype,
    w_dt: mlx.mlx_dtype,
    q_dt: mlx.mlx_dtype,
    sh_cand: ?A,
) ?OnlinePlan {
    if (!onlineTopkEligible(fused, tracing, stream_tile, gpu_stream, L, nb, ih, ihd, k_dt, w_dt, q_dt)) return null;
    const k = @min(@as(usize, c.index_topk), nb);
    if (k == 0 or k > 1024) return null;
    const max_pick = @min(max_elements, dec_topk_tg * max_ept);

    const cand_src = c.cand_source;
    const bs: usize = c.cand_block_size;
    const kb: usize = c.cand_blocks;
    const valid_bs = bs == 8 or bs == 16 or bs == 32;
    // Pinning the last visible block to +inf consumes one of the kb slots, so exact top-k requires kb > k.
    const cand_src_ok = valid_bs and kb > k and kb * bs >= k and kb * bs <= max_pick;

    var chosen: OnlinePlan = undefined;
    if (cand_src >= 0) {
        const csrc: usize = @intCast(cand_src);
        if (li > csrc) {
            if (sh_cand) |cand| {
                if (cand.ctx != null) {
                    const csh = mlx.getShape(cand);
                    if (mlx.mlx_array_dtype(cand) != .int32 or !valid_bs or csh.len != 2 or csh[0] != ci(L)) return null;
                    const ckb: usize = @intCast(csh[1]);
                    if (ckb == 0 or ckb * bs < k or ckb * bs > max_pick) return null;
                    chosen = .{ .post_cand = .{ .kb = ckb, .bs = bs } };
                    if (onlinePlanBytes(chosen, L, nb, ih, ihd, k) > spanBilledIndexerBytes(L, nb, ih, ihd, k)) return null;
                    return chosen;
                }
            } else {
                const n_blocks_cand = if (bs > 0) (nb + bs - 1) / bs else 0;
                if (bs == 0 or n_blocks_cand > kb) {
                    if (!cand_src_ok) return null;
                    chosen = .{ .post_cand = .{ .kb = kb, .bs = bs } };
                    if (onlinePlanBytes(chosen, L, nb, ih, ihd, k) > spanBilledIndexerBytes(L, nb, ih, ihd, k)) return null;
                    return chosen;
                }
            }
        } else if (li == csrc) {
            const n_blocks_cand = if (bs > 0) (nb + bs - 1) / bs else 0;
            if (bs == 0 or n_blocks_cand > kb) {
                if (!cand_src_ok) return null;
                chosen = .{ .cand_src = .{ .kb = kb, .bs = bs, .n_blocks = n_blocks_cand } };
                if (onlinePlanBytes(chosen, L, nb, ih, ihd, k) > spanBilledIndexerBytes(L, nb, ih, ihd, k)) return null;
                return chosen;
            }
        }
    }

    if (nb <= 16384 and nb <= max_pick) {
        chosen = .dense_nax;
        if (onlinePlanBytes(chosen, L, nb, ih, ihd, k) > spanBilledIndexerBytes(L, nb, ih, ihd, k)) return null;
        return chosen;
    }

    const b_sz = onlinePlainBs(nb, max_pick);
    const n_blocks = (nb + b_sz - 1) / b_sz;
    const k_blk = @min(k, n_blocks);
    if (k_blk * b_sz > max_pick) return null;
    chosen = .{ .plain = .{ .bs = b_sz, .n_blocks = n_blocks, .k_blk = k_blk } };
    if (onlinePlanBytes(chosen, L, nb, ih, ihd, k) > spanBilledIndexerBytes(L, nb, ih, ihd, k)) return null;
    return chosen;
}

fn selectBlocksTopk(sc: *Scope, bmax: A, n_blocks: usize, k_blk: usize, bs: usize, max_pick: usize) !A {
    if (n_blocks <= max_pick) {
        if (try pickRows(sc, bmax, null, n_blocks - 1, 1, k_blk, bs, k_blk)) |bids| return bids;
    }
    return kernels.topkIndices(sc, bmax, k_blk);
}

fn runScoresNax(sc: *Scope, q: A, ks: KeySource, w: A, wscale: f32, p0: usize, r: usize) !A {
    return switch (ks) {
        .plain => |keys| indexScoresNax(sc, q, keys, w, wscale, p0, r, 0),
        .fp4 => |f| indexScoresNaxFp4(sc, q, f.codes, f.scales, f.dt, w, wscale, p0, r, 0),
    };
}

fn runBmax(sc: *Scope, q: A, ks: KeySource, w: A, wscale: f32, p0: usize, r: usize, bs: usize, pin_last: bool) !A {
    return switch (ks) {
        .plain => |keys| indexScoresNaxBmax(sc, q, keys, w, wscale, p0, r, bs, pin_last),
        .fp4 => |f| indexScoresNaxBmaxFp4(sc, q, f.codes, f.scales, f.dt, w, wscale, p0, r, bs, pin_last),
    };
}

fn runGather(sc: *Scope, q: A, ks: KeySource, w: A, cand: A, wscale: f32, p0: usize, r: usize, bs: usize) !A {
    return switch (ks) {
        .plain => |keys| indexScoresNaxGather(sc, q, keys, w, cand, wscale, p0, r, bs),
        .fp4 => |f| indexScoresNaxGatherFp4(sc, q, f.codes, f.scales, f.dt, w, cand, wscale, p0, r, bs),
    };
}

/// Wide-span prefill indexer without `[L, nb]` f32 score materialization: block-max NAX reduction -> block top-K ->
/// register-staged NAX gather -> exact register top-K (`dec_topk`). Post-candidate layers reuse `cand.*` (`int32`)
/// directly to score only `kb * bs` keys. Returns `null` when ineligible so caller falls back cleanly.
pub fn indexerOnlineTopk(
    sc: *Scope,
    c: *const cfg_mod.Config,
    fused: bool,
    tracing: bool,
    stream_tile: usize,
    li: usize,
    ks: KeySource,
    w: A,
    q: A,
    wscale: f32,
    p0: usize,
    r: usize,
    cand: *A,
) !?A {
    const L = dim(q, 0);
    const nb = ks.count();
    const ih = dim(q, 1);
    const ihd = dim(q, 2);
    if (!mlx.streamIsGpu(sc.s)) return null;
    const k_dt = ks.dtype();
    const w_dt = mlx.mlx_array_dtype(w);
    const q_dt = mlx.mlx_array_dtype(q);
    if (!onlineTopkEligible(fused, tracing, stream_tile, true, L, nb, ih, ihd, k_dt, w_dt, q_dt)) return null;
    try probeDecTopk(sc.a, sc.s);
    const plan = onlinePlan(c, fused, tracing, stream_tile, li, true, L, nb, ih, ihd, k_dt, w_dt, q_dt, cand.*) orelse return null;
    const k = @min(@as(usize, c.index_topk), nb);
    const max_pick = @min(max_elements, dec_topk_tg * max_ept);

    if (c.cand_source >= 0 and li == @as(usize, @intCast(c.cand_source)) and plan != .cand_src) {
        if (cand.ctx != null) _ = mlx.mlx_array_free(cand.*);
        cand.* = none;
    }

    switch (plan) {
        .post_cand => |p| {
            const scores_g = try runGather(sc, q, ks, w, cand.*, wscale, p0, r, p.bs);
            return (try pickRows(sc, scores_g, cand.*, p0, r, k, p.bs, p.kb)) orelse return error.Dsv41CandidateIds;
        },
        .dense_nax => {
            const scores = try runScoresNax(sc, q, ks, w, wscale, p0, r);
            return (try pickRows(sc, scores, null, p0, r, k, 1, k)) orelse return null;
        },
        .cand_src, .plain => {
            const is_cand = plan == .cand_src;
            const bs = if (is_cand) plan.cand_src.bs else plan.plain.bs;
            const n_blocks = if (is_cand) plan.cand_src.n_blocks else plan.plain.n_blocks;
            const k_blk = if (is_cand) plan.cand_src.kb else plan.plain.k_blk;
            const bmax = try runBmax(sc, q, ks, w, wscale, p0, r, bs, is_cand);
            var bids = try selectBlocksTopk(sc, bmax, n_blocks, k_blk, bs, max_pick);
            if (is_cand) {
                if (@min(nb, (p0 + 1) / r) < nb) {
                    const sb_h = try sc.a.alloc(i32, L);
                    defer sc.a.free(sb_h);
                    for (sb_h, 0..) |*v, l| {
                        const seen_l = @min(nb, (p0 + l + 1) / r);
                        v.* = @intCast((seen_l + bs - 1) / bs);
                    }
                    const seen_b = try sc.ints(sb_h, &.{ ci(L), 1 });
                    bids = try sc.where(try sc.op2(mlx.mlx_less, bids, seen_b), bids, try sc.i(-1));
                }
                if (cand.ctx != null) _ = mlx.mlx_array_free(cand.*);
                cand.* = sc.out(bids);
            }
            const scores_g = try runGather(sc, q, ks, w, bids, wscale, p0, r, bs);
            return (try pickRows(sc, scores_g, bids, p0, r, k, bs, k_blk)).?;
        },
    }
}

/// Pre-compiles `idx_nax_bmax`, `idx_nax_gather`, and `dec_topk` pipelines at model load on tiny dummy tensors
/// so the first long-context prefill request does not stall on Metal compilation.
pub fn warmOnlineTopk(gpa: std.mem.Allocator, s: S, c: *const cfg_mod.Config, act: mlx.mlx_dtype, pack_fp4: bool) !void {
    if (!nax_gate.verifyQmmNaxAvailable() or !mlx.streamIsGpu(s)) return;
    if (act != .bfloat16 and act != .float16) return;
    const ih: usize = c.index_heads;
    const ihd: usize = c.index_head_dim;
    if (ih == 0 or ih > 32 or ihd % 32 != 0 or ihd > 128) return;
    try probeDecTopk(gpa, s);
    var sc = Scope.plain(gpa, s);
    defer sc.deinit();
    const L: usize = narrow_rows + 1;
    const nb: usize = 128;
    const bs: usize = if (c.cand_source >= 0 and (c.cand_block_size == 8 or c.cand_block_size == 16 or c.cand_block_size == 32)) c.cand_block_size else 8;
    const kb: usize = 4;
    const q = try sc.zeros(&.{ ci(L), ci(ih), ci(ihd) }, .float32);
    const w = try sc.zeros(&.{ ci(L), ci(ih) }, act);
    const bids = try sc.zeros(&.{ ci(L), ci(kb) }, .int32);
    const ks: KeySource = if (pack_fp4)
        .{ .fp4 = .{ .codes = try sc.zeros(&.{ ci(nb), ci(ihd / 2) }, .uint8), .scales = try sc.zeros(&.{ ci(nb), ci(ihd / 32) }, .uint8), .dt = act } }
    else
        .{ .plain = try sc.zeros(&.{ ci(nb), ci(ihd) }, act) };
    const b0 = try runBmax(&sc, q, ks, w, 1.0, 0, 1, bs, false);
    const b1 = try runBmax(&sc, q, ks, w, 1.0, 0, 1, bs, true);
    const sg = try runGather(&sc, q, ks, w, bids, 1.0, 0, 1, bs);
    const p0 = (try pickRows(&sc, b0, null, 7, 1, 4, bs, 4)).?;
    const p1 = (try pickRows(&sc, sg, bids, 127, 1, 4, bs, kb)).?;
    try sc_mod.evalAll(&.{ b0, b1, sg, p0, p1 });
}

test "sushi_ds41 index_nax: onlineTopkEligible and onlinePlan guard M<=16, invalid shapes, kb<=k, and select plans" {
    const testing = std.testing;
    const device = @import("sushi_exl3").device;
    const scope = device.withProfile(.from("applegpu_g17s", device.Os.parse("26.3"), false));
    defer scope.end();

    const tile = cfg_mod.index_stream_tile;

    try testing.expect(onlineTopkEligible(true, false, tile, true, 17, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, false, 17, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, narrow_rows, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, 1, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(false, false, tile, true, 32, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, true, tile, true, 32, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, 32, true, 32, 100, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, 32, 0, 32, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, 32, 100, 64, 128, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, 32, 100, 32, 16, .bfloat16, .bfloat16, .float32));
    try testing.expect(!onlineTopkEligible(true, false, tile, true, 32, 100, 32, 128, .float32, .bfloat16, .float32));

    var c = cfg_mod.Config{};
    c.index_heads = 32;
    c.index_head_dim = 128;
    c.index_topk = 512;
    c.cand_source = 20;
    c.cand_block_size = 8;
    c.cand_blocks = 2048;

    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 8192, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .dense_nax);
    try testing.expect(onlinePlan(&c, true, false, tile, 2, true, 256, 32768, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .plain);
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 32768, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .cand_src);
    try testing.expect(onlinePlan(&c, true, false, tile, 24, true, 256, 32768, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .post_cand);

    // Pinning consumes 1 slot, so kb <= k must decline cleanly.
    c.index_topk = 2;
    c.cand_blocks = 2;
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 64, 32, 128, .bfloat16, .bfloat16, .float32, null) == null);

    // Invalid BS or kb * bs > max_pick declines cleanly.
    c.index_topk = 512;
    c.cand_block_size = 4;
    c.cand_blocks = 2048;
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 32768, 32, 128, .bfloat16, .bfloat16, .float32, null) == null);
    c.cand_block_size = 8;
    c.cand_blocks = 9000;
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 131072, 32, 128, .bfloat16, .bfloat16, .float32, null) == null);

    // Admission budget cap: 512k (262k keys at r=2) fits inside spanBilledIndexerBytes, while 1M cand_src/plain
    // declines to bounded indexStream; post_cand remains well under budget at any context length.
    c.index_topk = 1024;
    c.cand_blocks = 2048;
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 262_144, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .cand_src);
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 1 << 20, 32, 128, .bfloat16, .bfloat16, .float32, null) == null);
    try testing.expect(onlinePlan(&c, true, false, tile, 24, true, 256, 1 << 20, 32, 128, .bfloat16, .bfloat16, .float32, null).? == .post_cand);

    // Probed dec_topk_tg limit (e.g. reduced to 128 -> max_pick=8192) prevents selecting kb*bs=16384.
    const saved_tg = dec_topk_tg;
    dec_topk_tg = 128;
    defer dec_topk_tg = saved_tg;
    try testing.expect(onlinePlan(&c, true, false, tile, 20, true, 256, 131072, 32, 128, .bfloat16, .bfloat16, .float32, null) == null);

    // Geometry checks
    const g = idxNaxGeom(256, 32768, 128);
    try testing.expectEqual(@as(usize, 32), g.bm);
    try testing.expectEqual(@as(usize, 64), g.dk);
    try testing.expectEqual(@as(usize, 64), g.tgn);
    try testing.expect(idxNaxGatherRanges(256, 16384) == 8);
    try testing.expect(idxNaxGatherRanges(17, 16384) == 32);
}

