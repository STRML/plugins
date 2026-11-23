// One thread a row's 4 columns: the joinless combine (dsv41_jl_combine_bf16's words: each routed row read where its wave
// wrote it, times its weight, folded in MLX's column reduce order, the shared row added, all f32, uncontracted), rounded
// once to T as the model's astype does, then glm5's hc expand of the row's HC streams (out[k] = post[k] * y + sum_j
// comb[j][k] * res[j], f32 fmas in its order, rounded once to T). Inputs: s0..s23 the routed outputs (S), loc int32
// [n, K, 2] (each assignment's source and row), w f32 [n, K], sh [n, D] (ST), res [n, HC, D] (T), pc f32 [n, HC + HC*HC]
// (each row's post, then comb [HC][HC]); out [n, HC, D] (T).
// Grid: (D/4, N, 1) with threadgroup (256, 1, 1): every threadgroup owns one row so loc/w/pc are TG-cached once, and
// vector 4-wide loads/stores handle all S/T/ST dtypes.
constexpr uint C = 4;
constexpr uint DC = uint(D) / C;
const uint tid = thread_position_in_threadgroup.x;
const uint tg_x = threads_per_threadgroup.x;
const uint dc_idx = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
if (row >= uint(NROWS)) return;
threadgroup int tg_loc[uint(K) * 2u];
threadgroup float tg_w[uint(K)];
threadgroup float tg_pc[uint(HC + HC * HC)];
const ulong lbase = ulong(row) * ulong(K);
const ulong pbase = ulong(row) * ulong(HC + HC * HC);
for (uint i = tid; i < uint(K) * 2u; i += tg_x) tg_loc[i] = loc[2u * lbase + i];
for (uint i = tid; i < uint(K); i += tg_x) tg_w[i] = static_cast<float>(w[lbase + i]);
for (uint i = tid; i < uint(HC + HC * HC); i += tg_x) tg_pc[i] = pc[pbase + i];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (dc_idx >= DC) return;
const uint col = dc_idx * C;
float4 t;
{
    const float wv = tg_w[0];
    const device S* p0 = tf_pick<S>(tg_loc[0], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
    const ulong o0 = ulong(tg_loc[1]) * ulong(D) + ulong(col);
    t = tf_load4(p0 + o0) * wv + 0.0f;
}
for (uint k = 1; k < uint(K); ++k) {
    const float wv = tg_w[k];
    const device S* pk = tf_pick<S>(tg_loc[2u * k], s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23);
    const ulong ok = ulong(tg_loc[2u * k + 1u]) * ulong(D) + ulong(col);
    t = (tf_load4(pk + ok) * wv + 0.0f) + t;
}
const ulong obase = ulong(row) * ulong(D) + ulong(col);
const float4 y = float4(vec<T, 4>(t + tf_load4(sh + obase)));
float4 sv[HC];
for (int j = 0; j < HC; ++j) {
    sv[j] = tf_load4(res + (ulong(row) * ulong(HC) + ulong(j)) * ulong(D) + ulong(col));
}
for (int k2 = 0; k2 < HC; ++k2) {
    float4 acc = float4(0.0f);
    for (int j = 0; j < HC; ++j) acc = metal::fma(tg_pc[HC + j * HC + k2], sv[j], acc);
    tf_store4(out + (ulong(row) * ulong(HC) + ulong(k2)) * ulong(D) + ulong(col), metal::fma(tg_pc[k2], y, acc));
}
