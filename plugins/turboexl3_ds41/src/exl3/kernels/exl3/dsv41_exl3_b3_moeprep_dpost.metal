#define BANKED_rd(c) (bk == 0u ? rd0[slot * rd0_strides[0] + (c) * rd0_strides[1]] : (bk == 1u ? rd1[slot * rd1_strides[0] + (c) * rd1_strides[1]] : rd2[slot * rd2_strides[0] + (c) * rd2_strides[1]]))

    const float SCALE = as_type<float>(1035273459u);
    uint l = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + 4u * l;

    uint slot_ = ids[row * ids_strides[0]]; uint bk = slot_ >> 24u; uint slot = slot_ & 0xFFFFFFu;
    float w[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        uint c = c0 + q;
        w[q] = zd[row * zd_strides[0] + c * zd_strides[1]];
    }
    {
        float a0 = w[0] + w[1]; float a1 = w[0] - w[1];
        float a2 = w[2] + w[3]; float a3 = w[2] - w[3];
        w[0] = a0 + a2; w[2] = a0 - a2;
        w[1] = a1 + a3; w[3] = a1 - a3;
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float o = simd_shuffle_xor(w[q], (ushort)h);
            w[q] = upper ? (o - w[q]) : (w[q] + o);
        }
    }
    float res[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        uint c = c0 + q;
        res[q] = (w[q] * SCALE) * float(BANKED_rd(c));
    }
    *(device float4*)(out + row * 5120u + c0) = float4(res[0], res[1], res[2], res[3]);
