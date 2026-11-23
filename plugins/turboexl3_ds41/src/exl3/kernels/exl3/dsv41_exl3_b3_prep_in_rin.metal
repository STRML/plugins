#define BANKED_rg(c) (bk == 0u ? rg0[slot * rg0_strides[0] + (c) * rg0_strides[1]] : (bk == 1u ? rg1[slot * rg1_strides[0] + (c) * rg1_strides[1]] : rg2[slot * rg2_strides[0] + (c) * rg2_strides[1]]))
#define BANKED_ru(c) (bk == 0u ? ru0[slot * ru0_strides[0] + (c) * ru0_strides[1]] : (bk == 1u ? ru1[slot * ru1_strides[0] + (c) * ru1_strides[1]] : ru2[slot * ru2_strides[0] + (c) * ru2_strides[1]]))

    const float SCALE = as_type<float>(1035273459u);
    uint l = thread_index_in_simdgroup;
    uint blk = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint c0 = blk * 128u + 4u * l;

    uint slot_ = ids[row * ids_strides[0]]; uint bk = slot_ >> 24u; uint slot = slot_ & 0xFFFFFFu;
    uint t = uint(tok[row * tok_strides[0]]);
    float wg[4];
    float wu[4];
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
        uint c = c0 + q;
        float xv = float(x[t * x_strides[0] + c * x_strides[1]]);
        wg[q] = xv * float(BANKED_rg(c));
        wu[q] = xv * float(BANKED_ru(c));
    }
    {
        float a0 = wg[0] + wg[1]; float a1 = wg[0] - wg[1];
        float a2 = wg[2] + wg[3]; float a3 = wg[2] - wg[3];
        wg[0] = a0 + a2; wg[2] = a0 - a2;
        wg[1] = a1 + a3; wg[3] = a1 - a3;
    }
    {
        float a0 = wu[0] + wu[1]; float a1 = wu[0] - wu[1];
        float a2 = wu[2] + wu[3]; float a3 = wu[2] - wu[3];
        wu[0] = a0 + a2; wu[2] = a0 - a2;
        wu[1] = a1 + a3; wu[3] = a1 - a3;
    }
    _Pragma("clang loop unroll(full)")
    for (uint h = 1u; h < 32u; h <<= 1u) {
        bool upper = (l & h) != 0u;
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
            float og = simd_shuffle_xor(wg[q], (ushort)h);
            float ou = simd_shuffle_xor(wu[q], (ushort)h);
            wg[q] = upper ? (og - wg[q]) : (wg[q] + og);
            wu[q] = upper ? (ou - wu[q]) : (wu[q] + ou);
        }
    }
    *(device float4*)(xg + row * 5120u + c0) = float4(wg[0], wg[1], wg[2], wg[3]) * SCALE;
    *(device float4*)(xu + row * 5120u + c0) = float4(wu[0], wu[1], wu[2], wu[3]) * SCALE;
