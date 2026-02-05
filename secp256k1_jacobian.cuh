#ifndef _SECP256K1_JACOBIAN_CUH
#define _SECP256K1_JACOBIAN_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Big256 for 256-bit integers (little-endian)
struct Big256 { uint32_t w[8]; };
struct Big512 { uint32_t w[16]; };

// Helpers
__device__ inline void big256_set_zero(Big256& a) { for (int i = 0; i < 8; i++) a.w[i] = 0; }
__device__ inline void big512_set_zero(Big512& a) { for (int i = 0; i < 16; i++) a.w[i] = 0; }
__device__ inline void copy256(const Big256& src, Big256& dst) { for (int i = 0; i < 8; i++) dst.w[i] = src.w[i]; }

// Compare
__device__ inline int cmp256(const Big256& a, const Big256& b) {
    for (int i = 7; i >= 0; i--) {
        if (a.w[i] < b.w[i]) return -1;
        if (a.w[i] > b.w[i]) return 1;
    }
    return 0;
}

// Add mod 2^256
__device__ inline void add256(const Big256& a, const Big256& b, Big256& r) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a.w[i] + b.w[i] + carry;
        r.w[i] = (uint32_t)t;
        carry = t >> 32;
    }
}

// Sub (assume a >= b)
__device__ inline void sub256(const Big256& a, const Big256& b, Big256& r) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = (uint64_t)a.w[i];
        uint64_t bv = (uint64_t)b.w[i] + borrow;
        if (av >= bv) { r.w[i] = (uint32_t)(av - bv); borrow = 0; }
        else { r.w[i] = (uint32_t)((1ULL << 32) + av - bv); borrow = 1; }
    }
}

// Multiply: Big256 * Big256 -> Big512
__device__ inline void mul256(const Big256& a, const Big256& b, Big512& c) {
    for (int i = 0; i < 16; i++) c.w[i] = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a.w[i] * b.w[j] + c.w[i + j] + carry;
            c.w[i + j] = (uint32_t)t;
            carry = t >> 32;
        }
        c.w[i + 8] = (uint32_t)carry;
    }
}

// secp256k1 prime p = 2^256 - 2^32 - 977
__device__ inline void get_secp256k1_p(Big256& p) {
    const uint32_t p_words[8] = {
        0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    };
    for (int i = 0; i < 8; i++) p.w[i] = p_words[i];
}

// Optimized reduction for p = 2^256 - 2^32 - 977
__device__ inline void reduce_secp256k1(const Big512& prod, Big256& out) {
    auto add_word = [](uint32_t& dst, uint64_t add, uint64_t& carry) {
        uint64_t s = (uint64_t)dst + add + carry;
        dst = (uint32_t)s;
        carry = s >> 32;
    };

    auto fold_once = [&](const Big512& in, Big512& out512) {
        Big256 H; for (int i = 0; i < 8; ++i) H.w[i] = in.w[8 + i];

        for (int i = 0; i < 16; ++i) out512.w[i] = 0;
        for (int i = 0; i < 8; ++i) out512.w[i] = in.w[i];

        uint64_t carry = 0;
        for (int i = 1; i < 16; ++i) {
            uint64_t add = (i - 1 < 8) ? (uint64_t)H.w[i - 1] : 0;
            add_word(out512.w[i], add, carry);
        }

        carry = 0;
        for (int i = 0; i < 8; ++i) {
            uint64_t add = (uint64_t)H.w[i] * 977ull;
            uint64_t s = (uint64_t)out512.w[i] + (uint32_t)add + carry;
            out512.w[i] = (uint32_t)s;
            carry = (s >> 32) + (add >> 32);
        }
        for (int i = 8; i < 16 && carry; ++i) {
            uint64_t s = (uint64_t)out512.w[i] + carry;
            out512.w[i] = (uint32_t)s;
            carry = s >> 32;
        }
    };

    Big512 t1; fold_once(prod, t1);
    Big512 t2; fold_once(t1, t2);

    for (int i = 0; i < 8; ++i) out.w[i] = t2.w[i];

    Big256 P; get_secp256k1_p(P);
    for (int k = 0; k < 2; ++k) {
        if (cmp256(out, P) >= 0) {
            Big256 tmp; sub256(out, P, tmp);
            copy256(tmp, out);
        }
        else break;
    }
}

// Modular multiply & square
__device__ inline void modmul(const Big256& a, const Big256& b, Big256& r) {
    Big512 prod; mul256(a, b, prod);
    reduce_secp256k1(prod, r);
}
__device__ inline void modsquare(const Big256& a, Big256& r) { modmul(a, a, r); }

// Add mod p
__device__ inline void addmod_p(const Big256& a, const Big256& b, Big256& r) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t t = (uint64_t)a.w[i] + b.w[i] + carry;
        r.w[i] = (uint32_t)t;
        carry = t >> 32;
    }
    Big256 P; get_secp256k1_p(P);
    if (carry || cmp256(r, P) >= 0) {
        Big256 t; sub256(r, P, t);
        copy256(t, r);
    }
}

// Sub mod p
__device__ inline void submod_p(const Big256& a, const Big256& b, Big256& r) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t av = (uint64_t)a.w[i];
        uint64_t bv = (uint64_t)b.w[i] + borrow;
        if (av >= bv) { r.w[i] = (uint32_t)(av - bv); borrow = 0; }
        else { r.w[i] = (uint32_t)((1ULL << 32) + av - bv); borrow = 1; }
    }
    if (borrow) {
        Big256 P; get_secp256k1_p(P);
        Big256 t; add256(r, P, t);
        copy256(t, r);
    }
}

// Modular inverse via Fermat's little theorem
__device__ __constant__ Big256 EXP_P_MINUS_2 = {
    { 0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
      0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }
};

__device__ inline void modexp_with_bigexp(const Big256& base, const Big256& exp, Big256& r) {
    Big256 acc; big256_set_zero(acc); acc.w[0] = 1;

    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = exp.w[wi];
        for (int bi = 31; bi >= 0; --bi) {
            Big256 acc2; modsquare(acc, acc2);
            copy256(acc2, acc);
            if ((w >> bi) & 1u) {
                Big256 tmp; modmul(acc, base, tmp);
                copy256(tmp, acc);
            }
        }
    }
    copy256(acc, r);
}

__device__ inline void modinv(const Big256& a, Big256& r) {
    modexp_with_bigexp(a, EXP_P_MINUS_2, r);
}

// Curve constants (Generator point G)
__device__ __constant__ Big256 Gx_const = {
    { 0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
      0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu }
};
__device__ __constant__ Big256 Gy_const = {
    { 0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
      0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u }
};

__device__ inline void get_curve_G(Big256& Gx, Big256& Gy) {
    copy256(Gx_const, Gx);
    copy256(Gy_const, Gy);
}

// Jacobian point structure
struct PointJ { Big256 X, Y, Z; };

// Precomputed table for windowed scalar multiplication
// Stores [1G, 2G, 3G, ..., 15G] in Jacobian coordinates
#define PRECOMP_TABLE_SIZE 16

struct PrecompTable {
    PointJ points[PRECOMP_TABLE_SIZE];
};

__device__ __constant__ PrecompTable G_precomp_table;

__device__ inline bool is_infty(const PointJ& p) {
    for (int i = 0; i < 8; i++) if (p.Z.w[i] != 0) return false;
    return true;
}

__device__ inline void to_jacobian(const Big256& x, const Big256& y, PointJ& out) {
    copy256(x, out.X);
    copy256(y, out.Y);
    big256_set_zero(out.Z);
    out.Z.w[0] = 1;
}

__device__ inline void from_jacobian(const PointJ& p, Big256& ax, Big256& ay) {
    if (is_infty(p)) { big256_set_zero(ax); big256_set_zero(ay); return; }
    Big256 zinv; modinv(p.Z, zinv);
    Big256 zinv2; modsquare(zinv, zinv2);
    Big256 zinv3; modmul(zinv2, zinv, zinv3);
    modmul(p.X, zinv2, ax);
    modmul(p.Y, zinv3, ay);
}

// Jacobian point doubling (no inversions!)
__device__ inline void jacobian_double(const PointJ& p, PointJ& r_out) {
    if (is_infty(p)) { r_out = p; return; }

    Big256 Y2, XY2, X2, M, M2, Y4;
    modsquare(p.Y, Y2);
    modmul(p.X, Y2, XY2);
    modsquare(p.X, X2);

    Big256 threeX2;
    addmod_p(X2, X2, threeX2);
    addmod_p(threeX2, X2, M);

    Big256 S;
    addmod_p(XY2, XY2, S);
    addmod_p(S, S, S);

    modsquare(M, M2);
    modsquare(Y2, Y4);

    Big256 nx;
    Big256 twoS; addmod_p(S, S, twoS);
    submod_p(M2, twoS, nx);

    Big256 ny;
    Big256 S_minus_nx; submod_p(S, nx, S_minus_nx);
    Big256 nytmp; modmul(M, S_minus_nx, nytmp);
    Big256 eightY4;
    addmod_p(Y4, Y4, eightY4);
    addmod_p(eightY4, eightY4, eightY4);
    addmod_p(eightY4, eightY4, eightY4);
    submod_p(nytmp, eightY4, ny);

    Big256 nz;
    Big256 YZ; modmul(p.Y, p.Z, YZ);
    addmod_p(YZ, YZ, nz);

    copy256(nx, r_out.X); copy256(ny, r_out.Y); copy256(nz, r_out.Z);
}

// Jacobian point addition (no inversions!)
__device__ inline void jacobian_add(const PointJ& p, const PointJ& q, PointJ& r_out) {
    if (is_infty(p)) { r_out = q; return; }
    if (is_infty(q)) { r_out = p; return; }

    Big256 Z2sq; modsquare(q.Z, Z2sq);
    Big256 U1;   modmul(p.X, Z2sq, U1);

    Big256 Z1sq; modsquare(p.Z, Z1sq);
    Big256 U2;   modmul(q.X, Z1sq, U2);

    Big256 Z2cu; modmul(Z2sq, q.Z, Z2cu);
    Big256 S1;   modmul(p.Y, Z2cu, S1);

    Big256 Z1cu; modmul(Z1sq, p.Z, Z1cu);
    Big256 S2;   modmul(q.Y, Z1cu, S2);

    if (cmp256(U1, U2) == 0) {
        if (cmp256(S1, S2) != 0) { big256_set_zero(r_out.X); big256_set_zero(r_out.Y); big256_set_zero(r_out.Z); return; }
        else { jacobian_double(p, r_out); return; }
    }

    Big256 H; submod_p(U2, U1, H);
    Big256 R; submod_p(S2, S1, R);

    Big256 H2; modsquare(H, H2);
    Big256 H3; modmul(H2, H, H3);
    Big256 U1H2; modmul(U1, H2, U1H2);

    Big256 R2; modsquare(R, R2);
    Big256 tmp1; submod_p(R2, H3, tmp1);

    Big256 twoU1H2; addmod_p(U1H2, U1H2, twoU1H2);
    Big256 nx; submod_p(tmp1, twoU1H2, nx);

    Big256 U1H2_minus_nx; submod_p(U1H2, nx, U1H2_minus_nx);
    Big256 Rmul;  modmul(R, U1H2_minus_nx, Rmul);
    Big256 S1H3;  modmul(S1, H3, S1H3);
    Big256 ny;    submod_p(Rmul, S1H3, ny);

    Big256 nz; modmul(H, p.Z, nz); modmul(nz, q.Z, nz);

    copy256(nx, r_out.X); copy256(ny, r_out.Y); copy256(nz, r_out.Z);
}

// Scalar multiplication using Jacobian coordinates
__device__ inline void scalar_mul_jacobian(const PointJ& base, const Big256& scalar, PointJ& res) {
    big256_set_zero(res.X); big256_set_zero(res.Y); big256_set_zero(res.Z);
    PointJ R = res;
    PointJ addp = base;
    
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = scalar.w[wi];
        for (int b = 31; b >= 0; --b) {
            jacobian_double(R, R);
            if ((w >> b) & 1u) jacobian_add(R, addp, R);
        }
    }
    res = R;
}

// Windowed scalar multiplication using 4-bit windows (uses precomputed table)
__device__ inline void scalar_mul_windowed(const Big256& scalar, PointJ& res) {
    big256_set_zero(res.X); big256_set_zero(res.Y); big256_set_zero(res.Z);
    PointJ R = res;
    
    // Process scalar in 4-bit windows from MSB to LSB
    // Big256 is little-endian: w[7] is MSW, w[0] is LSW
    // 256 bits / 4 = 64 windows
    for (int w_idx = 63; w_idx >= 0; --w_idx) {
        // Double 4 times
        for (int i = 0; i < 4; ++i) {
            jacobian_double(R, R);
        }
        
        // Extract 4-bit window
        // w_idx 63 = bits 252-255 (top 4 bits of scalar.w[7])
        // w_idx 0  = bits 0-3 (bottom 4 bits of scalar.w[0])
        int bit_pos = w_idx * 4;  // bit position from LSB
        int word_idx = bit_pos / 32;  // which word (0-7)
        int bit_in_word = bit_pos % 32;  // bit position within word
        
        uint32_t window;
        if (bit_in_word <= 28) {
            // Window fits in one word, extract 4 bits
            window = (scalar.w[word_idx] >> bit_in_word) & 0xF;
        } else {
            // Window spans two words
            uint32_t low_bits = scalar.w[word_idx] >> bit_in_word;
            uint32_t high_bits = (word_idx < 7) ? (scalar.w[word_idx + 1] << (32 - bit_in_word)) : 0;
            window = (low_bits | high_bits) & 0xF;
        }
        
        // Add corresponding precomputed point (skip if window = 0)
        if (window > 0) {
            jacobian_add(R, G_precomp_table.points[window - 1], R);
        }
    }
    
    res = R;
}

// Convert from our old format (unsigned int[8] big-endian) to Big256 (uint32_t[8] little-endian)
__device__ inline void convert_to_big256(const unsigned int privkey[8], Big256& out) {
    // privkey is big-endian: privkey[0] is MSW
    // Big256 is little-endian: w[0] is LSW
    for (int i = 0; i < 8; i++) {
        out.w[i] = privkey[7 - i];
    }
}

// Convert Big256 back to our old format
__device__ inline void convert_from_big256(const Big256& in, unsigned int out[8]) {
    for (int i = 0; i < 8; i++) {
        out[i] = in.w[7 - i];
    }
}

#endif
