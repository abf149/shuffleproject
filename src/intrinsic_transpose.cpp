#include <immintrin.h>
#include <cassert>
#include "xpose.h"

/* Wrapper for in-place in-register SIMD 16x16 matrix transpose */
void simd_square_transpose_intrinsic_16x16(int m, int k, DATA_TYPE* a, int lda) {
    // Must be 16x16
    assert(m==16 && k==16);
    simd_tran_16x16_permutexvar_inplace(a);
}

/*
    * In-place in-register SIMD 16x16 matrix transpose from:
    * https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions
    *
    * Specifically the tran_new2 implementation using AVX512 vector instructions
    *
*/
void simd_tran_16x16_permutexvar_inplace(int* mat) {

    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

    int mask;
    int64_t idx1[8] __attribute__((aligned(64))) = {2, 3, 0, 1, 6, 7, 4, 5}; 
    int64_t idx2[8] __attribute__((aligned(64))) = {1, 0, 3, 2, 5, 4, 7, 6}; 
    int32_t idx3[16] __attribute__((aligned(64))) = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi32(idx3);

    t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+0])), _mm256_load_si256((__m256i*)&mat[ 8*16+0]), 1);
    t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+0])), _mm256_load_si256((__m256i*)&mat[ 9*16+0]), 1);
    t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+0])), _mm256_load_si256((__m256i*)&mat[10*16+0]), 1);
    t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+0])), _mm256_load_si256((__m256i*)&mat[11*16+0]), 1);
    t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+0])), _mm256_load_si256((__m256i*)&mat[12*16+0]), 1);
    t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+0])), _mm256_load_si256((__m256i*)&mat[13*16+0]), 1);
    t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+0])), _mm256_load_si256((__m256i*)&mat[14*16+0]), 1);
    t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+0])), _mm256_load_si256((__m256i*)&mat[15*16+0]), 1);

    t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+8])), _mm256_load_si256((__m256i*)&mat[ 8*16+8]), 1);
    t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+8])), _mm256_load_si256((__m256i*)&mat[ 9*16+8]), 1);
    ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+8])), _mm256_load_si256((__m256i*)&mat[10*16+8]), 1);
    tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+8])), _mm256_load_si256((__m256i*)&mat[11*16+8]), 1);
    tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+8])), _mm256_load_si256((__m256i*)&mat[12*16+8]), 1);
    td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+8])), _mm256_load_si256((__m256i*)&mat[13*16+8]), 1);
    te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+8])), _mm256_load_si256((__m256i*)&mat[14*16+8]), 1);
    tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+8])), _mm256_load_si256((__m256i*)&mat[15*16+8]), 1);

    mask= 0xcc;
    r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
    r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
    r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
    r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
    r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
    r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
    ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
    rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);

    mask= 0x33;
    r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
    r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
    r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
    r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
    rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
    rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
    re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
    rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);

    mask = 0xaa;
    t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
    t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
    t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
    t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
    t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
    t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
    tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
    td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);

    mask = 0x55;
    t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
    t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
    t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
    t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
    ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
    tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
    te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
    tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);

    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
    r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
    r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
    r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
    r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
    ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
    rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
    re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    

    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
    r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
    r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
    r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
    r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
    rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
    rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
    rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);

    _mm512_store_epi32(&mat[ 0*16], r0);
    _mm512_store_epi32(&mat[ 1*16], r1);
    _mm512_store_epi32(&mat[ 2*16], r2);
    _mm512_store_epi32(&mat[ 3*16], r3);
    _mm512_store_epi32(&mat[ 4*16], r4);
    _mm512_store_epi32(&mat[ 5*16], r5);
    _mm512_store_epi32(&mat[ 6*16], r6);
    _mm512_store_epi32(&mat[ 7*16], r7);
    _mm512_store_epi32(&mat[ 8*16], r8);
    _mm512_store_epi32(&mat[ 9*16], r9);
    _mm512_store_epi32(&mat[10*16], ra);
    _mm512_store_epi32(&mat[11*16], rb);
    _mm512_store_epi32(&mat[12*16], rc);
    _mm512_store_epi32(&mat[13*16], rd);
    _mm512_store_epi32(&mat[14*16], re);
    _mm512_store_epi32(&mat[15*16], rf);
}

/* Work-in-progress */
void simd_tran_16x16_permutexvar_outofplace(int* mat, int* matT) {

    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

    int mask;
    int64_t idx1[8] __attribute__((aligned(64))) = {2, 3, 0, 1, 6, 7, 4, 5}; 
    int64_t idx2[8] __attribute__((aligned(64))) = {1, 0, 3, 2, 5, 4, 7, 6}; 
    int32_t idx3[16] __attribute__((aligned(64))) = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi32(idx3);

    t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+0])), _mm256_load_si256((__m256i*)&mat[ 8*16+0]), 1);
    t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+0])), _mm256_load_si256((__m256i*)&mat[ 9*16+0]), 1);
    t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+0])), _mm256_load_si256((__m256i*)&mat[10*16+0]), 1);
    t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+0])), _mm256_load_si256((__m256i*)&mat[11*16+0]), 1);
    t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+0])), _mm256_load_si256((__m256i*)&mat[12*16+0]), 1);
    t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+0])), _mm256_load_si256((__m256i*)&mat[13*16+0]), 1);
    t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+0])), _mm256_load_si256((__m256i*)&mat[14*16+0]), 1);
    t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+0])), _mm256_load_si256((__m256i*)&mat[15*16+0]), 1);

    t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+8])), _mm256_load_si256((__m256i*)&mat[ 8*16+8]), 1);
    t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+8])), _mm256_load_si256((__m256i*)&mat[ 9*16+8]), 1);
    ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+8])), _mm256_load_si256((__m256i*)&mat[10*16+8]), 1);
    tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+8])), _mm256_load_si256((__m256i*)&mat[11*16+8]), 1);
    tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+8])), _mm256_load_si256((__m256i*)&mat[12*16+8]), 1);
    td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+8])), _mm256_load_si256((__m256i*)&mat[13*16+8]), 1);
    te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+8])), _mm256_load_si256((__m256i*)&mat[14*16+8]), 1);
    tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+8])), _mm256_load_si256((__m256i*)&mat[15*16+8]), 1);

    mask= 0xcc;
    r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
    r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
    r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
    r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
    r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
    r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
    ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
    rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);

    mask= 0x33;
    r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
    r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
    r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
    r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
    rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
    rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
    re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
    rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);

    mask = 0xaa;
    t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
    t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
    t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
    t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
    t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
    t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
    tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
    td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);

    mask = 0x55;
    t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
    t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
    t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
    t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
    ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
    tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
    te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
    tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);

    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
    r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
    r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
    r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
    r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
    ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
    rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
    re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    

    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
    r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
    r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
    r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
    r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
    rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
    rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
    rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);

    _mm512_store_epi32(&matT[ 0*16], r0);
    _mm512_store_epi32(&matT[ 1*16], r1);
    _mm512_store_epi32(&matT[ 2*16], r2);
    _mm512_store_epi32(&matT[ 3*16], r3);
    _mm512_store_epi32(&matT[ 4*16], r4);
    _mm512_store_epi32(&matT[ 5*16], r5);
    _mm512_store_epi32(&matT[ 6*16], r6);
    _mm512_store_epi32(&matT[ 7*16], r7);
    _mm512_store_epi32(&matT[ 8*16], r8);
    _mm512_store_epi32(&matT[ 9*16], r9);
    _mm512_store_epi32(&matT[10*16], ra);
    _mm512_store_epi32(&matT[11*16], rb);
    _mm512_store_epi32(&matT[12*16], rc);
    _mm512_store_epi32(&matT[13*16], rd);
    _mm512_store_epi32(&matT[14*16], re);
    _mm512_store_epi32(&matT[15*16], rf);
}