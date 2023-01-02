#include <iostream>
#include "parameters.h"
#include "xpose.h"
#include "test_xpose.h"

/**
 * A collection of test routines for matrix transpose
 */

/** Transpose random square 16x16 matrix. Print both matrices. */
void test_print_naive_square_transpose() {
    std::cout<<"test 16x16 naive transpose"<<'\n';

    /* Benchmarking code from University of Texas HowToOptimizeGemm */

    int 
        p, 
        m, n, k,
        lda, ldb, ldc, 
        rep;

    double
        dtime, dtime_best,        
        gflops, 
        diff;

    DATA_TYPE 
        *a, *b, *c, *cref, *cold;  

    /* Target 16x16 matrix */
    p=16;

    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I (ABF) eliminated the extra column mentioned below.*/
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( DATA_TYPE * ) malloc( lda * (k) * sizeof( DATA_TYPE ) );  

    /* Generate random matrix A */
    dummy_int_matrix( m, k, a, lda );

    print_matrix( m, k, a, lda );

    naive_square_transpose(m, k, a, lda);    

    print_matrix( m, k, a, lda );

    /* Free memory */
    free(a);    
}

/** Transpose random square 16x16 matrix using in-place in-register SIMD routine. 
 * Print both matrices. 
 */
void test_print_simd_square_transpose_intrinsic_16x16() {
    std::cout<<"test 16x16 simd transpose using intrinsics"<<'\n';

    /* Benchmarking code from University of Texas HowToOptimizeGemm */

    int 
        p, 
        m, n, k,
        lda, ldb, ldc, 
        rep;

    double
        dtime, dtime_best,        
        gflops, 
        diff;

    DATA_TYPE 
        *a, *b, *c, *cref, *cold;  

    /* Target 16x16 matrix */
    p=16;

    m = ( M == -1 ? p : M );
    n = ( N == -1 ? p : N );
    k = ( K == -1 ? p : K );

    gflops = 2.0 * m * n * k * 1.0e-09;

    lda = ( LDA == -1 ? m : LDA );
    ldb = ( LDB == -1 ? k : LDB );
    ldc = ( LDC == -1 ? m : LDC );

    /* Allocate space for the matrices */
    /* Note: I (ABF) eliminated the extra column mentioned below.*/
    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault */
    a = ( DATA_TYPE * ) malloc( lda * (k) * sizeof( DATA_TYPE ) );  

    /* Generate random matrix A */
    dummy_int_matrix( m, k, a, lda );

    print_matrix( m, k, a, lda );

    //naive_square_transpose(m, k, a, lda);   
    simd_square_transpose_intrinsic_16x16(m, k, a, lda); 

    print_matrix( m, k, a, lda );

    /* Free memory */
    free(a);    
}