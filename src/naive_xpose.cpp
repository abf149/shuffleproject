#include "xpose.h"

/*
    * From:
    * https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions
    *
    * Naive scalar in-place transpose with no meaningful optimization
*/
void naive_square_transpose(int m, int k, DATA_TYPE* a, int lda) {

    #define swapd(x,y) {temp=x; x=y; y=temp;}

    DATA_TYPE temp;
    int r,c,idx,idx_transpose;
    for (r=1; r<lda; r++) {
        for (c=0; c<r; c++) {
            idx = r*lda+c;
            idx_transpose = c*lda+r;
            swapd(a[idx],a[idx_transpose]);
        }
    }
}