
#include <stdio.h>


inline void argmax_16(const int16_t *A, int32_t *C, int32_t N)
{
    int16_t maxA = INT16_MIN;
    int32_t maxIndex = 0;
    
    for (int32_t i=0; i<N; i++) {
        printf("%ld     %d\n", i, A[i]);
        if (A[i] > maxA) {
            maxIndex = i;
            maxA = A[i];
        }
    }

    *C = maxIndex;
}
