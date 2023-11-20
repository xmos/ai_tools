#ifndef __dsp_h__
#define __dsp_h__

#define UPSAMPLE_RATE            3
#define WINDOW_SIZE              512
#define WINDOW_ADVANCE           (WINDOW_SIZE/4)
#define WINDOW_OVERLAP           (WINDOW_SIZE - WINDOW_ADVANCE)
#define UPSAMPLED_WINDOW_ADVANCE (WINDOW_ADVANCE * UPSAMPLE_RATE)

void dsp_preprocess(int data_to_be_processed[UPSAMPLED_WINDOW_ADVANCE]);
void dsp_postprocess(int data_processed[UPSAMPLED_WINDOW_ADVANCE]);

#endif
