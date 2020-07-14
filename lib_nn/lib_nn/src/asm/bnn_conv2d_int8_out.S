
#if defined(__XS3A__)


/*
void bnn_conv2d_int8_asm(int32_t* Y_p, const bnn_t* X_p, const bnn_t* K_p,
                float * effective_post_activation_multiplier,  //[out_channel];
                float * effective_post_activation_bias,        //[out_channel];
                const nn_bnn_conv2d_plan_t* plan)
*/

#define FUNCTION_NAME bnn_conv2d_binary_output_asm

#define NSTACKWORDS  16
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define Y_p       r0
#define X_p       r1
#define K_p       r2


#define pa_mul    r3
#define pa_bias   r3
#define plan_p    r3

#define data      r4
#define i         r5



//There must be multiples of 16 output channels
FUNCTION_NAME:
/*
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    std r10, r11, sp[4]


    stride, 

    h, w, ic, kh, kw, oc, oc_inner

    h_loop:
        w_loop:

            ic_loop:
                kh_loop:
                    
                    kw_loop:
                        //vD =  X[h + kh][w + kw][ic]
                        ldw oc_outer_loop_counter, sp[OC_OUTER_LOOP_COUNTER]
                        oc_outer_loop:
                            
                            ldw oc_inner_loop_counter, sp[OC_INNER_LOOP_COUNTER]
                            oc_inner_loop:
                                vlmaccr1 K_p[oc][kh][kw][ic] ; add K, K, one_oc
                            {bt oc_inner_loop_counter, oc_inner_loop; sub oc_inner_loop_counter, oc_inner_loop_counter, 1}
                                                        
                            
                            //store to Y[h][w][oc:oc+16]
                        
                        {bt oc_outer_loop_counter, oc_outer_loop; sub oc_outer_loop_counter, oc_outer_loop_counter, 1}

                    bt kw_loop
                bt kh_loop
            bt ic_loop

        bt w_loop

    bt h_loop




.Lfunc_end:
    ldd r4, r5, sp[1]
    ldd r6, r7, sp[2]
    ldd r8, r9, sp[3]
    ldd r10, r11, sp[4]
*/
    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif



