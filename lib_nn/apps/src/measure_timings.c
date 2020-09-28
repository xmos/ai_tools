#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_conv2d_bin.h"
#include "xcore/hwtimer.h"

//Xscope is disabled for now as fileoi isn't supported yet but when it is this
//will execute much faster on hw.

// #include <xscope.h>
// void xscope_user_init() {
//    xscope_register(0, 0, "", 0, "");
//    xscope_config_io(XSCOPE_IO_BASIC);
// }

void bnn_conv2d_bin_out_asm(nn_bnn_conv2d_bin_out_asm_plan_t * plan);
void bnn_conv2d_int8_out_asm(nn_bnn_conv2d_int8_out_asm_plan_t * plan);

static const char * bin_out_filename = "bin.csv";
static const char * int8_out_filename = "int8.csv";
static const unsigned test_count = (1<<15);

void profile_bin_out(){

    int64_t memory[62*1024];

    nn_bnn_conv2d_bin_out_asm_plan_t p;
    nn_bnn_conv2d_bin_out_asm_plan_t p_copy;

    //These can be ignored as we they don't effect timing.
    p.outer_x_h_step = 0;
    p.inner_x_v_step = 0;
    p.k_v_step = 0;
    p.inner_x_h_step = 0;
    p.k_h_step = 0;
    p.outer_x_v_step = 0;
    p.y_v_step = 0;

    //Pointers - These need to be valid but can overlap to save memory
    p.Y = (bnn_b32_t *)&memory;
    p.X = (bnn_b256_t *)&memory;
    p.K = (bnn_b256_t *)&memory;
    p.threshold_p = (int32_t *)&memory;

    //These are what we are testing
    p.output_channel_loop_counter = 0;
    p.k_height_loop_counter = 0;
    p.k_width_loop_counter = 0;
    p.x_height_loop_counter = 0;
    p.x_width_loop_counter = 0;
    p.input_channel_loop_counter = 0;

    const unsigned max_output_channel_loop_counter = 8; 
    const unsigned max_k_height_loop_counter = 8; 
    const unsigned max_k_width_loop_counter = 8; 
    const unsigned max_x_height_loop_counter = 16; 
    const unsigned max_x_width_loop_counter = 16; 
    const unsigned max_input_channel_loop_counter = 8; 
    
    hwtimer_t t = hwtimer_alloc();
    
    FILE * fp = fopen(bin_out_filename, "w");
    unsigned test = test_count;
    while(test > 0){

        //These are what we are testing
        p.output_channel_loop_counter = rand()%max_output_channel_loop_counter;
        p.k_height_loop_counter = rand()%max_k_height_loop_counter;
        p.k_width_loop_counter = rand()%max_k_width_loop_counter;
        p.input_channel_loop_counter = rand()%max_input_channel_loop_counter;

        unsigned b = (p.output_channel_loop_counter+1) * (p.k_height_loop_counter+1) *
            (p.k_width_loop_counter+1) * (p.input_channel_loop_counter+1) * 32 * (256/8);

        if(b > sizeof(memory)){
            continue;
        } else {
            test--;
        }

        p.x_height_loop_counter = (rand()%max_x_height_loop_counter) + 1;
        p.x_width_loop_counter = rand()%max_x_width_loop_counter;

        //copy it as the function call clobbers the values
        memcpy(&p_copy, &p, sizeof(p));
        
        //now time it
        uint32_t before = hwtimer_get_time(t);  
        bnn_conv2d_bin_out_asm(&p_copy);
        uint32_t after = hwtimer_get_time(t);

        unsigned elapsed = after - before;

        fprintf(fp, "%u,%u,%u,%u,%u,%u,%u\n", 
            p.x_height_loop_counter, 
            p.x_width_loop_counter+1,
            p.output_channel_loop_counter+1, 
            p.k_height_loop_counter+1, 
            p.k_width_loop_counter+1, 
            p.input_channel_loop_counter+1, 
            elapsed);

    }
    fclose(fp);
  hwtimer_free(t);
}


void profile_int8_out(){

    int64_t memory[62*1024];

    nn_bnn_conv2d_int8_out_asm_plan_t p;
    nn_bnn_conv2d_int8_out_asm_plan_t p_copy;

    //These can be ignored as we they don't effect timing.
    p.outer_x_h_step = 0;
    p.inner_x_v_step = 0;
    p.k_v_step = 0;
    p.inner_x_h_step = 0;
    p.k_h_step = 0;
    p.outer_x_v_step = 0;
    p.y_v_step = 0;

    //Pointers - These need to be valid but can overlap to save memory
    p.Y = (int8_t *)&memory;
    p.X = (bnn_b256_t *)&memory;
    p.K = (bnn_b256_t *)&memory;
    p.post_activation_bias = (int16_t *)&memory;
    p.post_activation_mul = (int16_t *)&memory;

    //These are what we are testing
    p.output_channel_loop_counter = 0;
    p.k_height_loop_counter = 0;
    p.k_width_loop_counter = 0;
    p.x_height_loop_counter = 0;
    p.x_width_loop_counter = 0;
    p.input_channel_loop_counter = 0;

    const unsigned max_output_channel_loop_counter = 8; 
    const unsigned max_k_height_loop_counter = 8; 
    const unsigned max_k_width_loop_counter = 8; 
    const unsigned max_x_height_loop_counter = 16; 
    const unsigned max_x_width_loop_counter = 16; 
    const unsigned max_input_channel_loop_counter = 8; 
    
    hwtimer_t t = hwtimer_alloc();
    FILE * fp = fopen(int8_out_filename, "w");
    unsigned test = test_count;
    while(test > 0){

        //These are what we are testing
        //TODO these can be chosen such that the product of the 4 of them doesnt require more than sizeof(memory) bytes
        p.output_channel_loop_counter = rand()%max_output_channel_loop_counter;
        p.k_height_loop_counter = rand()%max_k_height_loop_counter;
        p.k_width_loop_counter = rand()%max_k_width_loop_counter;
        p.input_channel_loop_counter = rand()%max_input_channel_loop_counter;

        unsigned b = (p.output_channel_loop_counter+1) * (p.k_height_loop_counter+1) *
            (p.k_width_loop_counter+1) * (p.input_channel_loop_counter+1) * 16 * (256/8); 

        if(b > sizeof(memory)){
            continue;
        } else {
            test--;
        }

        p.x_height_loop_counter = (rand()%max_x_height_loop_counter) + 1;
        p.x_width_loop_counter = rand()%max_x_width_loop_counter;

        //copy it as the function call clobbers the values
        memcpy(&p_copy, &p, sizeof(p));
        
        //now time it
        uint32_t before = hwtimer_get_time(t);  
        bnn_conv2d_int8_out_asm(&p_copy);
        uint32_t after = hwtimer_get_time(t);

        unsigned elapsed = after - before;

        fprintf(fp, "%u,%u,%u,%u,%u,%u,%u\n", 
            p.x_height_loop_counter, 
            p.x_width_loop_counter+1,
            p.output_channel_loop_counter+1, 
            p.k_height_loop_counter+1, 
            p.k_width_loop_counter+1, 
            p.input_channel_loop_counter+1, 
            elapsed);


    }
    fclose(fp);

  hwtimer_free(t);
}

int main(){

    srand(42);
    profile_bin_out();
    profile_int8_out();
    return 0;
}