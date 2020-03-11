// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_CONV2D_OPERATOR_HPP_
#define XCORE_CONV2D_OPERATOR_HPP_

#include <vector>
#include "operators.hpp"

namespace xcore {
namespace conv { 

struct Conv2DOptions {
    padding_mode_t padding_mode;
    int32_t C_in;
    int32_t C_out;
    int32_t K_h;
    int32_t K_w;
    int32_t stride_h;
    int32_t stride_w;
};

struct Conv2DDIDOThreadData {
    int8_t* Y;
    const nn_conv2d_dido_params_t* params;
    const int8_t* X;
    const int8_t* K;
    const int16_t* SS;
};

extern "C" {
    ATTRIBUTE_KERNEL_FUNCTION
    void conv2d_dido_thread_worker(void *context)
    {
        Conv2DDIDOThreadData *data = (Conv2DDIDOThreadData *)context;
        conv2d_deepin_deepout(data->Y, data->params, data->X, data->K, data->SS);
    }
}

class Conv2D_DIDO {
    public:
        Conv2D_DIDO() {}
        ~Conv2D_DIDO() {}

        void Init(int32_t X_h, int32_t X_w, int32_t zero_point, 
                  int32_t rows, int32_t cols, const int8_t* K, const int16_t* bias) {
            nn_conv2d_init_params_t init_params;
            nn_conv2d_region_params_t  region_params;

            init_params.X_height = X_h;
            init_params.X_width = X_w;
            init_params.K_h = options.K_h;
            init_params.K_w = options.K_w;
            init_params.C_in = options.C_in;
            init_params.C_out = options.C_out;
            init_params.pad_mode = options.padding_mode;
            init_params.zero_point = zero_point;

            for(const auto& region: par) {
                nn_conv2d_dido_params_t params;

                region_params.top = region.top;
                region_params.left = region.left;
                region_params.rows = region.rows;
                region_params.cols = region.cols;

                conv2d_deepin_deepout_init(
                    &params,
                    &init_params,
                    &region_params,
                    K,
                    (data16_t*) bias
                );

                params_.emplace_back(std::move(params));
            }
        }

        void Eval(int8_t* Y, const int8_t* X, const int8_t* K, const int16_t* SS) {
            ThreadGroup tg;

            //for(const auto& params: params_) {
            for (auto it=params_.cbegin(); it<params_.cend()-1; ++it) {
                Conv2DDIDOThreadData *data = (Conv2DDIDOThreadData *)malloc(sizeof(Conv2DDIDOThreadData));
                data->Y = Y;
                data->params = &(*it);
                data->X = X;
                data->K = K;
                data->SS = SS;
                int stack_words = 1024;
                void* stack = malloc(stack_words*4);
                tg.Add(conv2d_dido_thread_worker, (void*) data, stack_words, stack);
            }

            tg.Start();

            // execute last job in this thread
            nn_conv2d_dido_params_t const& params = params_.back();
            Conv2DDIDOThreadData *data = (Conv2DDIDOThreadData *)malloc(sizeof(Conv2DDIDOThreadData));
            data->Y = Y;
            data->params = &params;
            data->X = X;
            data->K = K;
            data->SS = SS;
            conv2d_dido_thread_worker((void*) data);

            tg.Wait();
        }

        Conv2DOptions options;
        ParPlan par;
    private:
        std::vector<nn_conv2d_dido_params_t> params_;
};

} // namespace conv
} // namespace xcore

#endif  // XCORE_CONV2D_OPERATOR_HPP_