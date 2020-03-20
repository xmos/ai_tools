// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "lib_ops/api/conv2d.h"

extern "C" {
    #include "lib_nn/api/nn_types.h"
}

namespace xcore {
namespace conv { 

extern "C" {
    ATTRIBUTE_KERNEL_FUNCTION void conv2d_dido_thread_worker(void *context)
    {
        Conv2DDIDOThreadData *data = (Conv2DDIDOThreadData *)context;
        conv2d_deepin_deepout(data->Y, data->params, data->X, data->K, data->SS);
    }
}

XCoreStatus Conv2D_DIDO::Init(int32_t X_h, int32_t X_w, int32_t zero_point, 
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

    // reserve threads and stack memory 
    KernelDispatcher& dispatcher = GetKernelDispatcher();
    size_t stack_words;
    GET_STACKWORDS(stack_words, conv2d_dido_thread_worker);
    dispatcher.Reserve(params_.size(), stack_words);

    return kXCoreOk;
}

XCoreStatus Conv2D_DIDO::Eval(int8_t* Y, const int8_t* X, const int8_t* K, const int16_t* SS) {
    KernelDispatcher& dispatcher = GetKernelDispatcher();

    size_t stack_words;
    GET_STACKWORDS(stack_words, conv2d_dido_thread_worker);

    for (auto it=params_.cbegin(); it<params_.cend(); ++it) {
        Conv2DDIDOThreadData *data = new Conv2DDIDOThreadData();

        data->Y = Y;
        data->params = &(*it);
        data->X = X;
        data->K = K;
        data->SS = SS;
        dispatcher.Add(conv2d_dido_thread_worker, (void*) data, stack_words);
    }

    dispatcher.Start();
    dispatcher.Wait();

    return kXCoreOk;
}

} // namespace conv
} // namespace xcore
