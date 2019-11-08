// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

#ifndef XCORE_MODEL_H
#define XCORE_MODEL_H

#include <string>

#include "tensorflow/lite/schema/schema_generated.h"


class XCOREModel {
    private:
        std::unique_ptr<tflite::ModelT> modelT;

    public:
        XCOREModel() {}

        bool Import(const std::string& model_filename);
        bool Export(const std::string& model_filename);
};

#endif /* XCORE_MODEL_H */