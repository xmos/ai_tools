// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

#include <fstream>
#include <iostream>

#include "xcore_model.h"

bool XCOREModel::Import(const std::string& model_filename)
{
    std::ifstream file(model_filename.c_str(), std::ios::binary | std::ios::ate);

    if (file) {
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);


        if (file.read(buffer.data(), size))
        {
            const tflite::Model* readonly_model = ::tflite::GetModel(buffer.data());
            // verify readonly_model
            ::flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*> (buffer.data()), buffer.size());

            if (readonly_model->Verify(verifier))
            {
                modelT.reset(readonly_model->UnPack());
                return true;
            } else {
                std::cerr << "Unable to verify file " << model_filename << std::endl;
            }
        } else {
            std::cerr << "Unable to read file " << model_filename << std::endl;
        }
    } else {
        std::cerr << "Unable to open file " << model_filename << std::endl;
    }

    return false;
}

bool XCOREModel::Export(const std::string& model_filename)
{
    flatbuffers::FlatBufferBuilder fb_builder(/*initial_size=*/10240);
    flatbuffers::Offset<tflite::Model> output_model = tflite::Model::Pack(fb_builder, modelT.get());
    FinishModelBuffer(fb_builder, output_model);

    std::fstream file(model_filename.c_str(), std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char*>(fb_builder.GetBufferPointer()), fb_builder.GetSize());
    file.close();

    return true;
}

bool XCOREModel::Transform(const std::string& description) {
    modelT->description = description;

    return true;
}

void XCOREModel::TransformTensors() {
    //TODO: implement me
}
