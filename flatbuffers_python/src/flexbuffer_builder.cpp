// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include <iostream>
#include <vector>

#include "flatbuffers/flexbuffers.h"

extern "C"
{
    // ************************
    // flexbuffers::Builder API
    //   see https://github.com/google/flatbuffers/blob/master/include/flatbuffers/flexbuffers.h
    //   see: https://google.github.io/flatbuffers/flexbuffers.html
    // ************************
       
    flexbuffers::Builder *new_builder()
    {
        return new flexbuffers::Builder();
    }

    size_t builder_start_map(flexbuffers::Builder *fbb)
    {
        return fbb->StartMap();
    }

    size_t builder_end_map(flexbuffers::Builder *fbb, size_t size)
    {
        return fbb->EndMap(size);
    }

    size_t builder_start_vector(flexbuffers::Builder *fbb, const char* key)
    {
        return fbb->StartVector(key);
    }

    size_t builder_end_vector(flexbuffers::Builder *fbb, size_t size, bool typed, bool fixed)
    {
        return fbb->EndVector(size, typed, fixed);
    }

    void builder_clear(flexbuffers::Builder *fbb)
    {
        fbb->Clear();
    }

    void builder_int(flexbuffers::Builder *fbb, const char* key, int64_t val)
    {
        fbb->Int(key, val);
    }

    void builder_vector_int(flexbuffers::Builder *fbb,  int64_t val)
    {
        fbb->Int(val);
    }

    void builder_uint(flexbuffers::Builder *fbb, const char* key, uint64_t val)
    {
        fbb->UInt(key, val);
    }

    void builder_vector_uint(flexbuffers::Builder *fbb,  uint64_t val)
    {
        fbb->UInt(val);
    }

    void builder_string(flexbuffers::Builder *fbb, const char* key, const char* val)
    {
        fbb->String(key, val);
    }

    void builder_finish(flexbuffers::Builder *fbb)
    {
        fbb->Finish();
    }

    size_t builder_get_buffer(flexbuffers::Builder *fbb, char *buf)
    {
        std::vector<uint8_t> bytes = fbb->GetBuffer();

        for(std::size_t i = 0; i < bytes.size(); ++i) {
            buf[i] = bytes[i];
        }

        return bytes.size();
    }

} //extern "C"
