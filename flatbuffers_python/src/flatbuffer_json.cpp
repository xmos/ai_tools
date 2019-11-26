// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include <iostream>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

extern "C"
{
    // ************************
    // flexbuffers::Builder API
    //   see https://github.com/google/flatbuffers/blob/master/include/flatbuffers/flexbuffers.h
    //   see: https://google.github.io/flatbuffers/flexbuffers.html
    // ************************
       
    size_t read_flatbuffer(const char *schema, const char *fbs, char *buf)
    {
        std::string schemafile;
        std::string fbsfile;
        flatbuffers::IDLOptions opts;
        opts.strict_json = true;
        opts.output_default_scalars_in_json = true;

        flatbuffers::Parser parser(opts);
        std::string json;
        bool ok;

        ok = flatbuffers::LoadFile(schema, false, &schemafile);
        if (!ok) 
        {
            std::cerr << "Error loading " << schema << std::endl;
            return 0;
        }

        ok = flatbuffers::LoadFile(fbs, false, &fbsfile);
        if (!ok) 
        {
            std::cerr << "Error loading " << fbs << std::endl;
            return 0;
        }

        ok = parser.Parse(schemafile.c_str());
        if (!ok) 
        {
            std::cerr << "Error parsing " << schema << std::endl;
            return 0;
        }

        if (!GenerateText(parser, fbsfile.c_str(), &json))
        {
            std::cerr << "Error serializing parsed data to JSON" << std::endl;
            return 0;
        }

        strncpy(buf, json.c_str(), json.length());

        return json.length();
    }

    size_t write_flatbuffer(const char *schema, const char *buf, const char *fbs)
    {

    }

}
