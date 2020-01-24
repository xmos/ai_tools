// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
#include <iostream>
#include <fstream>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

extern "C"
{
    size_t read_flatbuffer(const char *schema, const char *fb, char *buf)
    {
        std::string schema_contents;
        std::string fb_contents;
        flatbuffers::IDLOptions opts;
        opts.strict_json = true;
        opts.output_default_scalars_in_json = true;

        flatbuffers::Parser parser(opts);
        std::string json;
        bool ok;

        ok = flatbuffers::LoadFile(schema, false, &schema_contents);
        if (!ok) 
        {
            std::cerr << "Error loading " << schema << std::endl;
            return 0;
        }

        ok = flatbuffers::LoadFile(fb, false, &fb_contents);
        if (!ok) 
        {
            std::cerr << "Error loading " << fb << std::endl;
            return 0;
        }

        ok = parser.Parse(schema_contents.c_str());
        if (!ok) 
        {
            std::cerr << "Error parsing " << schema << std::endl;
            return 0;
        }

        if (!GenerateText(parser, fb_contents.c_str(), &json))
        {
            std::cerr << "Error serializing parsed data to JSON" << std::endl;
            return 0;
        }

        strncpy(buf, json.c_str(), json.length());

        return json.length();
    }

    size_t write_flatbuffer(const char *schema, const char *json, const char *fb)
    {
        std::string schema_contents;
        flatbuffers::IDLOptions opts;
        flatbuffers::Parser parser(opts);
        bool ok;

        ok = flatbuffers::LoadFile(schema, false, &schema_contents);
        if (!ok) 
        {
            std::cerr << "Error loading " << schema << std::endl;
            return 0;
        }

        ok = parser.Parse(schema_contents.c_str());
        if (!ok) 
        {
            std::cerr << "Error parsing " << schema << std::endl;
            return 0;
        }

        ok = parser.Parse(json);
        if (!ok) 
        {
            std::cerr << "Error parsing JSON" << std::endl;
            return 0;
        }

        uint8_t *buf = parser.builder_.GetBufferPointer();
        int size = parser.builder_.GetSize();
        std::ofstream ofile(fb, std::ios::binary);
        ofile.write((char *)buf, size);
        ofile.close();

        return size;
    }
}
