// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

#include <string>
#include <iostream>

#include "anyoption.h"
#include "xcore_model.h"

static int convert2xcore(const std::string& input_filename, const std::string& output_filename)
{
    XCOREModel model;

    if (model.Import(input_filename))
    {
        model.Transform("generated with tflite2xcore");
        model.Export(output_filename);
        return 0;
    }

    return 1;
}

int main(int argc, char **argv)
{
    std::unique_ptr<AnyOption> opt(new AnyOption()); //see: https://github.com/hackorama/AnyOption/blob/master/demo.cpp

    opt->addUsage("usage: ");
    opt->addUsage("");
    opt->addUsage("  tflite2xcore <input.tflite>  <output.tflite>");
    opt->addUsage("");
    opt->addUsage("  -h  --help           Prints this help");
    opt->addUsage("");

    opt->setFlag("help", 'h'); 
    // opt->setFlag("xcore", 'x'); 

    opt->processCommandArgs(argc, argv);

    if (!opt->hasOptions()) { /* print usage if no options */
        opt->printUsage();
        return 0;
    }

    if (opt->getFlag("help") || opt->getFlag('h'))
    {
        opt->printUsage();
        return 0;
    }

    if (opt->getArgc() < 2) 
    {
        std::cerr << "Two arguments expected." << std::endl;
        opt->printUsage();
        exit(1);
    }

    return convert2xcore(opt->getArgv(0), opt->getArgv(1));

}