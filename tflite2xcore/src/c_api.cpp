// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved

#import "xcore_model.h"

extern "C"
{
    XCOREModel *model_import(const char *filename)
    {
        std::unique_ptr<XCOREModel> model(new XCOREModel()); 
        if (model->Import(filename))
            return model.get();
        else
            return nullptr;
        
    }

} //extern "C"
