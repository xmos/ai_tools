// Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
//#include <Python.h>
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
        return  new flexbuffers::Builder();
    }

    const char *builder_get_buffer(flexbuffers::Builder *fbb)
    {
        std::vector<uint8_t> buf = fbb->GetBuffer();
        std::string str(reinterpret_cast<const char *>(&buf[0]), buf.size());
        return str.c_str();
    }

    void builder_int(flexbuffers::Builder *fbb, int64_t val)
    {
        fbb->Int(val);
    }

    void builder_finish(flexbuffers::Builder *fbb)
    {
        fbb->Finish();
    }

    // // ************************
    // // std::vector API
    // // ************************
    // std::vector<int> *new_vector()
    // {
    //     return new std::vector<int>;
    // }
    // void delete_vector(std::vector<int> *v)
    // {
    //     delete v;
    // }
    // int vector_size(std::vector<int> *v)
    // {
    //     return v->size();
    // }
    // int vector_get(std::vector<int> *v, int i)
    // {
    //     return v->at(i);
    // }
    // void vector_erase(std::vector<int> *v, int i)
    // {
    //    v->erase(v->begin()+i);
    // }
    // void vector_push_back(std::vector<int> *v, int i)
    // {
    //     v->push_back(i);
    // }

    // ************************
    // XCOREModel API
    // ************************

    // XCOREModel *model_import(const char *filename)
    // {
    //     XCOREModel *model = new XCOREModel();
    //     if (model->Import(filename)) 
    //         return model;
    //     else
    //         return nullptr;
    // }

    // PyObject *model_get_subgraphs(XCOREModel *m)
    // {
    //     // return m->GetModel()->subgraphs;
    //     //m->GetModel();
    //     std::cout << "000" << std::endl;
    //     //std::cout << (long)model.get() << std::endl;
    //     //std::cout << (long)model->GetModel() << std::endl;
    //     //tflite::ModelT *modelT = m->GetModel();
    //     //std::cout << "000" << std::endl;
    //     std::cout << (long)m->modelT.get() << std::endl;
    //     std::vector<std::unique_ptr<tflite::SubGraphT>> &subgraphs = m->modelT.get()->subgraphs;
    //     std::cout << "000" << " " << subgraphs.size() << std::endl;
    //     PyObject* listObj = PyList_New(subgraphs.size());
        
    //     std::cout << "000" << std::endl;
    //     int i = 0;
    //     for(auto const& subgraph : subgraphs)
    //     {
    //         std::cout << "111" << std::endl;
    //         PyObject *num = PyFloat_FromDouble( (double) 5.0);
    //         PyList_SET_ITEM(listObj, i++, num);
    //     }
    // //     if (!listObj) throw logic_error("Unable to allocate memory for Python list");
	// //     for (unsigned int i = 0; i < data.size(); i++) {
	// // 	    PyObject *num = PyFloat_FromDouble( (double) data[i]);
	// // 	if (!num) {
	// // 		Py_DECREF(listObj);
	// // 		throw logic_error("Unable to allocate memory for Python list");
	// // 	}
	// // 	PyList_SET_ITEM(listObj, i, num);
	// // }
    //     return listObj;
    // }

} //extern "C"
