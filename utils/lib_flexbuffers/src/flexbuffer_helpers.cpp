// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include <iostream>
#include <vector>

#include "flatbuffers/flexbuffers.h"

extern "C" {
// ************************
// flexbuffers::Builder API
//   see
//   https://github.com/google/flatbuffers/blob/master/include/flatbuffers/flexbuffers.h
//   see: https://google.github.io/flatbuffers/flexbuffers.html
// ************************

flexbuffers::Builder* new_builder() { return new flexbuffers::Builder(); }

size_t builder_start_map(flexbuffers::Builder* fbb, const char* key = nullptr) {
  if (key)
    return fbb->StartMap(key);
  else
    return fbb->StartMap();
}

size_t builder_end_map(flexbuffers::Builder* fbb, size_t size) {
  return fbb->EndMap(size);
}

size_t builder_start_vector(flexbuffers::Builder* fbb,
                            const char* key = nullptr) {
  if (key)
    return fbb->StartVector(key);
  else
    return fbb->StartVector();
}

size_t builder_end_vector(flexbuffers::Builder* fbb, size_t size, bool typed,
                          bool fixed) {
  return fbb->EndVector(size, typed, fixed);
}

void builder_clear(flexbuffers::Builder* fbb) { fbb->Clear(); }

void builder_int(flexbuffers::Builder* fbb, const char* key, int64_t val) {
  fbb->Int(key, val);
}

void builder_vector_int(flexbuffers::Builder* fbb, int64_t val) {
  fbb->Int(val);
}

void builder_uint(flexbuffers::Builder* fbb, const char* key, uint64_t val) {
  fbb->UInt(key, val);
}

void builder_vector_uint(flexbuffers::Builder* fbb, uint64_t val) {
  fbb->UInt(val);
}

void builder_bool(flexbuffers::Builder* fbb, const char* key, bool val) {
  fbb->Bool(key, val);
}

void builder_vector_bool(flexbuffers::Builder* fbb, bool val) {
  fbb->Bool(val);
}

void builder_float(flexbuffers::Builder* fbb, const char* key, float val) {
  fbb->Float(key, val);
}

void builder_vector_float(flexbuffers::Builder* fbb, float val) {
  fbb->Float(val);
}

void builder_string(flexbuffers::Builder* fbb, const char* key,
                    const char* val) {
  fbb->String(key, val);
}

void builder_vector_string(flexbuffers::Builder* fbb, const char* val) {
  fbb->String(val);
}

void builder_finish(flexbuffers::Builder* fbb) { fbb->Finish(); }

size_t builder_get_buffer(flexbuffers::Builder* fbb, char* buf) {
  std::vector<uint8_t> bytes = fbb->GetBuffer();

  for (std::size_t i = 0; i < bytes.size(); ++i) {
    buf[i] = bytes[i];
  }

  return bytes.size();
}

size_t parse_flexbuffer(const uint8_t* fb, size_t size, char* buf) {
  std::vector<uint8_t> bytes(fb, fb + size);
  std::string json;

  flexbuffers::GetRoot(bytes).ToString(true, true, json);
  strncpy(buf, json.c_str(), json.length());

  return json.length();
}

}  // extern "C"
