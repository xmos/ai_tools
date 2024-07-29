#include "Utils/TileRamSupport.h"

namespace mlir::xcore::utils {

std::vector<char> tileRamServerHeader() {
  // TODO: Change flash_t struct to mem_server_header_t
  // We are reusing the flash_t struct in lib_tflite_micro as the header
  // The header version is stored as one integer
  // There are four parameter integers in the flash_t struct
  // Altogether 20 bytes
  constexpr int headerSize = 20;
  std::vector<char> header(headerSize, 0);
  header[0] = 1;
  header[1] = 2;
  header[2] = ~1;
  header[3] = ~2;
  header[8] = headerSize;
  return header;
}

} // namespace mlir::xcore::utils
