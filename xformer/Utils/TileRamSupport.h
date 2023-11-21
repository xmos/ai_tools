#ifndef XFORMER_UTILS_TILESUPPORT_H
#define XFORMER_UTILS_TILESUPPORT_H

#include <stdint.h>
#include <string>
#include <vector>

namespace mlir {
namespace xcore {
namespace utils {

/** Function that creates a tile_ram_header
 */
std::vector<char> tileRamHeader();

} // namespace utils
} // namespace xcore
} // namespace mlir

#endif // XFORMER_UTILS_TILESUPPORT_H