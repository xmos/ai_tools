#ifndef XFORMER_UTILS_TILESUPPORT_H
#define XFORMER_UTILS_TILESUPPORT_H

#include <vector>

namespace mlir::xcore::utils {

/** Function that creates a tile_ram_header
 */
std::vector<char> tileRamHeader();

} // namespace mlir::xcore::utils

#endif // XFORMER_UTILS_TILESUPPORT_H
