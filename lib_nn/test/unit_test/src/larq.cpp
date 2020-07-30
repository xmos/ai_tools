#include "larq_compute_engine/core/packbits.h"

namespace compute_engine {
namespace core {

extern "C" void larq_ref_bsign(int8_t *input, uint32_t *output, size_t inputLength, int32_t zero_point)
{
    packbits_array<BitpackOrder::Canonical, std::int8_t, std::uint32_t>(input, inputLength, output, zero_point);
}

}  
}  
