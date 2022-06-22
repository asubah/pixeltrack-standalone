#ifndef DataFormats_SiPixelDigi_interface_PixelErrorsSoA_h
#define DataFormats_SiPixelDigi_interface_PixelErrorsSoA_h

#include <map>
#include <vector>
#include <cstdint>

#include "DataFormats/SiPixelRawDataError.h"
#include "DataFormats/SoACommon.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

GENERATE_SOA_LAYOUT(PixelErrorCompactSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(uint32_t, rawId),
                      SOA_COLUMN(uint32_t, word),
                      SOA_COLUMN(uint8_t, errorType),
                      SOA_COLUMN(uint8_t, fedId),
                      SOA_SCALAR(uint32_t, count))

using PixelFormatterErrors = std::map<uint32_t, std::vector<SiPixelRawDataError>>;

#endif  // DataFormats_SiPixelDigi_interface_PixelErrorsSoA_h
