#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigisHostLayout_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigisHostLayout_h

#include "DataFormats/SoACommon.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

GENERATE_SOA_LAYOUT(SiPixelDigisHostLayout,
                      SOA_COLUMN(uint16_t, adc),
                      SOA_COLUMN(int32_t, clus),
                      SOA_COLUMN(uint32_t, pdigi),
                      SOA_COLUMN(uint32_t, rawIdArr),
                      SOA_SCALAR(uint32_t, nModules),
                      SOA_SCALAR(uint32_t, nDigis))

using PHC_SiPixelDigis = PortableHostCollection<SiPixelDigisHostLayout<>>;

#endif  // DataFormats_SiPixelDigi_interface_SiPixelDigisHosHosttLayout_h
