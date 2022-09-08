#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPU_h

#include "DataFormats/SoACommon.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

#include "CUDADataFormats/PortableHostCollection.h"
#include "CUDADataFormats/PortableDeviceCollection.h"

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED = 150;
  constexpr unsigned int MAX_LINK = 48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC = 8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  constexpr unsigned int MAX_SIZE_BYTE_BOOL = MAX_SIZE * sizeof(unsigned char);
}  // namespace pixelgpudetails

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
GENERATE_SOA_LAYOUT(SiPixelFedCablingMapLayout,
                    SOA_COLUMN(uint32_t, fed),
                    SOA_COLUMN(uint32_t, link),
                    SOA_COLUMN(uint32_t, roc),
                    SOA_COLUMN(uint32_t, RawId),
                    SOA_COLUMN(uint32_t, rocInDet),
                    SOA_COLUMN(uint32_t, moduleId),
                    SOA_COLUMN(uint8_t, badRocs),
                    SOA_SCALAR(uint32_t, size))

using PHC_SiPixelFedCablingMap = PortableHostCollection<SiPixelFedCablingMapLayout<>>;
using PDC_SiPixelFedCablingMap = PortableDeviceCollection<SiPixelFedCablingMapLayout<>>;

#endif
