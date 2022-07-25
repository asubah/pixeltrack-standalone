#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigisDeviceLayout_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigisDeviceLayout_h

#include "DataFormats/SoACommon.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

GENERATE_SOA_LAYOUT(SiPixelDigisDeviceLayout,
                    SOA_COLUMN(uint16_t, xx),
                    SOA_COLUMN(uint16_t, yy),
                    SOA_COLUMN(uint16_t, adc),
                    SOA_COLUMN(uint16_t, moduleInd),
                    SOA_COLUMN(int32_t, clus),
                    SOA_COLUMN(uint32_t, pdigi),
                    SOA_COLUMN(uint32_t, rawIdArr))

using PDC_SiPixelDigisDeviceLayout = PortableDeviceCollection<SiPixelDigisDeviceLayout<>>;

#endif  // DataFormats_SiPixelDigi_interface_SiPixelDigisDeviceLayout_h
