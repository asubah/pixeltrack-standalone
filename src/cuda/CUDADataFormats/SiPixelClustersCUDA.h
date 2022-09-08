#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/cudaCompat.h"

#include "DataFormats/SoACommon.h"
#include "DataFormats/SoALayout.h"
#include "DataFormats/SoAView.h"

#include "CUDADataFormats/PortableDeviceCollection.h"

#include <cuda_runtime.h>

GENERATE_SOA_LAYOUT(SiPixelClustersCUDALayout,
                    SOA_COLUMN(uint32_t, moduleStart),
                    SOA_COLUMN(uint32_t, clusInModule),
                    SOA_COLUMN(uint32_t, moduleId),
                    SOA_COLUMN(uint32_t, clusModuleStart))

using PDC_SiPixelClustersCUDA = PortableDeviceCollection<SiPixelClustersCUDALayout<>>;

class SiPixelClustersCUDA {
public:
  SiPixelClustersCUDA() = default;
  explicit SiPixelClustersCUDA(size_t maxClusters, cudaStream_t stream);
  ~SiPixelClustersCUDA() = default;

  SiPixelClustersCUDA(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA &operator=(const SiPixelClustersCUDA &) = delete;
  SiPixelClustersCUDA(SiPixelClustersCUDA &&) = default;
  SiPixelClustersCUDA &operator=(SiPixelClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  PDC_SiPixelClustersCUDA::ConstView const view() const { return this->pdc.view(); }
  PDC_SiPixelClustersCUDA::View view() { return this->pdc.view(); }

//  std::byte const *moduleStart() const {
//    return this->pdc.buffer().get();
//  }
//
//  std::byte const *clusInModule() const {
//    auto offset = sizeof(uint32_t) * this->nClusters();
//    return this->pdc.buffer().get() + offset;
//  }

private:
  PDC_SiPixelClustersCUDA pdc;

  uint32_t nClusters_h;
};

#endif
