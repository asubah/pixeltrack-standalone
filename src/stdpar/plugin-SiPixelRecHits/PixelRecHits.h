#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/BeamSpot.h"
#include "CUDADataFormats/SiPixelClusters.h"
#include "CUDADataFormats/SiPixelDigis.h"
#include "CUDADataFormats/TrackingRecHit2D.h"

namespace pixelgpudetails {

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel() = default;
    ~PixelRecHitGPUKernel() = default;

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    TrackingRecHit2DCUDA makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                       SiPixelClustersCUDA const& clusters_d,
                                       BeamSpot const& bs_d,
                                       pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                       cudaStream_t stream) const;
  };
}  // namespace pixelgpudetails

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
