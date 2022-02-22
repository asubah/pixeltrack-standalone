#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include "CondFormats/pixelCPEforGPU.h"
#include "KokkosDataFormats/TrackingRecHit2DKokkos.h"

namespace testTrackingRecHit2DKokkos {

  template <typename MemorySpace>
  void fill(const Kokkos::View<TrackingRecHit2DSOAView, MemorySpace, RestrictUnmanaged>& hits) {
    assert(hits.data());
    auto hits_ = &hits();

    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1024), KOKKOS_LAMBDA(const size_t i) {
          assert(hits_->nHits() == 200);
          if (i > 200)
            return;
        });

    return;
  }

  template <typename MemorySpace>
  void verify(const Kokkos::View<TrackingRecHit2DSOAView, MemorySpace, RestrictUnmanaged>& hits) {
    assert(hits.data());

    auto const hits_ = &hits();

    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<KokkosExecSpace>(0, 1024), KOKKOS_LAMBDA(const size_t i) {
          assert(hits_->nHits() == 200);
          if (i > 200)
            return;
        });

    return;
  }

  template <typename MemorySpace>
  void runKernels(const Kokkos::View<TrackingRecHit2DSOAView, MemorySpace, RestrictUnmanaged>& hits) {
    assert(hits.data());

    fill(hits);
    verify(hits);
  }
}  // namespace testTrackingRecHit2DKokkos

namespace testTrackingRecHit2DKokkos {
  template <typename MemorySpace>
  void runKernels(const Kokkos::View<TrackingRecHit2DSOAView, MemorySpace, RestrictUnmanaged>& hits);

}

int main() {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});

  {
    auto nHits = 200;

    Kokkos::View<pixelCPEforGPU::ParamsOnGPU, KokkosDeviceMemSpace> _cpeParams("cpeparams");
    Kokkos::View<uint32_t*, KokkosDeviceMemSpace> _hitsModuleStart("hitsModuleStart", 1);

    TrackingRecHit2DKokkos<KokkosDeviceMemSpace> tkhit(nHits, _cpeParams, _hitsModuleStart, KokkosExecSpace());

    testTrackingRecHit2DKokkos::runKernels<KokkosExecSpace>(tkhit.mView());
  }

  return 0;
}
