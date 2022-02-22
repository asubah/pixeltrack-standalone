#include "BrokenLineFitOnGPU.h"

#include "KokkosCore/hintLightWeight.h"

namespace KOKKOS_NAMESPACE {
  void HelixFitOnGPU::launchBrokenLineKernels(HitsView const* hv,
                                              uint32_t hitsInFit,
                                              uint32_t maxNumberOfTuples,
                                              KokkosExecSpace const& execSpace) {
    //  Fit internals
    auto hitsGPU_ptr = cms::kokkos::make_shared<double[], KokkosDeviceMemSpace>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>), execSpace);
    auto hits_geGPU_ptr = cms::kokkos::make_shared<float[], KokkosDeviceMemSpace>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f), execSpace);
    auto fast_fit_resultsGPU_ptr = cms::kokkos::make_shared<double[], KokkosDeviceMemSpace>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d), execSpace);

    auto hitsGPU = cms::kokkos::to_view(hitsGPU_ptr);
    auto hits_geGPU = cms::kokkos::to_view(hits_geGPU_ptr);
    auto fast_fit_resultsGPU = cms::kokkos::to_view(fast_fit_resultsGPU_ptr);

    // avoid capturing this by the lambdas
    auto const bField = bField_;
    auto tuples = tuples_d;
    auto tupleMultiplicity = tupleMultiplicity_d;
    auto outputSoa = outputSoa_d;

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets
      Kokkos::parallel_for(
          "kernelBLFastFit_3",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFastFit<3>(tuples, tupleMultiplicity, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 3, offset, i);
          });

      Kokkos::parallel_for(
          "kernelBLFit_3",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFit<3>(
                tupleMultiplicity, bField, outputSoa, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 3, offset, i);
          });

      // fit quads
      Kokkos::parallel_for(
          "kernelBLFastFit_4",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFastFit<4>(tuples, tupleMultiplicity, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 4, offset, i);
          });

      Kokkos::parallel_for(
          "kernelBLFit_4",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFit<4>(
                tupleMultiplicity, bField, outputSoa, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 4, offset, i);
          });

      if (fit5as4_) {
        // fit penta (only first 4)
        Kokkos::parallel_for(
            "kernelBLFastFit_4",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFastFit<4>(tuples, tupleMultiplicity, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });

        Kokkos::parallel_for(
            "kernelBLFit_4",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFit<4>(
                  tupleMultiplicity, bField, outputSoa, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });
      } else {
        // fit penta (all 5)
        Kokkos::parallel_for(
            "kernelBLFastFit_5",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFastFit<5>(tuples, tupleMultiplicity, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });

        Kokkos::parallel_for(
            "kernelBLFit_5",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFit<5>(
                  tupleMultiplicity, bField, outputSoa, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });
      }
    }  // loop on concurrent fits
  }
}  // namespace KOKKOS_NAMESPACE
