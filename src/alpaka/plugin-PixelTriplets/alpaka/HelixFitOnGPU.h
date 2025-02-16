#ifndef plugin_PixelTriplets_alpaka_HelixFitOnGPU_h
#define plugin_PixelTriplets_alpaka_HelixFitOnGPU_h

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaDataFormats/alpaka/PixelTrackAlpaka.h"
#include "AlpakaDataFormats/alpaka/TrackingRecHit2DAlpaka.h"

#include "../CAConstants.h"
#include "FitResult.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace Rfit {
    // in case of memory issue can be made smaller
    constexpr uint32_t maxNumberOfConcurrentFits() { return CAConstants::maxNumberOfTuples(); }
    constexpr uint32_t stride() { return maxNumberOfConcurrentFits(); }
    using Matrix3x4d = Eigen::Matrix<double, 3, 4>;
    using Map3x4d = Eigen::Map<Matrix3x4d, 0, Eigen::Stride<3 * stride(), stride()> >;
    using Matrix6x4f = Eigen::Matrix<float, 6, 4>;
    using Map6x4f = Eigen::Map<Matrix6x4f, 0, Eigen::Stride<6 * stride(), stride()> >;

    // hits
    template <int N>
    using Matrix3xNd = Eigen::Matrix<double, 3, N>;
    template <int N>
    using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()> >;
    // errors
    template <int N>
    using Matrix6xNf = Eigen::Matrix<float, 6, N>;
    template <int N>
    using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()> >;
    // fast fit
    using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()> >;

  }  // namespace Rfit

  class HelixFitOnGPU {
  public:
    using HitsView = TrackingRecHit2DSoAView;

    using Tuples = pixelTrack::HitContainer;
    using OutputSoA = pixelTrack::TrackSoA;

    using TupleMultiplicity = CAConstants::TupleMultiplicity;

    explicit HelixFitOnGPU(float bf, bool fit5as4) : bField_(bf), fit5as4_(fit5as4) {}
    ~HelixFitOnGPU() { deallocateOnGPU(); }

    void setBField(double bField) { bField_ = bField; }
    void launchRiemannKernels(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples);
    void launchBrokenLineKernels(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples);

    void launchRiemannKernels(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples, Queue &queue);
    void launchBrokenLineKernels(HitsView const *hv, uint32_t nhits, uint32_t maxNumberOfTuples, Queue &queue);

    void allocateOnGPU(Tuples const *tuples, TupleMultiplicity const *tupleMultiplicity, OutputSoA *outputSoA);
    void deallocateOnGPU();

  private:
    static constexpr uint32_t maxNumberOfConcurrentFits_ = Rfit::maxNumberOfConcurrentFits();

    // fowarded
    Tuples const *tuples_d = nullptr;
    TupleMultiplicity const *tupleMultiplicity_d = nullptr;
    OutputSoA *outputSoa_d;
    float bField_;

    const bool fit5as4_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // plugin_PixelTriplets_alpaka_HelixFitOnGPU_h
