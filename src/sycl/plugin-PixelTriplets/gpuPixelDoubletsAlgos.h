#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "DataFormats/approx_atan2.h"
#include "SYCLCore/VecArray.h"
#include "SYCLCore/sycl_assert.h"
#include "SYCLCore/AtomicPairCounter.h"
#include "SYCLCore/syclAtomic.h"
#include "SYCLCore/printf.h"

#include "CAConstants.h"
#include "GPUCACell.h"

// #define GPU_DEBUG

namespace gpuPixelDoublets {

  using sycl::abs;
  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __attribute__((always_inline)) void doubletsFromHisto(uint8_t const* __restrict__ layerPairs,
                                                        uint32_t nPairs,
                                                        GPUCACell* cells,
                                                        uint32_t* nCells,
                                                        CellNeighborsVector* cellNeighbors,
                                                        CellTracksVector* cellTracks,
                                                        TrackingRecHit2DSOAView const& __restrict__ hh,
                                                        GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                                        int16_t const* __restrict__ phicuts,
                                                        float const* __restrict__ minz,
                                                        float const* __restrict__ maxz,
                                                        float const* __restrict__ maxr,
                                                        bool ideal_cond,
                                                        bool doClusterCut,
                                                        bool doZ0Cut,
                                                        bool doPtCut,
                                                        uint32_t maxNumOfDoublets,
                                                        sycl::nd_item<3> item) {
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true
    constexpr int minYsizeB1 = 36;
    constexpr int minYsizeB2 = 28;
    constexpr int maxDYsize12 = 28;
    constexpr int maxDYsize = 20;
    constexpr int maxDYPred = 20;
    constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"

    bool isOuterLadder = ideal_cond;

    using Hist = TrackingRecHit2DSOAView::Hist;

    auto const& __restrict__ hist = hh.phiBinner();
    uint32_t const* __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);

    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = CAConstants::maxNumberOfLayerPairs();

    // SYCL_BUG_ multi_ptrs don't work, number of tracks is zero with them :(
    // auto innerLayerCumulativeSizebuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[nPairsMax]>(item.get_group());
    // uint32_t* innerLayerCumulativeSize = (uint32_t*)innerLayerCumulativeSizebuff.get();
    // auto ntotbuff = sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(item.get_group());
    // uint32_t* ntot = (uint32_t*)ntotbuff.get();

    assert(nPairs <= nPairsMax);
    uint32_t innerLayerCumulativeSize[nPairsMax];
    // const uint32_t* const ntot = innerLayerCumulativeSize + nPairs - 1; // THIS DOESN'T WORK

    innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
    for (uint32_t i = 1; i < nPairs; ++i) {
      innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(layerPairs[2 * i]);
    }
    const uint32_t ntot = innerLayerCumulativeSize[nPairs - 1];  // THIS WORKS

    // x runs faster
    auto idy = item.get_group(1) * item.get_local_range().get(1) + item.get_local_id(1);
    auto first = item.get_local_id(2);
    auto stride = item.get_local_range().get(2);

    uint32_t pairLayerId = 0;  // cannot go backward

    for (auto j = idy; j < ntot; j += item.get_local_range().get(1) * item.get_group_range(1)) {
      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;  // move to lower_bound ??

      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = layerPairs[2 * pairLayerId];
      uint8_t outer = layerPairs[2 * pairLayerId + 1];
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

      assert(i >= offsets[inner]);
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our sycl thread, now do the job
      auto mi = hh.detectorIndex(i);
      if (mi > 2000)
        continue;  // invalid

      /*maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */

      auto mez = hh.zGlobal(i);
      if (mez < minz[pairLayerId] || mez > maxz[pairLayerId])
        continue;

      int16_t mes = -1;  // make compiler happy
      if (doClusterCut) {
        // if ideal treat inner ladder as outer
        if (inner == 0)
          assert(mi < 96);
        isOuterLadder = ideal_cond ? true : 0 == (mi / 8) % 2;  // only for B1/B2/B3 B4 is opposite, FPIX:noclue...

        // in any case we always test mes>0 ...
        mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;

        if (inner == 0 && outer > 3)  // B1 and F1
          if (mes > 0 && mes < minYsizeB1)
            continue;                 // only long cluster  (5*8)
        if (inner == 1 && outer > 3)  // B2 and F1
          if (mes > 0 && mes < minYsizeB2)
            continue;
      }
      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      // all cuts: true if fails
      constexpr float z0cut = 12.f;      // cm
      constexpr float hardPtCut = 0.5f;  // GeV
      constexpr float minRadius =
          hardPtCut * 87.78f;  // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f * minRadius * minRadius;
      auto ptcut = [&](int j, int16_t idphi) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        auto dphi = short2phi(idphi);
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        auto dr = ro - mer;
        return dr > maxr[pairLayerId] || dr < 0 || abs((mez * ro - mer * zo)) > z0cut * dr;
      };

      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer < 4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner == 0 ? maxDYsize12 : maxDYsize;
        // in the barrel cut on difference in size
        // in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
        // FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        return onlyBarrel ? mes > 0 && so > 0 && int(abs(so - mes)) > dy
                          : (inner < 4) && mes > 0 &&
                                abs(mes - int(abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5f)) > maxDYPred;
      };

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep - iphicut));
      auto kh = Hist::bin(int16_t(mep + iphicut));
      auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };
      // bool piWrap = abs(kh-kl) > Hist::nbins()/2;
      // if (item.get_local_id(1)==0)
      //   printf("%d %d %d %d\n", kl, kh, mep, iphicut);

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif

      auto khh = kh;
      incr(khh);
      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += hist.size(kk + hoff);
#endif
        auto const* __restrict__ p = hist.begin(kk + hoff);
        auto const* __restrict__ e = hist.end(kk + hoff);
        p += first;
        for (; p < e; p += stride) {
          auto oi = *(p);
          assert(oi >= offsets[outer]);
          assert(oi < offsets[outer + 1]);
          auto mo = hh.detectorIndex(oi);
          if (mo > 2000)
            continue;  //    invalid

          if (doZ0Cut && z0cutoff(oi))
            continue;
          auto mop = hh.iphi(oi);
          uint16_t idphi = abs(static_cast<int16_t>(mep - mop));  // different from the original CUDA version
          if (idphi > iphicut)
            continue;

          if (doClusterCut && zsizeCut(oi))
            continue;
          if (doPtCut && ptcut(oi, idphi))
            continue;

          auto ind = cms::sycltools::atomic_fetch_add<uint32_t>(nCells, (uint32_t)1);
          if (ind >= maxNumOfDoublets) {
            cms::sycltools::atomic_fetch_sub<uint32_t>(nCells, (uint32_t)1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }
      }

#ifdef GPU_DEBUG
      if (tooMany > 0)
        printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif
    }  // loop in block...
  }
}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoupletsAlgos_h
