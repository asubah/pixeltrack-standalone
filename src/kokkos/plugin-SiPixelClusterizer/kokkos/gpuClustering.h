#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h

#include <cstdint>
#include <cstdio>

#include "Geometry/phase1PixelTopology.h"
#include "KokkosCore/hintLightWeight.h"
#include "KokkosCore/HistoContainer.h"
#include "KokkosCore/atomic.h"
#include "KokkosCore/memoryTraits.h"
#include "KokkosDataFormats/gpuClusteringConstants.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuClustering {

#ifdef GPU_DEBUG
    __device__ uint32_t gMaxHit = 0;
#endif

    KOKKOS_INLINE_FUNCTION void countModules(
        const Kokkos::View<uint16_t const*, KokkosDeviceMemSpace, RestrictUnmanaged>& id,
        const Kokkos::View<uint32_t*, KokkosDeviceMemSpace, RestrictUnmanaged>& moduleStart,
        const Kokkos::View<int32_t*, KokkosDeviceMemSpace, RestrictUnmanaged>& clusterId,
        int numElements,
        const size_t index) {
      clusterId[index] = index;
      if (::gpuClustering::InvId == id[index])
        return;
      int j = index - 1;
      while (j >= 0 and id[j] == ::gpuClustering::InvId)
        --j;
      if (j < 0 or id[j] != id[index]) {
        // boundary... replacing atomicInc with explicit logic
        auto loc = cms::kokkos::atomic_fetch_add(&moduleStart(0), 1U);
        assert(moduleStart(0) < ::gpuClustering::MaxNumModules);
        moduleStart(loc + 1) = index;
      }
    }
  }  // namespace gpuClustering
}  // namespace KOKKOS_NAMESPACE

namespace KOKKOS_NAMESPACE::gpuClustering {
  //  __launch_bounds__(256,4)
  inline void findClus(
      const Kokkos::View<const uint16_t*, KokkosDeviceMemSpace, RestrictUnmanaged>& id,  // module id of each pixel
      const Kokkos::View<const uint16_t*, KokkosDeviceMemSpace, RestrictUnmanaged>& x,  // local coordinates of each pixel
      const Kokkos::View<const uint16_t*, KokkosDeviceMemSpace, RestrictUnmanaged>& y,  //
      const Kokkos::View<const uint32_t*, KokkosDeviceMemSpace, RestrictUnmanaged>&
          moduleStart,  // index of the first pixel of each module
      const Kokkos::View<uint32_t*, KokkosDeviceMemSpace, RestrictUnmanaged>&
          nClustersInModule,  // output: number of clusters found in each module
      const Kokkos::View<uint32_t*, KokkosDeviceMemSpace, RestrictUnmanaged>&
          moduleId,                                                                  // output: module id of each module
      const Kokkos::View<int*, KokkosDeviceMemSpace, RestrictUnmanaged>& clusterId,  // output: cluster id of each pixel
      int numElements,
      Kokkos::TeamPolicy<KokkosExecSpace>& teamPolicy,
      KokkosExecSpace const& execSpace) {
    constexpr int maxPixInModule = 4000;
    constexpr auto nbins = phase1PixelTopology::numColsInModule + 2;  //2+2;
    using Hist = cms::kokkos::HistoContainer<uint16_t, nbins, maxPixInModule, 9, uint16_t>;

    using member_type = Kokkos::TeamPolicy<KokkosExecSpace>::member_type;
    using shared_team_view = Kokkos::View<uint32_t, KokkosExecSpace::scratch_memory_space, RestrictUnmanaged>;
    using HistView = Kokkos::View<Hist, KokkosExecSpace::scratch_memory_space, RestrictUnmanaged>;
    using SizeView = Kokkos::View<int, KokkosExecSpace::scratch_memory_space, RestrictUnmanaged>;
    size_t shared_view_bytes = shared_team_view::shmem_size() + HistView::shmem_size() + SizeView::shmem_size();

    int shared_view_level = 0;
    Kokkos::parallel_for(
        "findClus",
        hintLightWeight(teamPolicy.set_scratch_size(shared_view_level, Kokkos::PerTeam(shared_view_bytes))),
        KOKKOS_LAMBDA(const member_type& teamMember) {
          if (teamMember.league_rank() >= static_cast<int>(moduleStart(0)))
            return;

          int firstPixel = moduleStart(1 + teamMember.league_rank());
          auto thisModuleId = id(firstPixel);
          assert(thisModuleId < ::gpuClustering::MaxNumModules);

          auto first = firstPixel + teamMember.team_rank();

          // find the index of the first pixel not belonging to this module (or invalid)
          SizeView d_msize(teamMember.team_scratch(shared_view_level));
          d_msize() = numElements;
          teamMember.team_barrier();

          // skip threads not associated to an existing pixel
          for (int i = first; i < numElements; i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (id(i) != thisModuleId) {  // find the first pixel in a different module
              cms::kokkos::atomic_min_fetch(&d_msize(), i);
              break;
            }
          }

          //init hist  (ymax=416 < 512 : 9bits)
          constexpr int loop_count = Hist::totbins();
          HistView d_hist(teamMember.team_scratch(shared_view_level));
          Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, loop_count),
                               [&](const int index) { d_hist().off[index] = 0; });
          teamMember.team_barrier();

          assert((d_msize() == numElements) or ((d_msize() < numElements) and (id(d_msize()) != thisModuleId)));

          // limit to maxPixInModule  (FIXME if recurrent (and not limited to simulation with low threshold) one will need to implement something cleverer)
          Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
            if (d_msize() - firstPixel > maxPixInModule) {
              printf("too many pixels in module %d: %d > %d\n", thisModuleId, d_msize() - firstPixel, maxPixInModule);
              d_msize() = maxPixInModule + firstPixel;
            }
          });

          teamMember.team_barrier();
          assert(d_msize() - firstPixel <= maxPixInModule);

          // fill histo
          for (int i = first; i < d_msize(); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            d_hist().count(y(i));
          }

          teamMember.team_barrier();
          d_hist().finalize(teamMember);
          teamMember.team_barrier();

          for (int i = first; i < d_msize(); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            d_hist().fill(y(i), i - firstPixel);
          }

          const uint32_t hist_size = d_hist().size();

#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
          const uint32_t maxiter = hist_size;
          // When compiling with gcc directly, the VLA compiles. When compiling via nvcc, it doesn't.
          // GPU backend uses 256 threads per team, therefore the
          // effective maximum number of iterations is 16*256 for
          // Serial case.
          // TODO: maybe this works well-enough for PTHREAD case too (even if it is number of threads/team too large)?
          constexpr uint32_t maxiterSize = 16 * 256;
#else
          constexpr uint32_t maxiter = 16;
          constexpr uint32_t maxiterSize = maxiter;
#endif

          constexpr int maxNeighbours = 10;
          assert((hist_size / teamMember.team_size()) <= maxiterSize);
          // nearest neighbour
          uint16_t nn[maxiterSize][maxNeighbours];
          uint8_t nnn[maxiterSize];
          for (uint32_t k = 0; k < maxiter; ++k) {
            nnn[k] = 0;
          }

          teamMember.team_barrier();  // for hit filling!

          // fill NN
          for (uint32_t j = teamMember.team_rank(), k = 0U; j < hist_size; j += teamMember.team_size(), ++k) {
            assert(k < maxiter);
            auto p = d_hist().begin() + j;
            auto i = *p + firstPixel;
            assert(id(i) != ::gpuClustering::InvId);
            assert(id(i) == thisModuleId);  // same module
            int be = Hist::bin(y(i) + 1);
            auto e = d_hist().end(be);
            ++p;
            assert(0 == nnn[k]);
            for (; p < e; ++p) {
              auto m = (*p) + firstPixel;
              assert(m != i);
              assert(int(y(m)) - int(y(i)) >= 0);
              assert(int(y(m)) - int(y(i)) <= 1);
              if (std::abs(int(x(m)) - int(x(i))) > 1)
                continue;
              auto l = nnn[k]++;
              assert(l < maxNeighbours);
              nn[k][l] = *p;
            }
          }

          // for each pixel, look at all the pixels until the end of the module;
          // when two valid pixels within +/- 1 in x or y are found, set their id to the minimum;
          // after the loop, all the pixel in each cluster should have the id equeal to the lowest
          // pixel in the cluster ( clus[i] == i ).
          int more = 1;
          int nloops = 0;
          while (more) {
            if (1 == nloops % 2) {
              for (uint16_t j = teamMember.team_rank(), k = 0U; j < d_hist().size(); j += teamMember.team_size(), ++k) {
                auto p = d_hist().begin() + j;
                auto i = *p + firstPixel;
                auto m = clusterId(i);
                while (m != clusterId(m)) {
                  m = clusterId(m);
                }
                clusterId(i) = m;
              }
            } else {
              more = 0;
              for (uint16_t j = teamMember.team_rank(), k = 0U; j < d_hist().size(); j += teamMember.team_size(), ++k) {
                auto p = d_hist().begin() + j;
                auto i = *p + firstPixel;
                for (uint16_t kk = 0; kk < nnn[k]; ++kk) {
                  auto l = nn[k][kk];
                  auto m = l + firstPixel;
                  assert(m != i);
                  auto old = cms::kokkos::atomic_fetch_min(&clusterId(m), clusterId(i));
                  if (old != clusterId(i)) {
                    // end the loop only if no changes were applied
                    more = 1;
                  }
                  cms::kokkos::atomic_fetch_min(&clusterId(i), old);
                }  // nnloop
              }    // pixel loop
            }
            ++nloops;
            teamMember.team_reduce(Kokkos::Sum<decltype(more)>(more));
          }  // end while

          shared_team_view foundClusters(teamMember.team_scratch(shared_view_level));
          foundClusters() = 0;
          teamMember.team_barrier();

          // find the number of different clusters, identified by a pixels with clus[i] == i;
          // mark these pixels with a negative id.
          for (int i = first; i < d_msize(); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (clusterId(i) == i) {
              auto old = cms::kokkos::atomic_fetch_add(&foundClusters(), 1U);
              assert(foundClusters() < 0xffffffff);
              clusterId(i) = -(old + 1);
            }
          }
          teamMember.team_barrier();

          // propagate the negative id to all the pixels in the cluster.
          for (int i = first; i < d_msize(); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId)  // skip invalid pixels
              continue;
            if (clusterId(i) >= 0) {
              // mark each pixel in a cluster with the same id as the first one
              clusterId(i) = clusterId(clusterId(i));
            }
          }
          teamMember.team_barrier();

          // adjust the cluster id to be a positive value starting from 0
          for (int i = first; i < d_msize(); i += teamMember.team_size()) {
            if (id(i) == ::gpuClustering::InvId) {  // skip invalid pixels
              clusterId(i) = -9999;
              continue;
            }
            clusterId(i) = -clusterId(i) - 1;
          }
          teamMember.team_barrier();

          Kokkos::single(Kokkos::PerTeam(teamMember), [&]() {
            nClustersInModule(thisModuleId) = foundClusters();
            moduleId(teamMember.league_rank()) = thisModuleId;
          });
        });
  }  // end findClus()
}  // namespace KOKKOS_NAMESPACE::gpuClustering
#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuClustering_h
