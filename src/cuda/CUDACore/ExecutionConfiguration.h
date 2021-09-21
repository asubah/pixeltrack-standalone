//#ifndef HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
//#define HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h

namespace cms {
  namespace cuda {

    class ExecutionConfiguration {
      public:

      //constexpr void construct(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
      template < typename T >
      ExecutionConfiguration(T kernel, int* threads, int* minimum_blocks = nullptr) : 
        m_threads((*threads)),
        m_minimum_blocks((*minimum_blocks))
      {
        // Use cuda occupencey API
        
        cudaOccupancyMaxPotentialBlockSize(minimum_blocks, threads, kernel, 0, 0);
                
        // if (blocks < minimum_blocks) {
        //   threads = (wordCounter + threads -1) / minimum_blocks;
        //   threads = ((threads + 32) / 32) * 32;
        //   blocks = minimum_blocks;
        // }
      }

      private:
        int m_threads;
        int m_minimum_blocks;
      // dim3 m_gird;
      // dim3 m_block;
      // size_t m_shmem;
      // cudaStream_t m_stream;
    };

  }  // namespace cuda
}  // namespace cms

//#endif  // HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
