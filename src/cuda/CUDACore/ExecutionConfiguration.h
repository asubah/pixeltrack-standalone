//#ifndef HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
//#define HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h

namespace cms {
  namespace cuda {

    class ExecutionConfiguration {
      public:

      //constexpr void construct(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
      template < typename T >
      ExecutionConfiguration(T kernel, int* blockSize, size_t dynamicSMemSize = 0, int  blockSizeLimit = 0) : 
        m_blockSize((*blockSize))
      {
        // Use cuda occupencey API
        int gridSize = 0;
        
        cudaOccupancyMaxPotentialBlockSize(&gridSize, blockSize, kernel, dynamicSMemSize, blockSizeLimit);
        // printf("gridSize = %d, blockSize = %d\n", gridSize, (*blockSize));
                
        // if (blocks < gridSize) {
        //   blockSize = (wordCounter + blockSize -1) / gridSize;
        //   blockSize = ((blockSize + 32) / 32) * 32;
        //   blocks = gridSize;
        // }
      }

      private:
        int m_blockSize;
        // int m_gridSize;
      // dim3 m_gird;
      // dim3 m_block;
      // size_t m_shmem;
      // cudaStream_t m_stream;
    };

  }  // namespace cuda
}  // namespace cms

//#endif  // HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
