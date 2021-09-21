namespace cms {
  namespace cuda {

    class ExecutionConfiguration {
      public:
      // constexpr ExecutionConfiguration() = default;

      //constexpr void construct(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
      //  m_gird = grid;
      //  m_block = block;
      //  m_shmem = shmem;
      //  m_stream = stream;
      //}

      // usehost_unique_ptr api or read from file or use compile time defined values
      template < typename T >
      ExecutionConfiguration(T kernel, int* threadsPerBlock, int* minGridSize = nullptr) : 
        m_threadsPerBlock((*threadsPerBlock)),
        m_minGridSize((*minGridSize))
      {
        // Use cuda occupencey API
        
        cudaOccupancyMaxPotentialBlockSize(minGridSize, threadsPerBlock, kernel, 0, 0);
                
        // if (blocks < minGridSize) {
        //   threadsPerBlock = (wordCounter + threadsPerBlock -1) / minGridSize;
        //   threadsPerBlock = ((threadsPerBlock + 32) / 32) * 32;
        //   blocks = minGridSize;
        // }
      }

      private:
        int m_threadsPerBlock;
        int m_minGridSize;
      // dim3 m_gird;
      // dim3 m_block;
      // size_t m_shmem;
      // cudaStream_t m_stream;
    };

  }  // namespace cuda
}  // namespace cms
