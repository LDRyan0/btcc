// NOTE: includes from previous btcc.cu
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#include <bifrost/array.h>
#include <bifrost/common.h>
#include "utils.hpp"
#include "cuda.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda_fp16.h>
#include "btcc.h"
#include "libtcc/Correlator.h"


// NOTE: additional includes from xcorr_lite.cu
#include <bifrost/ring.h>
#include <bifrost/xcorr_lite.h>

// NOTE: required by TCC's SimpleExample.cu
//#include "test/Common/ComplexInt4.h"

#include <complex>
#include <iostream>

#define NR_BITS 16

#define DCP_DEBUG
#define NR_CHANNELS 480
#define NR_POLARIZATIONS 2
#define NR_SAMPLES_PER_CHANNEL 3072
#define NR_RECEIVERS 576
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 64
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))


inline void checkCudaCall(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::cerr << "error " << error << std::endl;
    exit(1);
  }
}

BFstatus launch_tcc(BFarray *data_in, BFarray *data_out, int reset)
{
  #if NR_BITS == 4
  typedef complex_int4_t	      Sample;
  typedef std::complex<int32_t> Visibility;
  #elif NR_BITS == 8
  typedef std::complex<int8_t>  Sample;
  typedef std::complex<int32_t> Visibility;
  #elif NR_BITS == 16
  typedef std::complex<__half>  Sample;
  typedef std::complex<float>   Visibility;
  #endif


  // TODO: if this is a preprocessor directive then how will this typedef work at runtime?
  typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
  typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

  // typedef Sample Samples[n_chan][n_samp_per_chan / n_time_per_block][n_ant][n_pol][n_time_per_block];
  // typedef Visibility Visibilities[n_chan][(n_ant*(n_ant+1)) / 2][n_pol][n_pol];



  // get the parameters from shape of input
  int n_chan = data_in->shape[0]; // Frequency 
  int n_ant = data_in->shape[2]; // Antenna
  int n_pol = data_in->shape[3]; // Polarization
  int n_time_per_block = data_in->shape[4]; // Fine time
  int n_samp_per_chan = data_in->shape[1]*n_time_per_block; // Total samples (in time)

  // sanity check for comparing hardcoded values vs. dyanmically read
  printf("CONSTANTS: %d %d %d %d %d \n", NR_CHANNELS, NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK, NR_RECEIVERS, NR_POLARIZATIONS, NR_TIMES_PER_BLOCK);
  printf("  READ IN: %d %d %d %d %d \n", n_chan, n_samp_per_chan / n_time_per_block, n_ant, n_pol, n_time_per_block);

  cudaStream_t stream;
  Sample* samples = (Sample *)data_in->data;
  Visibility* visibilities = (Visibility *)data_out->data;
  
  printf("Creating CUDA stream...\n");
  checkCudaCall(cudaStreamCreate(&stream));

  // TODO: When do we actually compile the correlator in the Bifrost pipeline?
  // TODO: How does this work with pipeline.run()
  printf("\tCreating correlator...\n");
  // tcc::Correlator correlator(NR_BITS, NR_RECEIVERS, NR_CHANNELS, NR_SAMPLES_PER_CHANNEL, NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);
  // TODO: Check why NR_RECEIVERS_PER_BLOCK is 64 by default and how this comes into play?
  tcc::Correlator correlator(16, n_ant, n_chan, n_samp_per_chan, n_pol, 64);

  printf("\tPerforming correlation...\n");
  correlator.launchAsync((CUstream) stream, (CUdeviceptr) visibilities, (CUdeviceptr) samples);

  printf("\tSynchronising device...\n");
  checkCudaCall(cudaDeviceSynchronize());

  return BF_STATUS_SUCCESS;
}

// old xcorr_lite.cu code
    // BFstatus XcorrLite(BFarray *bf_data, BFarray *bf_xcorr, int reset)
    // {
    //     int* data = (int *)bf_data->data;
    //     float* xcorr = (float *)bf_xcorr->data;

    //     int H = bf_data->shape[0]; // Heap (slow time axis)
    //     int F = bf_data->shape[1]; // Frequency
    //     int N = bf_data->shape[2]; // Antenna
    //     int T = bf_data->shape[3]; // Fine time
        
    //     //printf("ispan dims F: %d N: %d T: %d\n", F, N, T);
    //     launch_xcorr_lite(data, xcorr, H, F, N, T, reset);
        
    //     BF_CHECK_CUDA(cudaGetLastError(), BF_STATUS_DEVICE_ERROR);

    //     return BF_STATUS_SUCCESS;
    // }
