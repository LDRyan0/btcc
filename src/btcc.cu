/*
 * Copyright (c) 2021, The Bifrost Authors. All rights reserved.
 # Copyright (c) 2021, The University of New Mexico. All rights reserved.

 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of The Bifrost Authors nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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


// enable full debugging output
// #define DCP_DEBUG

// disable ALL sanity checks 
// ONLY USE FOR SLIGHT INCREASE IN PERFORMANCE ON KNOWN STABLE BUILDS
#define NO_CHECKS

// enable timing of the .launchAsync() function
#define TIME_CORR

#ifdef TIME_CORR
#include <fstream>
#endif

thread_local cudaStream_t g_cuda_stream = cudaStreamPerThread;

// Convert from a TCC bit depth to a Bifrost data type
inline BFdtype bf_dtype_from_tcc(int nr_bits) {
    switch(nr_bits) {
        case 4: return BF_DTYPE_CI4;
        case 8: return BF_DTYPE_CI8;
        case 16: return BF_DTYPE_CF16;
        default: return BF_DTYPE_CF32;
    }
}

class btcc_impl {
private:
    int _nbits;
    int _ntime;
    int _nchan;
    int _nstand;
    int _npol;
    int _ntime_per_block;
    
    tcc::Correlator* _tcc;
    cudaStream_t _stream;
    
public:
    btcc_impl() : _tcc(NULL), _stream(g_cuda_stream) {}
    ~btcc_impl() {
        cudaDeviceSynchronize();
        
        if(_tcc) {
           delete _tcc;
        }
    }
    inline int ntime() const { return _ntime; }
    inline int nchan() const { return _nchan; }
    inline int nstand() const { return _nstand; }
    inline int npol() const { return _npol; }
    inline int ntime_per_block() const { return _ntime_per_block; }
    inline int nbaseline() const { return (_nstand+1)*(_nstand/2); }
    inline BFdtype in_dtype() const { return bf_dtype_from_tcc(_nbits); }
    inline BFdtype out_dtype() const { return _nbits == 16 ? BF_DTYPE_CF32 : BF_DTYPE_CI32; }
    void init(int nbits, int ntime, int nchan, int nstand, int npol) {
        #ifdef DCP_DEBUG
        printf("Initialising tcc...\n");
        printf("\t          _nbits = %d\n", nbits);
        printf("\t          _ntime = %d\n", ntime);
        printf("\t          _nchan = %d\n", nchan);
        printf("\t         _nstand = %d\n", nstand);
        printf("\t           _npol = %d\n", npol);
        printf("\t_ntime_per_block = %d\n", 128 / nbits);
        #endif
        
        _nbits = nbits;
        _ntime = ntime;
        _nchan = nchan;
        _nstand = nstand;
        _npol = npol;
        _ntime_per_block = 128 / _nbits;
        
        #ifndef NO_CHECKS
        BF_ASSERT_EXCEPTION((_nbits == 4) || (_nbits == 8) || (_nbits == 16), BF_STATUS_UNSUPPORTED_DTYPE);
        BF_ASSERT_EXCEPTION(_ntime % _ntime_per_block == 0, BF_STATUS_UNSUPPORTED_SHAPE);
        #endif

        _tcc = new tcc::Correlator(_nbits, _nstand, _nchan, _ntime, _npol);
    }
    void set_stream(cudaStream_t stream) {
        cudaDeviceSynchronize();
        _stream = stream;
    }

    // TODO: since we removed _accumulate and _reorder is this even needed?
    void reset_state() {
        BF_ASSERT_EXCEPTION(_tcc, BF_STATUS_INVALID_STATE); 
        
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_MEM_OP_FAILED);
    }
    void exec(BFarray const* in, BFarray* out, BFbool dump) {
        
        #ifndef NO_CHECKS
        BF_ASSERT_EXCEPTION(_tcc, BF_STATUS_INVALID_STATE); 
        #endif
        
        #ifdef TIME_CORR
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        #endif

        #ifdef DCP_DEBUG
        printf("Launching correlator...\n");
        #endif

        // Launch the main TCC kernel
        (*_tcc).launchAsync((CUstream) _stream, (CUdeviceptr) out->data, (CUdeviceptr) in->data);
        
       
        #ifdef TIME_CORR
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
       
        // Print time to terminal
        // std::cout << "  CUDA | .launchAsync() time: " << milliseconds << " ms" << std::endl;
        
        // Write (nant, time_ms) to file
        std::ofstream myfile;
        myfile.open("cuda_results.csv", std::ofstream::app);
        
        myfile << milliseconds << "\n";
        myfile.close();
        #endif

        #ifndef NO_CHECKS
        BF_CHECK_CUDA_EXCEPTION(cudaGetLastError(), BF_STATUS_INTERNAL_ERROR);
        #endif
    }
};

BFstatus BTccCreate(btcc* plan_ptr) {
    #ifndef NO_CHECKS
    BF_ASSERT(plan_ptr, BF_STATUS_INVALID_POINTER);
    #endif
    BF_TRY_RETURN_ELSE(*plan_ptr = new btcc_impl(),
                       *plan_ptr = 0);
}

BFstatus BTccInit(btcc  plan,
                  int   nbits,
                  int   ntime,
                  int   nchan,
                  int   nstand,
                  int   npol) {
    #ifndef NO_CHECKS
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    #endif
    BF_TRY_RETURN(plan->init(nbits, ntime, nchan, nstand, npol));
}

BFstatus BTccSetStream(btcc        plan,
                       void const* stream) {
        #ifndef NO_CHECKS
        BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        BF_ASSERT(stream, BF_STATUS_INVALID_POINTER);
        #endif
        BF_TRY_RETURN(plan->set_stream(*(cudaStream_t*)stream));
}

BFstatus BTccResetState(btcc        plan) {
        #ifndef NO_CHECKS
        BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
        #endif
        BF_TRY_RETURN(plan->reset_state());
}

// Porting existing wrapper to new (lightweight) wrapper required the 
// following change in input and output shape:
// OLD (Jayce): 
//      [NR_TIME][NR_CHANNELS][NR_RECEIVERS*NR_POLARISATIONS]
//      [NR_CHANNELS][NR_BASELINES*NR_POLARISATIONS*NR_POLARISATIONS]
// NEW (Liam) : 
//      [NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK]
//      [NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS]
BFstatus BTccExecute(btcc           plan,
                     BFarray const* in,
                     BFarray*       out,
                     BFbool         dump) {
    #ifndef NO_CHECKS
    BF_ASSERT(plan, BF_STATUS_INVALID_POINTER);
    BF_ASSERT(in,   BF_STATUS_INVALID_POINTER);
  	BF_ASSERT(out,  BF_STATUS_INVALID_POINTER);
    #endif
    
    #ifdef DCP_DEBUG
    printf("Checking input shape...\n");
    printf("(nchan, ntime / ntime_per_block, nstand, npol, ntime_per_block)\n");
    printf("\t(dim)            in->ndim: %8d ?= %d\n", in->ndim, 5);
    printf("\t(0)                 nchan: %8ld ?= %d\n", in->shape[0], plan->nchan());
    printf("\t(1) ntime/ntime_per_block: %8ld ?= %d\n", in->shape[1], plan->ntime() / plan->ntime_per_block());
    printf("\t(2)                nstand: %8ld ?= %d\n", in->shape[2], plan->nstand());
    printf("\t(3)                  npol: %8ld ?= %d\n", in->shape[3], plan->npol());
    printf("\t(4)       ntime_per_block: %8ld ?= %d\n", in->shape[4], plan->ntime_per_block());
    #endif
    
    #ifndef NO_CHECKS
    BF_ASSERT( in->ndim == 5, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[0] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[1] == plan->ntime() / plan->ntime_per_block(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[2] == plan->nstand(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[3] == plan->npol(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(in->shape[4] == plan->ntime_per_block(), BF_STATUS_INVALID_SHAPE);
    #endif

    #ifdef DCP_DEBUG
    printf("Checking output shape...\n");
    printf("(nchan, nbaselines, npol, npol)\n");
    printf("\t(dim)           out->ndim: %8d ?= %d\n", out->ndim, 4);
    printf("\t(0)                 nchan: %8ld ?= %d\n", out->shape[0], plan->nchan());
    printf("\t(1)             nbaseline: %8ld ?= %d\n", out->shape[1], plan->nbaseline());
    printf("\t(2)                  npol: %8ld ?= %d\n", out->shape[2], plan->npol());
    printf("\t(3)                  npol: %8ld ?= %d\n", out->shape[3], plan->npol());
    #endif

    #ifndef NO_CHECKS
    BF_ASSERT(out->ndim == 4, BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[0] == plan->nchan(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[1] == plan->nbaseline(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[2] == plan->npol(), BF_STATUS_INVALID_SHAPE);
    BF_ASSERT(out->shape[3] == plan->npol(), BF_STATUS_INVALID_SHAPE);
    #endif

    #ifdef DCP_DEBUG
    printf("Checking Bifrost data types....\n");
    printf("\t in: %d ?= %d\n", in->dtype, plan->in_dtype());
    printf("\tout: %d ?= %d\n", out->dtype, plan->out_dtype());
    #endif

    #ifndef NO_CHECKS
    BF_ASSERT(in->dtype == plan->in_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    BF_ASSERT(out->dtype == plan->out_dtype(), BF_STATUS_UNSUPPORTED_DTYPE);
    
    BF_ASSERT(space_accessible_from(in->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    BF_ASSERT(space_accessible_from(out->space, BF_SPACE_CUDA),
              BF_STATUS_UNSUPPORTED_SPACE);
    #endif
    
    // actually execute the correlation
    BF_TRY_RETURN(plan->exec(in, out, dump));
}

BFstatus BTccDestroy(btcc plan) {
    BF_ASSERT(plan, BF_STATUS_INVALID_HANDLE);
    delete plan;
    return BF_STATUS_SUCCESS;
}