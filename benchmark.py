import sys
import time
import os
import numpy as np
import bifrost as bf
from btcc import Btcc
from bifrost.libbifrost import _bf

nbits = 16 # number of bits MUST be 16 for current ICRAR hardware (Tesla V100)
n_time_per_block = 128 // nbits # = 8

def test_correlate(nant, nchan, ntime, npol):
    # initialise the tensor core correlator
    tcc = Btcc()
    tcc.init(nbits, ntime, nchan, nant, npol)

    # Random int8 0->64
    tcc_input = bf.ndarray( shape=(nchan, ntime // n_time_per_block, nant, npol, n_time_per_block),
                        dtype='cf16',
                        space='system')

    tcc_input['re'] = np.random.randint(64, size=(nchan, ntime // n_time_per_block, nant, npol, n_time_per_block)).astype('float16')
    tcc_input['im'] = np.random.randint(64, size=(nchan, ntime // n_time_per_block, nant, npol, n_time_per_block)).astype('float16')
    tcc_input = tcc_input.copy(space='cuda')

    tcc_output = bf.ndarray(shape=(nchan, nant*(nant+1)//2, npol, npol),
                            dtype='cf32',
                            space='cuda')

    # execute TCC
    tcc_start = time.perf_counter()
    tcc.execute(tcc_input, tcc_output, True)
    tcc_end = time.perf_counter()
    tcc_time = tcc_end - tcc_start
    tcc_output_cpu = tcc_output.copy(space='system')

    # obtain accuracy results through cross-checking
    # correct, total = check_results(tcc_output_cpu, xcorr_output_cpu)
    # print(f'Results: {correct}/{total} correct\n')

    # delete the tcc object, this was required at some point w/ multiple iterations?
    del tcc

    # return tcc_time, xcorr_time
    return tcc_time


if __name__ == '__main__':
    # default parameters

    # MWA Spec
    # nchan = 3072
    # ntime = 
    # npol = 2
    # nant = 256

    # Romein's parameters
    # nchan = 480
    # ntime = 3072
    # npol = 2
    # nant = 256

    nchan = 240
    ntime = 2048
    npol = 2
    nant = 576
    nbaselines = (nant*(nant+1))//2
    
    # Remove cuda_results.csv file if for some reason it's there
    if os.path.exists('cuda_results.csv'):
        os.remove('cuda_results.csv')

    if (len(sys.argv) != 4):
        
        print("\nParameters")
        print("\tnant  =", nant)
        print("\tnpol  =", npol)
        print("\tnchan =", nchan)
        print("\tntime =", ntime, "\n")
        tcc_time = test_correlate(nant, nchan, ntime, npol)
        print(f'  btcc time: {tcc_time*1000:10.3f} ms')

        samples_per_second_per_chan = ntime / tcc_time
        max_samples_per_second = samples_per_second_per_chan * nchan
        max_bandwidth = max_samples_per_second / 2  # Nyquist sampling
        max_bandwidth_mhz = max_bandwidth / 1e6

        n_bit_ingest = 8 # assuming we get 8-bit data from the network card (we would then convert into 16 bit)
        max_data_rate = max_bandwidth_mhz * 2 * nchan * n_bit_ingest
        print(f' Max ingest: {max_data_rate/1000:8.3f} Gb/s')

        n_op_per_cmac = 8
        performance_tops = nbaselines * npol**2 * n_op_per_cmac * nchan / tcc_time * ntime / 1e12
        print(f'Performance: {performance_tops:8.3f} TOp/s')

    else:

        min_nant = int(sys.argv[1])
        step_nant = int(sys.argv[2])
        max_nant = int(sys.argv[3])

        num_trials = (max_nant - min_nant) // step_nant + 1
        results = np.zeros((num_trials, 5))
        trial = 0

        for nant in range(min_nant, max_nant + 1, step_nant):
            print(f'Testing: {nant}...')
            nbaselines = (nant*(nant+1))//2

            tcc_time = test_correlate(nant, nchan, ntime, npol)

            # Calculations of data rate and performance
            samples_per_second_per_chan = ntime / tcc_time
            max_samples_per_second = samples_per_second_per_chan * nchan
            max_bandwidth = max_samples_per_second / 2  # Nyquist sampling
            max_bandwidth_mhz = max_bandwidth / 1e6
    
            n_bit_ingest = 8 # 8-bit data from the network card, convert into 16 bit
            max_data_rate = max_bandwidth_mhz * 2 * nchan * n_bit_ingest
    
            n_op_per_cmac = 8
            performance_tops = nbaselines * npol**2 * n_op_per_cmac * nchan / tcc_time * ntime / 1e12

            results[trial, 0] = nant
            results[trial, 2] = tcc_time*1000 # ms
            results[trial, 3] = max_data_rate/1000 # Mega -> Giga
            results[trial, 4] = performance_tops

            trial += 1

        # if it exists obtain raw CUDA timing results form cuda_results.csv file
        if not os.path.exists('cuda_results.csv'):
            print('No CUDA results found: to obtain raw CUDA kernel timing compile wrapper with TIME_CORR identifier.')
        else:
            cuda_results = np.genfromtxt('cuda_results.csv')
            results[:,1] = cuda_results
            os.remove('cuda_results.csv')

        # Make results/ directory if not already there
        if not os.path.isdir('results'):
            print('Making new results folder...')
            os.mkdir(os.path.join(os.getcwd(), 'results'))

        # write results as .csv to "results/" subdirectory
        np.savetxt(f'results/N{min_nant}-{step_nant}-{max_nant}_T{ntime}_F{nchan}.csv', results, delimiter=',', 
            fmt=['%d', '%8f', '%8f', '%8f', '%8f'])       
