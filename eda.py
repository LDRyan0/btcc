import sys
import time
import os
import numpy as np
import bifrost as bf
import h5py
from btcc import Btcc
from bifrost.libbifrost import _bf

nchan = 1
nant = 256
npol = 2
ntime = 262144

nbits = 16 # number of bits MUST be 16 for current ICRAR hardware (Tesla V100)
n_time_per_block = 128 // nbits # = 8
file_path = "/group/director2183/data/test/eda2/2021_12_30_ch204_station_beam/merged/channel_cont_20211230_14676_0.hdf5"


h5 = h5py.File(file_path)
data = h5['chan_']['data'][:]

tcc_input = bf.ndarray( shape=(nchan, ntime // n_time_per_block, nant, npol, n_time_per_block),
                    dtype='cf16',
                    space='system')

tcc_output = bf.ndarray(shape=(nchan, nant*(nant+1)//2, npol, npol),
                        dtype='cf32',
                        space='cuda')


start = time.perf_counter()
data = data.reshape((1, 262144 // n_time_per_block, n_time_per_block,  256, 2))
data = data.transpose((0, 1, 3, 4, 2))

tcc_input['re'] = data['real'].astype('float16')
tcc_input['im'] = data['imag'].astype('float16')
tcc_input = tcc_input.copy(space='cuda')


# initialise the tensor core correlator
tcc = Btcc()
tcc.init(nbits, ntime, nchan, nant, npol)

# execute TCC
tcc_start = time.perf_counter()
tcc.execute(tcc_input, tcc_output, True)
tcc_end = time.perf_counter()
tcc_time = tcc_end - tcc_start
tcc_output_cpu = tcc_output.copy(space='system')   
print(f'RAW TCC:{tcc_time*1000} ms')
check1 = time.perf_counter()
print(f'1: Post correlation {(check1 - start)*1000} ms')


# reshape into correlation matrix upper triangle (nchan, nant, nant, npol, npol)
tcc_comp = np.zeros((nchan, nant, nant, npol, npol)).astype('complex64')
a, b = np.tril_indices(nant)
tcc_comp[:, a, b] = tcc_output_cpu

# reflect upper triangle onto lower triangle with conjugate for full correlation matrix
a, b = np.triu_indices(nant, 1)
tcc_comp[:, a, b] = np.conjugate(tcc_comp.swapaxes(1,2).swapaxes(3,4)[:, a, b])
check2 = time.perf_counter()
print(f'2: Post rearrange {(check2 - start)*1000} ms')

with h5py.File('output.hdf5', 'w') as h5o:
    h5o.create_dataset('data_baselines', data = tcc_output_cpu)
    h5o.create_dataset('data_matrix', data = tcc_comp)    
