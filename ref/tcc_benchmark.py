# The following script benchmarks TCC correlation from 1...max_nstand 


from bifrost import ndarray
from btcc import Btcc
import time
import numpy as np

bits_per_sample = 16
ntime_per_gulp = 256
nchan = 100
npol = 2

# range of antenna numbers, inclusive maximum
min_nstand = 32
max_nstand = 256
step_nstand = 4

# iterations per number of antenna
iter_per_nstand = 3

prog_start = time.perf_counter()
times = np.zeros((max_nstand-min_nstand)//step_nstand + 1)


f = open("times_N" + str(min_nstand) + "-" + str(step_nstand) + "-" + str(max_nstand) + "_T" + str(ntime_per_gulp) + "_F" + str(nchan) + ".csv", "w")

# iterate from 1..max_nstand
for nstand in range(min_nstand, max_nstand + 1, step_nstand):
    tcc = Btcc()
    # print(bits_per_sample, ntime_per_gulp, nchan, nstand, npol)
    tcc.init(bits_per_sample,
                ntime_per_gulp,
                nchan,
                nstand,
                npol)
    
    # used for averaging across iter_per_nstand number for each nstand
    total_time_ms = 0
    
    for iter in range(iter_per_nstand):

        d = np.random.rand(ntime_per_gulp, nchan, nstand*npol).astype('float16')
        d = ndarray(d, dtype='cf16')
        # print(d)

        # create arrays in GPU space
        input_data = ndarray(d,
                                dtype='cf16',
                                space='cuda')
        output_data = ndarray(shape=(nchan, nstand*(nstand+1)//2*npol*npol),
                                dtype='cf32',
                                space='cuda')

        print("Data shape: " + str(input_data.shape))
        print("Output shape: " + str(output_data.shape))

        dump = True # unsure what this does...

        tcc_start = time.perf_counter()
        tcc.execute(input_data, output_data, dump)
        tcc_end = time.perf_counter()
        tcc_time_ms = (tcc_end - tcc_start) * 1000
        total_time_ms += tcc_time_ms
        # print("\t", nstand, iter + 1, "=", tcc_time*1000, "ms")

        # copy output data from GPU to CPU
        output_data = output_data.copy(space='system')

    # get average time for particular nstand
    time_nstand_ms = total_time_ms/iter_per_nstand
    # print(times)
    # times[(nstand-min_nstand)//step_nstand] = time_nstand
    # print(nstand, "=", time_nstand*1000, "ms\n")
    f.write(str(nstand) + "," + str(time_nstand_ms) + "\n")
    print(str(nstand) + ": " + str(time_nstand_ms) + "ms")

    # delete the TCC object
    # del tcc

f.close()
prog_end = time.perf_counter()
print("Total time =", prog_end-prog_start, "s")


    

