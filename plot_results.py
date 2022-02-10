# Author: Liam Ryan
# Purpose: Reads benchmark information from a .csv file in the form
#   nant, cuda time (ms), Python time (ms), data rate (Gb/s), performance (TOPS)
#   specified through a command line argument. Obtains description of parameters
#   through the descriptive .csv file name (eg: N8-4-576_T2048_F240.csv)  
#   Creates multiple visually appealing, relevant plots using matplotlib and saves 
#   them all as .png with a description of parameters prepended.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Specify .csv file as command line argument")
    exit()

# remove '.csv'
description = sys.argv[1][:-4]

# remove .\ before command line arguments (VSCode/Windows)
if description[:2] == '.\\':
    description = description[2:]

# Format: 
# (nant, CUDA time (ms), Python time (ms), overhead (ms), max ingest (Gb/s), performance (TOPS))
py_data = np.genfromtxt(sys.argv[1], delimiter=',')

# Used for formatting of x-axis in steps of 64
a = 0 # first x-axis tick
b = py_data[-1,0] // 64 * 64 + 1

# In these benchmarks there is a file open, file write and file close in the CUDA code
# called from Python. This adds unrealistic time to the Python time and we therefore
# calculate this time by performing the benchmark with/without file I/O and obtain
# a constant overhead of 0.5-0.8 ms associated with this file I/O. Set this free
# parameter for more accurate results 
file_write_time = 0

ants = py_data[:,0]
cuda_times = py_data[:,1]
py_times = py_data[:,2] - file_write_time
overheads = np.subtract(py_times, cuda_times)
ingests = py_data[:,3]
py_performances = py_data[:,4]
cu_performances = np.divide(np.multiply(py_performances, py_times), cuda_times)

# Order 2 polynomial fitting of CUDA times
coeffs = np.polynomial.polynomial.polyfit(ants, cuda_times, 2)
pfit = np.polynomial.polynomial.polyval(ants, coeffs)

fig1, ax1 = plt.subplots()
ax1.plot(ants, cuda_times, label='CUDA', color='lime')
ax1.plot(ants, py_times, label='Python/CUDA', color='C0')
ax1.plot(ants, pfit, '--', label='Quadratic fit', color='black', linewidth='1')
ax1.legend()
ax1.set_xlabel('Number of antennas')
ax1.set_ylabel('Time (ms)')

plt.xticks(np.arange(a, b, 64))
plt.minorticks_on()
plt.savefig(f'{description}_times.png')

fig2, ax2 = plt.subplots()
ax2.plot(ants, overheads)
ax2.set_xlabel('Number of antennas')
ax2.set_ylabel('Python overhead (ms)')
plt.xticks(np.arange(a, b, 64))
plt.minorticks_on()
plt.savefig(f'{description}_overhead.png')

fig3, ax3 = plt.subplots()
ax3.plot(ants, overheads/cuda_times*100)
ax3.set_xlabel('Number of antennas')
ax3.set_ylabel('Relative overhead (%)')
plt.xticks(np.arange(a, b, 64))
plt.minorticks_on()
plt.savefig(f'{description}_rel_overhead.png')

fig4, ax4 = plt.subplots()
ax4.plot(ants, ingests)
ax4.plot(ants, np.full(ants.shape, 256), '--', color='red', label='PCIe3 Interconnect Bandwidth')
ax4.legend()
ax4.set_xlabel('Number of antennas')
ax4.set_ylabel('Maximum ingest (Gb/s)')
plt.xticks(np.arange(a, b, 64))
plt.minorticks_on()
plt.savefig(f'{description}_ingest.png')

fig5, ax5 = plt.subplots()
ax5.plot(ants, py_performances, label='Python/CUDA', color='C0')
ax5.plot(ants, cu_performances, label='CUDA', color='lime')
ax5.plot(ants, np.full(ants.shape, 14), '--', color='red', label='Single-Precision Performance')
# ax5.plot(ants, np.full(ants.shape, 112), '--', color='red', label='Tensor Performance')
ax5.legend()
ax5.set_xlabel('Number of antennas')
ax5.set_ylabel('Performance (TOPS)')
plt.xticks(np.arange(a, b, 64))
plt.minorticks_on()
plt.savefig(f'{description}_performance.png')

fig, ((sax1, sax2, sax3), (sax4, sax5, sax6)) = plt.subplots(2,3)
plt.subplots_adjust(hspace=0.5, left=0.1, right=0.95)
sax1.plot(ants, cuda_times, label='CUDA', color='lime')
sax1.plot(ants, py_times, label='Python', color='C0')
sax2.plot(ants, overheads)
sax3.plot(ants, overheads/cuda_times*256)
sax4.plot(ants, ingests)
sax4.plot(ants, np.full(ants.shape, 256), '--', color='red', label='Maximum data rate of network card')
sax5.plot(ants, cu_performances, color='lime')
sax5.plot(ants, py_performances, color='C0')
sax5.plot(ants, np.full(ants.shape, 14), '--', color='red', label='Single-Precision Performance') 
plt.savefig(f'{description}_master.png')
