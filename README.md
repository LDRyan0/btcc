## btcc

A [Bifrost](https://github.com/telegraphic/bifrost) wrapper for the [Tensor-Core Correlator](https://git.astron.nl/RD/tensor-core-correlator).

### Compiling BTCC

To build BTCC:

1) Install appropriate dependancies
  * Bifrost (https://github.com/telegraphic/bifrost)
  * Tensor-Core Correlator (https://git.astron.nl/RD/tensor-core-correlator)
  * Meson build system (https://github.com/mesonbuild/meson)
2) Setup your build environment (on topaz, run `source setup_env.sh`).
3) Ensure correct paths to dependancies are setup in `meson.build'
4) Compile with meson by running:

```
meson setup build
cd build
meson compile
```

There are several conditional compliation constants included in btcc.cu:
* `DCP_DEBUG`: enables full debugging output
* `NO_CHECKS`: removes all sanity checks
* `TIME_CORR`: write all correlation times (ms) to `cuda_results.csv`


### Using BTCC

```
python
from btcc import Btcc

tcc = Btcc()
tcc.init(nbits, ntime, nchan, nant, npol)
tcc.execute(tcc_input, tcc_output, True)
```

See `validate.py`, `benchmark.py` and `eda.py` for example usages of BTCC.
