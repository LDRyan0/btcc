## btcc

A [Bifrost](https://github.com/telegraphic/bifrost) wrapper for the [Tensor-Core Correlator](https://git.astron.nl/RD/tensor-core-correlator).

### Compiling BTCC

To build BTCC:

1. Install appropriate dependancies
   1. Bifrost (https://github.com/telegraphic/bifrost)
   2. Tensor-Core Correlator (https://git.astron.nl/RD/tensor-core-correlator)
   3. Meson build system (https://github.com/mesonbuild/meson)
2. Setup your build environment (on topaz, run `source setup_env.sh`).
3. Ensure correct paths to dependancies are setup in `meson.build`
4. Compile with meson by running:

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

Example Performance (Tesla V100)|  Example correlation matrix | Example image 
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/86461236/153545168-1b854b20-b004-4778-a541-ab9fe5a31dac.png" height="200"/> | <img src="https://user-images.githubusercontent.com/86461236/153545216-5a405dac-7889-48b2-a47d-4470080591ff.png" height="200"/> | <img src="https://user-images.githubusercontent.com/86461236/153545271-e660153b-ef2b-4dee-80ce-f99940ae9df3.JPG" height="200"/>
