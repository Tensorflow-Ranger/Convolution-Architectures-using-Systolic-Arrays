## Convolutions in SystemC using Systolic Arrays


### Project Structure

- `src/pe.h` — Processing Element (PE) module: a single MAC unit in the systolic array
- `src/main.cpp` — Testbench that instantiates a PE, drives signals, and generates a VCD waveform

### Prerequisites

- **GCC** (g++ with C++17 support)
- **SystemC 3.0.1** (installed to `/usr/local/systemc`)

### Installing SystemC

```bash
# Download and extract
cd /tmp
curl -L -o systemc-3.0.1.tar.gz https://github.com/accellera-official/systemc/archive/refs/tags/3.0.1.tar.gz
tar xzf systemc-3.0.1.tar.gz

# Build and install
cd systemc-3.0.1
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/systemc -DCMAKE_CXX_STANDARD=17
make -j$(nproc)
sudo make install
```

Set the environment variable (add to your `~/.bashrc` or `~/.zshrc`):

```bash
export SYSTEMC_HOME=/usr/local/systemc
```

### Building & Running

#### Using g++ directly

```bash
cd src
g++ main.cpp \
    -I$SYSTEMC_HOME/include \
    -L$SYSTEMC_HOME/lib64 \
    -lsystemc \
    -Wl,-rpath,$SYSTEMC_HOME/lib64 \
    -o sim
./sim
```

#### Using CMake

```bash
mkdir build && cd build
cmake .. -DSYSTEMC_ROOT=/usr/local/systemc
make
./sim
```

### Viewing Waveforms

After running the simulation, a `waveforms.vcd` file is generated. Open it with GTKWave:

```bash
gtkwave waveforms.vcd
```