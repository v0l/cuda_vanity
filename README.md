# CUDA Vanity Nostr npub Generator

A high-performance CUDA-accelerated vanity address generator for Nostr npub addresses. Searches for Nostr keypairs whose npub (bech32-encoded public key) starts with a specified pattern.

## Performance

Benchmarked performance on real hardware (pattern: "cuda"):

| GPU | Architecture | Compute Capability | Avg Hash Rate | vs rana CPU |
|-----|--------------|-------------------|---------------|-------------|
| **NVIDIA GB10** | Blackwell | 12.1 | **4.06M keys/s** | **2.45x faster** ✅ |
| GTX 1070 | Pascal | 6.1 | **532K keys/s** | 3.1x slower |

**CPU Comparison:**
- **rana** on Intel i9-14900K (24 cores, 32 threads): **~1.66M keys/s**

## Features

- ✅ **Correct secp256k1 implementation** - Generates valid Nostr keypairs
- ✅ **Optimized Jacobian coordinates** - No expensive modular inversions during computation
- ✅ **Windowed scalar multiplication** - 4-bit windows with precomputed point tables
- ✅ **Architecture-specific tuning** - Optimized kernel configuration per GPU generation
- ✅ **Early pattern rejection** - Fast rejection before full bech32 encoding
- ✅ **Fast pattern matching** - Validates bech32 character set (rejects b, i, o, 1)
- ✅ **Real-time progress** - Shows hash rate and keys generated
- ✅ **Efficient batch processing** - Quick termination after finding matches
- ✅ **Verified compatibility** - Keys tested with `nak` CLI tool

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with 12.9)
- Make

### Build on Linux/macOS

```bash
# Default build (compute capability 6.1)
make

# Build for specific GPU
make COMPUTE_CAP=121 CUDA_HOME=/usr/local/cuda-13.0

# Auto-detect GPU settings
make detect

# See all build options
make help
```

### Build on Windows

**Option 1: Visual Studio (Recommended)**

```cmd
# Using default compute capability 6.1 (GTX 1070)
nvcc -O3 -std=c++11 -gencode=arch=compute_61,code="sm_61,compute_61" -o vanity_npub.exe vanity_npub.cu -lcurand

# For RTX 50-series (compute capability 12.1)
nvcc -O3 -std=c++11 -gencode=arch=compute_121,code="sm_121,compute_121" -o vanity_npub.exe vanity_npub.cu -lcurand

# For RTX 40-series (compute capability 8.9)
nvcc -O3 -std=c++11 -gencode=arch=compute_89,code="sm_89,compute_89" -o vanity_npub.exe vanity_npub.cu -lcurand

# For RTX 30-series (compute capability 8.6)
nvcc -O3 -std=c++11 -gencode=arch=compute_86,code="sm_86,compute_86" -o vanity_npub.exe vanity_npub.cu -lcurand
```

**Option 2: WSL (Windows Subsystem for Linux)**

Install WSL2 with CUDA support and follow the Linux build instructions above.

**Finding your GPU's compute capability:**

```cmd
# Run in Command Prompt or PowerShell
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Usage

### Linux/macOS

```bash
# Generate vanity npub
./vanity_npub <pattern>

# Example: Find npub starting with "npub1test"
./vanity_npub test

# Example: Find npub starting with "npub1cuda"
./vanity_npub cuda
```

### Windows

```cmd
# Generate vanity npub
vanity_npub.exe <pattern>

# Example: Find npub starting with "npub1test"
vanity_npub.exe test

# Example: Find npub starting with "npub1cuda"
vanity_npub.exe cuda
```

### Example Output

```
Searching for npub starting with: npub1cuda
Initializing precomputed point table...
Precomputed table initialization complete.
[3s] Generated: 10245632 keys | Speed: 3415210.67 keys/s
Total: 10245632 keys in 3.00s (avg 3415210.67 keys/s)

Found matching npub!

Private key (hex): 16d8087ec4f5fc3d1e6e379f19058777f8578c7db8f1b9212a38a10f17b748e7
Public key (hex):  5e60b2b4dbe489b199dff9cb03e992847ada4e6b4b52478dfbd129f3c27b3c68
```

### Encode Keys

```bash
# Encode with nak CLI
echo "<privkey_hex>" | nak encode nsec
echo "<pubkey_hex>" | nak encode npub
```

## Pattern Constraints

- **Valid characters**: a-z, 0-9 (excluding b, i, o, 1)
- **Case**: Lowercase only (bech32 is case-insensitive but npub uses lowercase)
- **Position**: Currently only supports prefix matching (after "npub1")

### Difficulty Estimates

Each additional character increases difficulty by ~32x:

| Pattern Length | Estimated Keys | Time (GB10 @ 4.06M keys/s) | Time (GTX 1070 @ 532K keys/s) |
|----------------|----------------|----------------------------|-------------------------------|
| 4 chars | ~1M | 0.25s | 1.9s |
| 5 chars | ~33M | 8s | 62s |
| 6 chars | ~1B | 4 min | 31 min |
| 7 chars | ~34B | 2.3 hours | 18 hours |
| 8 chars | ~1T | 2.9 days | 22 days |

## Testing

```bash
# Run automated test with nak verification
make test
```

This will:
1. Build the project
2. Generate a vanity npub with pattern "q"
3. Verify the keypair using `nak`
4. Confirm the npub matches the pattern

## Benchmarking

Run performance benchmarks:

```bash
./benchmark.sh <pattern>

# Example
./benchmark.sh q
```

This will run 5 iterations and show min/max/avg hash rates.

## Technical Details

### Algorithm

1. **Key Generation**: Uses CUDA `curand` to generate random 256-bit private keys
2. **Point Multiplication**: Windowed scalar multiplication (4-bit windows) on secp256k1 using Jacobian coordinates
3. **Precomputed Tables**: Uses constant memory table of [1G, 2G, ..., 15G] for fast lookups
4. **Optimization**: Only one modular inversion per key (at final affine conversion)
5. **Early Rejection**: Quick pattern check before full bech32 encoding
6. **Encoding**: Converts public key to bech32 npub format
7. **Matching**: Compares against target pattern, rejects invalid bech32 characters

### Key Files

- `vanity_npub.cu` - Main CUDA kernel and host code
- `secp256k1_jacobian.cuh` - Optimized secp256k1 curve operations
- `ptx.cuh` - PTX assembly utilities for low-level operations
- `Makefile` - Build configuration

### Kernel Configuration

The kernel automatically detects your GPU architecture and optimizes configuration:

| Architecture | Compute Cap | Blocks | Threads | Keys/Thread | Total Keys/Batch |
|--------------|-------------|--------|---------|-------------|------------------|
| **Blackwell** | sm_120+ | 2048 | 256 | 512 | 268M |
| **Ada** | sm_89 | 1536 | 256 | 768 | 302M |
| **Ampere** | sm_86 | 1280 | 256 | 896 | 293M |
| **Turing** | sm_75 | 1024 | 256 | 1024 | 268M |
| **Pascal** | sm_61 | 1024 | 256 | 1024 | 268M |

**Register usage:** 111 registers/thread (GTX 1070), 128 registers/thread (GB10)

## Known Limitations

1. **Prefix only**: No suffix or middle pattern matching yet

## Future Improvements

- [ ] Multi-GPU support
- [ ] Batch mode (generate multiple vanity addresses)
- [ ] Suffix and middle pattern matching
- [ ] Case-insensitive matching
- [ ] Difficulty estimation and ETA display
- [ ] Further register optimization for even better occupancy

## License

MIT

## Acknowledgments

- **secp256k1**: Bitcoin's elliptic curve
- **Nostr**: Decentralized social protocol
- **rana**: CPU-based vanity address generator for performance comparison
- **nak**: Nostr Army Knife CLI tool for verification

## Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- Multi-GPU support
- Additional pattern matching modes
- Better occupancy through register optimization
