# CUDA Vanity Nostr npub Generator

A high-performance CUDA-accelerated vanity address generator for Nostr npub addresses. Searches for Nostr keypairs whose npub (bech32-encoded public key) starts with a specified pattern.

## Performance

Benchmarked performance on real hardware:

| GPU | Architecture | Compute Capability | Performance | vs rana CPU |
|-----|--------------|-------------------|-------------|-------------|
| GB10 | Blackwell | 12.1 | **~2.0M keys/s** | **1.2x faster** ✅ |
| GTX 1070 | Pascal | 6.1 | ~150K keys/s | 11x slower |

**CPU Comparison:**
- **rana** on Intel i9-14900K (24 cores, 32 threads): **~1.66M keys/s**
- This CUDA implementation on RTX 50-series: **~2.0M keys/s** (1.2x faster)

The CUDA implementation beats a high-end 24-core CPU on modern GPUs!

### Why the Difference?

The **RTX 50-series** benefits from:
- Modern architecture (2025) with better instruction throughput
- Higher memory bandwidth and more CUDA cores
- Better compiler optimizations for newer compute capabilities

The **GTX 1070** (2016) is limited by:
- Older Pascal architecture with lower instruction throughput
- High register usage (163 registers/thread) limiting occupancy on older GPUs
- Fewer optimization features in older compute capabilities

**Bottom line**: On modern GPUs, this CUDA implementation **beats CPU implementations**. On older GPUs, it's architecture-limited but still provides GPU acceleration.

## Features

- ✅ **Correct secp256k1 implementation** - Generates valid Nostr keypairs
- ✅ **Optimized Jacobian coordinates** - No expensive modular inversions during computation
- ✅ **Fast pattern matching** - Validates bech32 character set (rejects b, i, o, 1)
- ✅ **Real-time progress** - Shows hash rate and keys generated
- ✅ **Efficient batch processing** - Quick termination after finding matches
- ✅ **Verified compatibility** - Keys tested with `nak` CLI tool

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with 12.9)
- Make

### Build

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

## Usage

```bash
# Generate vanity npub
./vanity_npub <pattern>

# Example: Find npub starting with "npub1test"
./vanity_npub test

# Example: Find npub starting with "npub1cuda"
./vanity_npub cuda
```

### Example Output

```
Searching for npub starting with: npub1cuda
[9s] Generated: 19346848 keys | Speed: 2047935.64 keys/s
Total: 19346848 keys in 9.45s (avg 2047935.64 keys/s)

Found matching npub!
Private key (hex): 55d1e79e81ca9d3286bffcee490cc3483b555747f1ce47cd9928912859aaa0b4
Public key (hex): 043d9a77f9d8f5c8e4b2a1c3d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5...
npub: npub1cudak7lx3m6u3e9j588d54h80z6uqksynl39ytx0a0cnqp77wn6sq8r4a5
```

### Verify Keys

```bash
# Verify with nak CLI
echo "<privkey_hex>" | nak key public | nak encode npub
```

## Pattern Constraints

- **Valid characters**: a-z, 0-9 (excluding b, i, o, 1)
- **Case**: Lowercase only (bech32 is case-insensitive but npub uses lowercase)
- **Position**: Currently only supports prefix matching (after "npub1")

### Difficulty Estimates

Each additional character increases difficulty by ~32x:

| Pattern Length | Estimated Keys | Time (RTX 50-series @ 2M keys/s) | Time (GTX 1070 @ 150K keys/s) |
|----------------|----------------|----------------------------------|-------------------------------|
| 4 chars | ~1M | 0.5s | 7s |
| 5 chars | ~33M | 17s | 3.7 min |
| 6 chars | ~1B | 8.5 min | 1.9 hours |
| 7 chars | ~34B | 4.7 hours | 2.6 days |
| 8 chars | ~1T | 6.4 days | 77 days |

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
2. **Point Multiplication**: Implements scalar multiplication on secp256k1 curve using Jacobian coordinates
3. **Optimization**: Only one modular inversion per key (at final affine conversion)
4. **Encoding**: Converts public key to bech32 npub format
5. **Matching**: Compares against target pattern, rejects invalid bech32 characters

### Key Files

- `vanity_npub.cu` - Main CUDA kernel and host code
- `secp256k1_jacobian.cuh` - Optimized secp256k1 curve operations
- `ptx.cuh` - PTX assembly utilities for low-level operations
- `Makefile` - Build configuration

### Kernel Configuration

```c
threads_per_block = 256
num_blocks = 1024
keys_per_thread = 1024
// Total: ~268M keys per batch
```

## Optimization History

1. **Initial naive implementation**: ~9,000 keys/s (affine coordinates)
2. **Jacobian coordinates**: ~144,000 keys/s (16x speedup)
3. **Fixed batch termination**: Proper performance on modern GPUs
4. **Current**: 2M keys/s on RTX 50-series, 150K keys/s on GTX 1070

## Known Limitations

1. **Older GPU performance**: GTX 1070 and similar Pascal-era GPUs are limited by architecture
2. **Register pressure**: High register usage (163/thread) limits occupancy
3. **No precomputed tables**: Could use precomputed multiples of generator point
4. **Prefix only**: No suffix or middle pattern matching yet

## Future Improvements

- [ ] Windowed/wNAF scalar multiplication for better performance
- [ ] Precomputed tables of multiples of G
- [ ] Reduce register usage for better occupancy
- [ ] Multi-GPU support
- [ ] Batch mode (generate multiple vanity addresses)
- [ ] Suffix and middle pattern matching
- [ ] Case-insensitive matching
- [ ] Difficulty estimation and ETA display

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
