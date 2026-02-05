# Contrib

This directory contains packaging and build scripts for various platforms.

## Debian Package

### Building Manually

To build a `.deb` package locally:

```bash
# Build the binary first
make

# Create the .deb package
./contrib/build-deb.sh 1.0.0
```

This will create `cuda-vanity-npub_1.0.0_amd64.deb` in the current directory.

### Installing

```bash
sudo dpkg -i cuda-vanity-npub_1.0.0_amd64.deb
```

### Package Contents

- Binary: `/usr/local/bin/vanity_npub`
- Documentation: `/usr/share/doc/cuda-vanity-npub/README.md`

### Dependencies

- `libc6` - GNU C Library

Note: You must have NVIDIA GPU drivers and CUDA runtime installed separately.

## Automated Builds

The GitHub Actions workflow (`.github/workflows/release.yml`) automatically builds packages for:

- **Linux**: `.deb` package (Ubuntu/Debian)
- **Windows**: `.exe` executable

Releases are created automatically when you push a git tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```
