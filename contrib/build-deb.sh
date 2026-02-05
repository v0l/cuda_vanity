#!/bin/bash
# Build script for creating .deb package

set -e

# Determine version: use argument, or extract from git tag, or use commit hash
if [ -n "$1" ]; then
    VERSION="$1"
elif [ -n "${GITHUB_REF}" ]; then
    # Extract version from tag if it's a tag ref
    if [[ "${GITHUB_REF}" =~ ^refs/tags/v(.+)$ ]]; then
        VERSION="${BASH_REMATCH[1]}"
    else
        # Not a tag, use short commit hash with 0.0.0 prefix for debian compatibility
        VERSION="0.0.0-$(git rev-parse --short HEAD)"
    fi
else
    # Fallback to git tag or commit hash
    if git describe --tags --exact-match >/dev/null 2>&1; then
        VERSION=$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//')
    else
        VERSION="0.0.0-$(git rev-parse --short HEAD)"
    fi
fi
PACKAGE_NAME="cuda-vanity-npub"
BUILD_DIR="build/debian"

echo "Building ${PACKAGE_NAME} version ${VERSION}"

# Clean and create build directory
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/DEBIAN"
mkdir -p "${BUILD_DIR}/usr/local/bin"
mkdir -p "${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}"

# Copy binary
if [ ! -f "vanity_npub" ]; then
    echo "Error: vanity_npub binary not found. Run 'make' first."
    exit 1
fi
cp vanity_npub "${BUILD_DIR}/usr/local/bin/"
chmod 755 "${BUILD_DIR}/usr/local/bin/vanity_npub"

# Copy documentation
cp README.md "${BUILD_DIR}/usr/share/doc/${PACKAGE_NAME}/"

# Create control file with version
sed "s|VERSION_PLACEHOLDER|${VERSION}|" contrib/debian/DEBIAN/control > "${BUILD_DIR}/DEBIAN/control"

# Build .deb package
dpkg-deb --build "${BUILD_DIR}"
mv "${BUILD_DIR}.deb" "${PACKAGE_NAME}_${VERSION}_amd64.deb"

echo "Package created: ${PACKAGE_NAME}_${VERSION}_amd64.deb"
