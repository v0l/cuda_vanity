#!/bin/bash
# Build script for creating .deb package

set -e

VERSION=${1:-"0.0.0"}
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
sed "s/VERSION_PLACEHOLDER/${VERSION}/" contrib/debian/DEBIAN/control > "${BUILD_DIR}/DEBIAN/control"

# Build .deb package
dpkg-deb --build "${BUILD_DIR}"
mv "${BUILD_DIR}.deb" "${PACKAGE_NAME}_${VERSION}_amd64.deb"

echo "Package created: ${PACKAGE_NAME}_${VERSION}_amd64.deb"
