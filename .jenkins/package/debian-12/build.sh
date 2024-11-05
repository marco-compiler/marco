#!/bin/sh

src_path=$1
install_path=$2

# Create folders.
mkdir -p marco/DEBIAN
mkdir -p marco/usr/bin
mkdir -p marco/usr/lib/marco

# Copy the control file.
cp "${src_path}/.jenkins/package/debian-12/control" marco/DEBIAN/control

# Copy the driver.
cp "${install_path}/bin/marco" marco/usr/lib/marco/marco

# Copy the driver wrapper.
cp "${src_path}/.jenkins/package/debian-12/marco-wrapper.sh" marco/usr/bin/marco
chmod +x marco/usr/bin/marco

# Build the package.
dpkg-deb --build marco

# Clean the work directory.
rm -rf marco

# Extract version and architecture from the control file.
version=$(grep "^Version:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)
architecture=$(grep "^Architecture:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)

# Rename the package.
mv marco.deb marco-${version}_${architecture}.deb
