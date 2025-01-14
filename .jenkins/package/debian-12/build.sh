#!/bin/sh

# The path where the project was cloned.
src_path=$1

# The path where the project was installed.
install_path=$2

# Extract version and architecture from the control file.
version=$(grep "^Version:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)
architecture=$(grep "^Architecture:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)

# Create folders.
package_name=marco-${version}_${architecture}

mkdir -p "${package_name}"/DEBIAN
mkdir -p "${package_name}"/usr/bin
mkdir -p "${package_name}"/usr/lib/marco

# Instantiate the control file.
TAG=$(git -C "${src_path}" describe --tags --abbrev=0)
VERSION="${TAG}" envsubst < "${src_path}/.jenkins/package/debian-12/control" > "${package_name}"/DEBIAN/control

# Copy the driver.
cp "${install_path}/bin/marco" "${package_name}"/usr/lib/marco/marco

# Copy the driver wrapper.
cp "${src_path}/.jenkins/package/debian-12/marco-wrapper.sh" "${package_name}"/usr/bin/marco
chmod +x "${package_name}"/usr/bin/marco

# Build the package.
dpkg-deb --build "${package_name}"

# Clean the work directory.
rm -rf "${package_name}"
