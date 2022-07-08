#!/bin/bash

if [[ "$#" -ne 1 ]]; then
  echo "Usage: ./sundials.sh install-path"
  exit 1
fi

ORIGINAL_DIR=$(pwd)
TEMP_DIR=""

fail() {
    echo $1

    if [[ $var ]]; then
      rm -r "$TEMP_DIR"
    fi

    exit 1
}

command -v gfortran > /dev/null          || fail "Missing fortran compiler"
ldconfig -p | grep 'libgmp' > /dev/null  || fail "Missing libgmp"
ldconfig -p | grep 'libmpfr' > /dev/null || fail "Missing libmpfr"

INSTDIR=$1
mkdir -p $INSTDIR || fail "Error creating directory $INSTDIR"

TEMP_DIR=$(mktemp -d)
echo $TEMP_DIR
cd "$TEMP_DIR" || fail "Error crating the temporary directory"

OPENBLAS=OpenBLAS-0.3.20
SUITESPARSE=SuiteSparse-5.12.0
SUNDIALS=sundials-6.2.0

OPENBLAS_URL=https://github.com/xianyi/OpenBLAS/releases/download/v0.3.20/OpenBLAS-0.3.20.tar.gz
SUITESPARSE_URL=https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.12.0.tar.gz
SUNDIALS_URL=https://github.com/LLNL/sundials/releases/download/v6.2.0/sundials-6.2.0.tar.gz

[ -f $OPENBLAS.tar.gz ]    || wget $OPENBLAS_URL    || fail "Error downloading $OPENBLAS.tar.gz"
[ -f $SUITESPARSE.tar.gz ] || wget $SUITESPARSE_URL || fail "Error downloading $SUITESPARSE.tar.gz"
[ -f $SUITESPARSE.tar.gz ] || mv v5.12.0.tar.gz $SUITESPARSE.tar.gz # HACK to fix file name
[ -f $SUNDIALS.tar.gz ]    || wget $SUNDIALS_URL    || fail "Error downloading $SUNDIALS.tar.gz"

tar xzvf $OPENBLAS.tar.gz    || fail "Error extracting $OPENBLAS.tar.gz"
tar xzvf $SUITESPARSE.tar.gz || fail "Error extracting $SUITESPARSE.tar.gz"
tar xzvf $SUNDIALS.tar.gz    || fail "Error extracting $SUNDIALS.tar.gz"

cd $OPENBLAS || fail "Error extracting $OPENBLAS.tar.gz"
## TODO: make builds also the .so but compiles with O2,
## while cmake builds with O3 but only makes the .a, why?
make -j`nproc` || fail "Openblas build failed"
make install PREFIX="$INSTDIR" || fail "Openblas install failed"
# mkdir build
# cd build
# # By default it builds in debug mode!
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTDIR"
# make -j`nproc` || fail "Openblas build failed"
# make install || fail "Openblas install failed"
# cd ..
cd ..

cd $SUITESPARSE || fail "Error extracting $SUITESPARSE.tar.gz"
# NOTE: the cmake projects despite having CMAKE_BUILD_TYPE not set are built
# with O3 anyway, so it's ok
# TODO: Readme.md says using openblas resuts in severe performance degradations,
# that they investigating the issue and to use the Intel MKL blas, which however
# may not work well with ryzen.
JOBS=`nproc` make library BLAS="$INSTDIR/lib/libopenblas.so" LAPACK="$INSTDIR/lib/libopenblas.so" || fail "Suitesparse build failed"
make install INSTALL=$INSTDIR || fail "Suitesparse install failed"
cd ..

cd $SUNDIALS|| fail "Error extracting $SUNDIALS.tar.gz"
# TODO: sundials has so many options and it is unclear if enabling them would make it more capable
# or faster or how faster. For now enabling only KLU as it's required
# ENABLE_CUDA                      OFF
# ENABLE_HIP                       OFF
# ENABLE_HYPRE                     OFF
# ENABLE_KLU                       OFF
# ENABLE_LAPACK                    OFF
# ENABLE_MAGMA                     OFF
# ENABLE_MPI                       OFF
# ENABLE_OPENMP                    OFF
# ENABLE_OPENMP_DEVICE             OFF
# ENABLE_PETSC                     OFF
# ENABLE_PTHREAD                   OFF
# ENABLE_RAJA                      OFF
# ENABLE_SUPERLUDIST               OFF
# ENABLE_SUPERLUMT                 OFF
# ENABLE_SYCL                      OFF
# ENABLE_TRILINOS                  OFF
# ENABLE_XBRAID                    OFF
mkdir build
cd build || fail "Error creating the build folder for Sundials"
cmake .. -DENABLE_KLU=ON -DKLU_INCLUDE_DIR="$INSTDIR/include" -DKLU_LIBRARY_DIR="$INSTDIR/lib" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$INSTDIR" -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON
make -j `nproc` || fail "Sundials build failed"
make install || fail "Sundials install failed"

cd "$ORIGINAL_DIR" || fail "Can't go back to the original directory"

# Remove the temporary directory
rm -r "$TEMP_DIR"

