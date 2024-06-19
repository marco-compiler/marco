ARG VERSION=latest
FROM fedora:$VERSION

ARG LLVM_BUILD_TYPE=Release
ARG LLVM_ENABLE_ASSERTIONS=OFF
ARG LLVM_PARALLEL_COMPILE_JOBS=4
ARG LLVM_PARALLEL_LINK_JOBS=1

ARG MARCO_RUNTIME_BUILD_TYPE=Release

RUN dnf update -y && \
    dnf install -y gcc gcc-c++ gfortran perl ninja-build cmake ccache git \
    python3-pip python3-virtualenv libxml2-devel wget openblas-devel \
    suitesparse-devel sundials-devel

COPY ./install_llvm.sh /tmp/

RUN chmod +x /tmp/install_llvm.sh && \
    cd /root && \
    /tmp/install_llvm.sh

COPY ./install_runtime.sh /tmp/

RUN chmod +x /tmp/install_runtime.sh && \
    cd /root && \
    /tmp/install_runtime.sh
