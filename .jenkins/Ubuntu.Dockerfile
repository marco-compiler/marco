ARG VERSION=latest
FROM ubuntu:$VERSION

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"

ARG LLVM_BUILD_TYPE=Release
ARG LLVM_ENABLE_ASSERTIONS=OFF
ARG LLVM_PARALLEL_COMPILE_JOBS=4
ARG LLVM_PARALLEL_LINK_JOBS=1

ARG MARCO_RUNTIME_BUILD_TYPE=Release

RUN apt update -y && \
    apt install -y build-essential gfortran ninja-build lld cmake ccache git \
    python3-pip python3-venv libxml2-dev libtinfo-dev wget libopenblas-dev \
    libsuitesparse-dev libsundials-dev doxygen

COPY ./setup_venv.sh /tmp/
RUN chmod +x /tmp/setup_venv.sh && /tmp/setup_venv.sh

COPY ./install_llvm.sh /tmp/

RUN chmod +x /tmp/install_llvm.sh && \
    cd /root && \
    /tmp/install_llvm.sh

COPY ./install_runtime.sh /tmp/

RUN chmod +x /tmp/install_runtime.sh && \
    cd /root && \
    /tmp/install_runtime.sh

RUN pip install nltk