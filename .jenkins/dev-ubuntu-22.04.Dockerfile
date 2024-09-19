FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"

ARG LLVM_PARALLEL_COMPILE_JOBS=4
ARG LLVM_PARALLEL_LINK_JOBS=1

RUN apt update -y && \
    apt install -y build-essential gfortran ninja-build lld mold cmake ccache \
    git python3-pip python3-venv libxml2-dev libtinfo-dev wget doxygen \
    libopenblas-dev=0.3.20+ds-1 \
    libsuitesparse-dev=1:5.10.1+dfsg-4build1 \
    libsundials-dev=5.8.0+dfsg-1build1

COPY ./setup_venv.sh /tmp/
RUN chmod +x /tmp/setup_venv.sh && /tmp/setup_venv.sh

COPY ./install_llvm.sh /tmp/

RUN chmod +x /tmp/install_llvm.sh && \
    cd /root && \
    LLVM_BUILD_TYPE=Release \
    LLVM_ENABLE_ASSERTIONS=ON \
    /tmp/install_llvm.sh

COPY ./install_runtime.sh /tmp/

RUN chmod +x /tmp/install_runtime.sh && \
    cd /root && \
    MARCO_RUNTIME_BUILD_TYPE=Debug \
    /tmp/install_runtime.sh

RUN pip install nltk
