FROM fedora:40

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"

ARG LLVM_PARALLEL_COMPILE_JOBS=4
ARG LLVM_PARALLEL_LINK_JOBS=1

RUN dnf update -y && \
    dnf install -y gcc gcc-c++ gfortran perl ninja-build mold cmake ccache \
    git python3-pip python3-virtualenv libxml2-devel wget doxygen \
    openblas-devel-0.3.26-4.fc40 \
    suitesparse-devel-7.6.0-1.fc40 \
    sundials-devel-6.6.2-7.fc40

COPY ./setup_venv.sh /tmp/
RUN chmod +x /tmp/setup_venv.sh && /tmp/setup_venv.sh

COPY ./version_llvm.txt /tmp/
COPY ./install_llvm.sh /tmp/

RUN chmod +x /tmp/install_llvm.sh && \
    cd /root && \
    LLVM_COMMIT=$(cat /tmp/version_llvm.txt) \
    LLVM_BUILD_TYPE=Release \
    LLVM_ENABLE_ASSERTIONS=ON \
    /tmp/install_llvm.sh

COPY ./version_marco_runtime.txt /tmp/
COPY ./install_marco_runtime.sh /tmp/

RUN chmod +x /tmp/install_marco_runtime.sh && \
    cd /root && \
    MARCO_RUNTIME_COMMIT=$(cat /tmp/version_marco_runtime.txt) \
    MARCO_RUNTIME_BUILD_TYPE=Debug \
    /tmp/install_marco_runtime.sh

RUN pip install nltk
