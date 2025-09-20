FROM debian:12

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"

# Add user.
ARG USERNAME=marco
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt update -y && \
    apt install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Install compilation dependencies.
RUN apt update -y && \
    apt install -y build-essential gfortran ninja-build lld mold cmake ccache \
    git python3-pip python3-venv libxml2-dev libtinfo-dev wget doxygen libasan6

# Create a Python virtual environment.
COPY ./setup_venv.sh /tmp/
RUN chmod +x /tmp/setup_venv.sh && /tmp/setup_venv.sh

# Install LLVM.
ARG LLVM_PARALLEL_COMPILE_JOBS=4
ARG LLVM_PARALLEL_LINK_JOBS=1
ARG LLVM_BUILD_TYPE=Release
ARG LLVM_ENABLE_ASSERTIONS=OFF
ARG LLVM_SANITIZER=""

COPY ./version_llvm.txt /tmp/
COPY ./install_llvm.sh /tmp/

RUN chmod +x /tmp/install_llvm.sh && \
    cd /root && \
    LLVM_COMMIT=$(cat /tmp/version_llvm.txt) \
    /tmp/install_llvm.sh

# Install MARCO runtime libraries.
ARG MARCO_RUNTIME_BUILD_TYPE=Release

RUN apt update -y && \
    apt install -y libopenblas-dev=0.3.21+ds-4 \
    libsuitesparse-dev=1:5.12.0+dfsg-2 \
    libsundials-dev=6.4.1+dfsg1-3

COPY ./version_marco_runtime.txt /tmp/
COPY ./install_marco_runtime.sh /tmp/

RUN chmod +x /tmp/install_marco_runtime.sh && \
    cd /root && \
    MARCO_RUNTIME_COMMIT=$(cat /tmp/version_marco_runtime.txt) \
    /tmp/install_marco_runtime.sh

# Install additional MARCO dependencies.
RUN apt update -y && \
    apt install -y python3-nltk

# Install OpenModelica.
RUN apt update -y && \
    apt install -y autoconf automake libboost-all-dev expat default-jre uuid-dev

COPY ./version_openmodelica.txt /tmp/
COPY ./install_openmodelica.sh /tmp/

RUN chmod +x /tmp/install_openmodelica.sh && \
    cd /root && \
    OPENMODELICA_COMMIT=$(cat /tmp/version_openmodelica.txt) \
    /tmp/install_openmodelica.sh

# Install packaging tools.
RUN apt update -y && \
    apt install -y gettext-base

# Reduce image size.
RUN apt clean && \
    rm -rf /var/lib/apt/lists/*
