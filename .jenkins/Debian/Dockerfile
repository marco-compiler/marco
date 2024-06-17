ARG VERSION=latest
FROM debian:$VERSION

RUN apt update && \
    apt -y install build-essential gfortran ninja-build lld cmake ccache git python3-pip python3-venv libxml2-dev libtinfo-dev wget libopenblas-dev libsuitesparse-dev libsundials-dev
