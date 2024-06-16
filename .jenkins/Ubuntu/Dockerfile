ARG VERSION=latest
FROM ubuntu:$VERSION

RUN apt update && \
    apt -y install build-essential gfortran ninja-build lld cmake ccache git python3-pip python3-venv libxml2-dev libtinfo-dev libgmp3-dev libmpc-dev wget
