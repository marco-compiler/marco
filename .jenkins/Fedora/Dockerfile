ARG VERSION=latest
FROM fedora:$VERSION

RUN dnf update -y && \
    dnf install -y gcc gcc-c++ gfortran perl ninja-build cmake ccache git python3-pip python3-virtualenv libxml2-devel libmpc-devel wget
