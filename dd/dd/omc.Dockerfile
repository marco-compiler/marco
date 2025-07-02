ARG BASE_IMAGE=marco-local-2:latest
FROM $BASE_IMAGE

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"


RUN apt update -y && \
    apt install -y autoconf automake libboost-all-dev expat default-jre uuid-dev

COPY ./version_openmodelica.txt /tmp/
COPY ./install_openmodelica.sh /tmp/

RUN chmod +x /tmp/install_openmodelica.sh && \
    cd /root && \
    OPENMODELICA_COMMIT=$(cat /tmp/version_openmodelica.txt) \
    /tmp/install_openmodelica.sh