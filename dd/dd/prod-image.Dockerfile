ARG BASE_IMAGE=marco-local-63:latest
FROM $BASE_IMAGE

ARG MARCO_COMMIT=Bellani_Bruno-external_functions
ARG PYTHON3_EXECUTABLE=/virtualenv/bin/python

LABEL org.opencontainers.image.source="https://github.com/marco-compiler/marco"

# Install MARCO.
COPY ./install_marco.sh /tmp/

RUN chmod +x /tmp/install_marco.sh && \
    cd /root && \
    MARCO_BUILD_TYPE=Release \
    /tmp/install_marco.sh
