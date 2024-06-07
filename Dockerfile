FROM ubuntu:24.04

LABEL maintainer "Oceanum Developers <developers@oceanum.science>"

# Set variables
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC
ARG REPOS="/source"

# Install system packages
RUN apt update && apt -y upgrade && \
    apt -y install \
        curl \
        git \
        python3-dev \
        python3-pip \
        wget

# Set default python3 as python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install onstats
COPY setup.py README.rst HISTORY.rst $REPOS/onstats/
COPY onstats $REPOS/onstats/onstats
COPY tests $REPOS/onstats/tests
RUN pip install --break-system-packages -e $REPOS/onstats --no-cache-dir
