FROM registry.gitlab.com/oceanum/docker/core-ubuntu:v0.1.3
LABEL maintainer "Oceanum Developers <developers@oceanum.science>"

ENV REPOS="/source"

RUN echo "--------------- System packages ---------------" && \
    apt update && apt upgrade -y

RUN echo "--------------- Conflicting libs ---------------" && \
    pip install --no-cache-dir -U packaging

RUN echo "--------------- Installing onstats ---------------"
COPY setup.py README.rst HISTORY.rst $REPOS/onstats/
COPY onstats $REPOS/onstats/onstats
COPY tests $REPOS/onstats/tests
RUN cd $REPOS/onstats && \
    pip install -e . --no-cache-dir
