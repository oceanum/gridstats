FROM registry.gitlab.com/oceanum/docker/core-ubuntu:v0.1.3
LABEL maintainer "Oceanum Developers <developers@oceanum.science>"

ENV REPOS="/source"

RUN echo "--------------- System packages ---------------" && \
    apt update && apt upgrade -y

RUN echo "--------------- Installing onstats ---------------"
COPY setup.py requirements.txt README.rst HISTORY.rst $REPOS/onstats/
COPY onstats $REPOS/onstats/onstats
COPY tests $REPOS/onstats/tests
RUN cd $REPOS/onstats && \
    pip install -U -r requirements.txt --no-cache-dir && \
    pip install -e . --no-cache-dir
