FROM registry.gitlab.com/oceanum/docker/core-ubuntu:v0.1.3
LABEL maintainer "Oceanum Developers <developers@oceanum.science>"

ENV REPOS="/source"

RUN echo "--------------- System packages ---------------" && \
    apt update && apt upgrade -y

RUN echo "--------------- Installing stats ---------------"
COPY setup.py requirements.txt README.rst HISTORY.rst $REPOS/stats/
COPY stats $REPOS/stats/stats
COPY tests $REPOS/stats/tests
RUN cd $REPOS/stats && \
    pip install -U -r requirements.txt --no-cache-dir && \
    pip install -e . --no-cache-dir
