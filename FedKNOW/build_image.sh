#!/bin/bash

if [ -z "${CI}" ]; then
    BUILDKIT=1
else
    BUILDKIT=0
fi

DOCKER_BUILDKIT=${BUILDKIT} docker build -t gcr.io/sdml-349720/flower-client-zstd:latest . -f FedKNOW/docker/Dockerfile