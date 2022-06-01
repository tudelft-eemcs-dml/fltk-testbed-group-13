#!/bin/bash

if [ -z "${CI}" ]; then
    BUILDKIT=1
else
    BUILDKIT=0
fi

DOCKER_BUILDKIT=${BUILDKIT} docker build -t flower_client:latest . -f docker/Dockerfile