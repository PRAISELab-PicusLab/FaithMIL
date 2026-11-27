#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="trace_owl"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  --file "$SCRIPT_DIR/Dockerfile" \
  "$SCRIPT_DIR/.." 2>&1
