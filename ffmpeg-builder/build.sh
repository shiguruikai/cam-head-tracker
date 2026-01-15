#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"

TARGET_TAG=$(cat ./version)

docker build --build-arg FFMPEG_TAG="$TARGET_TAG" -t ffmpeg-builder -o out .
