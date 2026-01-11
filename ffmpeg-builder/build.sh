#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

docker build -t ffmpeg-builder -o out .
