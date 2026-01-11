# -----------------------------------------------------------------------
# FFmpeg Windows Cross-Compilation Dockerfile (Minimal Build for WebCam)
# -----------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

# build-essential: gcc, make等のビルドツール
# mingw-w64, mingw-w64-tools: Windows向けクロスコンパイル環境
# nasm, yasm: FFmpegのビルドに必要
# git: FFmpegのソース取得に必要
# ca-certificates: git clone でのHTTPS通信に必要
RUN apt-get -yqq update && apt-get install -yq --no-install-recommends \
    build-essential \
    mingw-w64 \
    mingw-w64-tools \
    nasm \
    yasm \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/workdir

RUN git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git ffmpeg

WORKDIR /tmp/workdir/ffmpeg

RUN ./configure \
    --arch=x86_64 \
    --target-os=mingw32 \
    --cross-prefix=x86_64-w64-mingw32- \
    --pkg-config-flags="--static" \
    --extra-ldflags="-static" \
    --disable-all \
    --disable-doc \
    --disable-debug \
    --disable-network \
    --enable-ffmpeg \
    --enable-avdevice \
    --enable-avcodec \
    --enable-avformat \
    --enable-swscale \
    --enable-avfilter \
    --enable-w32threads \
    --enable-indev=dshow \
    --enable-decoder=rawvideo,mjpeg,h264 \
    --enable-encoder=rawvideo \
    --enable-parser=mjpeg,h264 \
    --enable-demuxer=rawvideo,mjpeg,h264 \
    --enable-muxer=rawvideo \
    --enable-protocol=pipe,file \
    --enable-filter=format,scale,rotate,transpose,hflip,vflip,crop,null

RUN make -j$(nproc)

FROM scratch
COPY --from=builder /tmp/workdir/ffmpeg/ffmpeg.exe /ffmpeg.exe
