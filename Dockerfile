# Build with CUDA 12.9, even if your host's nvidia-smi prints "CUDA Version: 13".
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 git cmake build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pascal.py models.lock.json ./
COPY pascal_stack ./pascal_stack
COPY patches ./patches
ARG BUILD_JOBS=2
RUN python3 pascal.py --state-dir /build build --kernels pascal --jobs ${BUILD_JOBS}

FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /build/build-cuda61-pascal/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /build/build-cuda61-pascal/bin/llama-bench /usr/local/bin/llama-bench
COPY --from=build /build/build-cuda61-pascal/pascal-build.json /usr/local/bin/pascal-build.json
COPY pascal.py models.lock.json ./
COPY pascal_stack ./pascal_stack
COPY LICENSE ./LICENSE
COPY licenses ./licenses
ENV PASCAL_SERVER=/usr/local/bin/llama-server
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["python3", "/app/pascal.py", "--state-dir", "/state", "--models-dir", "/models"]
CMD ["serve"]
