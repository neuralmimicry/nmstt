FROM docker.io/library/rust:1.88-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        libclang-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo ./.cargo
COPY src ./src
# Build with a single Cargo job so the RK1 control-plane host can publish
# arm64 images without tripping over peak memory spikes during release builds.
RUN cargo build --release -j 1

FROM docker.io/library/debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/nmstt /usr/local/bin/nmstt
COPY models ./models
EXPOSE 7079
ENTRYPOINT ["/usr/local/bin/nmstt"]
CMD ["--model", "/app/models/ggml-tiny.en.bin", "--bind", "0.0.0.0:7079", "--lang", "en-GB", "--threads", "2", "--workers", "4", "--max-audio-bytes", "8000000"]
