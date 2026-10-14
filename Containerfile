ARG TARGET_PAGE_SIZE=4k
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
ARG TARGET_PAGE_SIZE
LABEL org.opencontainers.image.page-size="${TARGET_PAGE_SIZE}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/nmstt /usr/local/bin/nmstt
COPY models ./models
# Text-to-speech: Piper binary (bundles espeak-ng data) and an en_GB voice,
# both sha256-verified. nmstt enables TTS automatically when /app/voices exists.
ARG TARGETARCH
ARG PIPER_RELEASE=2023.11.14-2
ARG PIPER_SHA256_ARM64=fea0fd2d87c54dbc7078d0f878289f404bd4d6eea6e7444a77835d1537ab88eb
ARG PIPER_SHA256_AMD64=a50cb45f355b7af1f6d758c1b360717877ba0a398cc8cbe6d2a7a3a26e225992
ARG VOICE=en_GB-alan-medium
ARG VOICE_URL=https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium
ARG VOICE_SHA256=0a309668932205e762801f1efc2736cd4b0120329622adf62be09e56339d3330
ARG VOICE_JSON_SHA256=c0f0d124e5895c00e7c03b35dcc8287f319a6998a365b182deb5c8e752ee8c1e
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends curl; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      arm64) pa=aarch64; ps="$PIPER_SHA256_ARM64" ;; \
      amd64) pa=x86_64; ps="$PIPER_SHA256_AMD64" ;; \
      *) echo "unsupported arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/piper.tgz "https://github.com/rhasspy/piper/releases/download/${PIPER_RELEASE}/piper_linux_${pa}.tar.gz"; \
    echo "$ps  /tmp/piper.tgz" | sha256sum -c -; \
    tar -xzf /tmp/piper.tgz -C /opt; rm /tmp/piper.tgz; \
    mkdir -p /app/voices; \
    curl -fsSL -o "/app/voices/${VOICE}.onnx" "${VOICE_URL}/${VOICE}.onnx"; \
    curl -fsSL -o "/app/voices/${VOICE}.onnx.json" "${VOICE_URL}/${VOICE}.onnx.json"; \
    echo "$VOICE_SHA256  /app/voices/${VOICE}.onnx" | sha256sum -c -; \
    echo "$VOICE_JSON_SHA256  /app/voices/${VOICE}.onnx.json" | sha256sum -c -; \
    echo "Build check." | /opt/piper/piper --model "/app/voices/${VOICE}.onnx" --output_file /tmp/check.wav; \
    test "$(head -c 4 /tmp/check.wav)" = RIFF; rm /tmp/check.wav; \
    apt-get purge -y curl; apt-get autoremove -y; rm -rf /var/lib/apt/lists/*
ENV NMSTT_TTS_PIPER_BIN=/opt/piper/piper NMSTT_TTS_VOICE_DIR=/app/voices NMSTT_TTS_DEFAULT_VOICE=en_GB-alan-medium
EXPOSE 7079
ENTRYPOINT ["/usr/local/bin/nmstt"]
CMD ["--model", "/app/models/ggml-tiny.en.bin", "--bind", "0.0.0.0:7079", "--lang", "en-GB", "--threads", "2", "--workers", "4", "--max-audio-bytes", "8000000"]

# OCI metadata (final stage) so GHCR links the package to its source repository.
ARG VCS_REF=unknown
ARG BUILD_VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/neuralmimicry/nmstt" \
      org.opencontainers.image.url="https://github.com/neuralmimicry/nmstt" \
      org.opencontainers.image.description="On-premises speech-to-text service (Whisper-based), privacy-first and ARM64-native, with gesture and avatar-motion planning" \
      org.opencontainers.image.vendor="NeuralMimicry" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${BUILD_VERSION}"
