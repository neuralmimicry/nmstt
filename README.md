# nmstt

## Sponsor NeuralMimicry

`nmstt` is an open-source, on-premises speech-to-text service built on Whisper-based inference — designed for privacy-first deployments, native ARM64 performance, and gesture and avatar-motion planning without relying on cloud STT vendors. NeuralMimicry is an independent open-source initiative and we rely on community support to sustain this work.

**[☕ Support us on Crowdfunder](https://www.crowdfunder.co.uk/p/qr/aWggxwPW?utm_campaign=sharemodal&utm_medium=referral&utm_source=shortlink)**

---

`nmstt` is NeuralMimicry's standalone speech-to-text service.
It was extracted from Refiner so speech recognition, motion planning, and audio decoding can be deployed, scaled, and maintained independently while keeping the existing Refiner `/api/voice/stt` contract stable.

## Scope

- On-prem speech-to-text using Whisper-based inference
- Browser-audio decoding and resampling inside the service
- Gesture and avatar-motion planning for Refiner-compatible responses
- High-throughput native `arm64` deployment support
- Backward-compatible configuration aliases for existing `REFINER_STT_*` settings

## Platform topology

`nmstt` is intentionally not exposed as a second public API origin.
The stable public path remains:

1. `https://neuralmimicry.ai`
2. `https://api.neuralmimicry.ai`
3. `vega.neuralmimicry.ai`
4. `spirit.neuralmimicry.ai`
5. internal service routing to `nmstt`

Refiner calls `nmstt` on the internal network, normally at `http://nmstt.nmstt.svc.cluster.local:7079`.
That keeps browser integrations and the website unchanged while STT runs as an independent deployable service.

## Compatibility

The service now prefers `NMSTT_*` environment variables.
For migration safety it still accepts the legacy `REFINER_STT_*` names where the binary previously read them directly.
That means Refiner can switch over incrementally instead of requiring an all-at-once configuration cut-over.
Legacy `refiner-stt` unit names and `rag_demo/stt_rust` install paths are intentionally no longer shipped from this repository.

## Persistence model

Continuum deployment now defaults `nmstt` to shared NFS-backed model storage:

- PVC: `nmstt-models`
- storage class: `continuum-shared`
- access mode: `ReadWriteMany`
- default mounted model path: `/var/lib/nmstt/models/ggml-tiny.en.bin`

The tenant role seeds the mounted model file from the image bundle on first deploy if the PVC is empty.
That keeps Whisper assets durable across pod recreation and decouples model lifecycle from image rollout timing.

## Build

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Run

```bash
./target/release/nmstt \
  --model /opt/nmstt/models/ggml-base.en.bin \
  --bind 127.0.0.1:7079 \
  --lang en-GB \
  --threads 2 \
  --workers 23 \
  --max-audio-bytes 8000000
```

## API

- `GET /health`
- `POST /transcribe`
- `POST /gesture-plan`

The response contract intentionally remains Refiner-compatible so existing frontend and backend consumers do not need a second parser.

## Runtime Settings

Preferred settings:

- `NMSTT_GESTURE_ENABLED=1`
- `NMSTT_BSL_ENABLED=1`
- `NMSTT_GESTURE_DEFAULT_MODE=gesticulation`
- `NMSTT_GESTURE_DEFAULT_AVATAR_MODE=chat`
- `NMSTT_BUILTIN_CONTEXT_ENABLED=1`
- `NMSTT_BUILTIN_CONTEXT_PROMPT=...`
- `NMSTT_PROMPT_ALLOW_CLIENT=0`
- `NMSTT_CANONICALIZE_ENTITIES=1`
- `NMSTT_COLLABORATION_DEFAULT=0`

Legacy aliases:

- `REFINER_STT_GESTURE_ENABLED`
- `REFINER_STT_BSL_ENABLED`
- `REFINER_STT_GESTURE_DEFAULT_MODE`
- `REFINER_STT_GESTURE_DEFAULT_AVATAR_MODE`
- `REFINER_STT_BUILTIN_CONTEXT_ENABLED`
- `REFINER_STT_BUILTIN_CONTEXT_PROMPT`
- `REFINER_STT_PROMPT_ALLOW_CLIENT`
- `REFINER_STT_CANONICALIZE_ENTITIES`
- `REFINER_STT_COLLABORATION_DEFAULT`

## Native arm64 Install

Use the provided installer:

```bash
./install_native_arm64.sh
```

Default install targets:

- binary directory: `/opt/nmstt`
- systemd unit: `nmstt.service`
- env file: `/etc/default/nmstt`

Manual install:

```bash
sudo install -d -m 755 /opt/nmstt
sudo install -m 755 target/release/nmstt /opt/nmstt/nmstt
sudo install -m 644 nmstt.service /etc/systemd/system/nmstt.service
sudo install -m 644 native_arm64_46core.env /etc/default/nmstt
sudo systemctl daemon-reload
sudo systemctl enable --now nmstt
sudo systemctl status nmstt
```

## Continuum deployment

Tenant playbook:

- `/home/pbisaacs/Developer/swarmhpc/swarmhpc/ansible/continuum_tenant_nmstt_site.yml`

Role:

- `roles/continuum_tenant_nmstt`

Deployment defaults assume:

- internal service URL `http://nmstt.nmstt.svc.cluster.local:7079`
- NFS-backed shared model PVC on the `continuum-shared` storage class
- image-seeded default model `ggml-tiny.en.bin` copied into the PVC when missing

Relevant role variables:

- `continuum_tenant_nmstt_persistent_models_enable`
- `continuum_tenant_nmstt_models_mount_path`
- `continuum_tenant_nmstt_models_pvc_name`
- `continuum_tenant_nmstt_models_storage_class_name`
- `continuum_tenant_nmstt_models_access_modes`
- `continuum_tenant_nmstt_models_storage_size`
- `continuum_tenant_nmstt_model_path`
- `continuum_tenant_nmstt_bundled_model_path`

## Refiner Integration

Set these variables on Refiner:

```bash
REFINER_STT_BACKEND=server
REFINER_STT_SERVER_URL=http://nmstt.nmstt.svc.cluster.local:7079
REFINER_STT_SERVER_TIMEOUT=25
REFINER_STT_SERVER_PREPROCESS=0
REFINER_STT_GESTURE_PREFER_SERVER=1
```

This keeps Refiner's public API unchanged while the actual STT workload runs in the separate `nmstt` project.

For the wider service-boundary design, see `/home/pbisaacs/Developer/neuralmimicry/rag_demo/SERVICE_SPLIT_ARCHITECTURE.md`.

## OpenAPI

The service spec is in `openapi_stt.yaml`.
