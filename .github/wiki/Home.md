# nmstt — Wiki Home

**nmstt** is NeuralMimicry's standalone on-premises speech-to-text service, extracted from Refiner so speech recognition, motion planning, and audio decoding can be deployed, scaled, and maintained independently. It is built on Whisper-based inference and is tuned for native ARM64 performance.

> ☕ [Support NeuralMimicry on Crowdfunder](https://www.crowdfunder.co.uk/p/qr/aWggxwPW?utm_campaign=sharemodal&utm_medium=referral&utm_source=shortlink)

---

## Quick navigation

| Page | Description |
|---|---|
| [Getting Started](Getting-Started) | Build and run nmstt locally |
| [API Reference](API-Reference) | `/transcribe`, `/gesture-plan`, `/health` |
| [Configuration](Configuration) | `NMSTT_*` environment variables |
| [Refiner Integration](Refiner-Integration) | Connecting Refiner to nmstt |
| [ARM64 Deployment](ARM64-Deployment) | systemd service, native installer |
| [Contributing](Contributing) | Running tests, PR guidelines |

---

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
  --workers 23
```

## API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health check |
| `/transcribe` | POST | Transcribe audio (WAV/WebM/OGG) |
| `/gesture-plan` | POST | Avatar gesture planning for transcribed text |

## Refiner integration

Set these variables on Refiner to route voice traffic to nmstt:

```bash
REFINER_STT_BACKEND=server
REFINER_STT_SERVER_URL=http://127.0.0.1:7079
REFINER_STT_SERVER_TIMEOUT=25
REFINER_STT_SERVER_PREPROCESS=0
```

For Kubernetes: `http://nmstt.nmstt.svc.cluster.local:7079`

## Full local stack

```bash
./rag_demo/scripts/start_refiner_stack.sh   # starts nmstt + Refiner together
```

## Get involved

- 🐛 [Report a bug or request a feature](https://github.com/neuralmimicry/nmstt/issues)
- 💬 [Join the discussion](https://github.com/neuralmimicry/nmstt/discussions)
- 📧 Direct support: [info@neuralmimicry.ai](mailto:info@neuralmimicry.ai) · **£1,000/day + VAT**
- 🌐 [neuralmimicry.ai](https://neuralmimicry.ai)
