# DrowsinessDetection Inference API

Local FastAPI server for the DMD eyes-state TCN checkpoint `saved_weights/best_accuracy.pt`.

## Model input contract (from checkpoint + code)

| Item | Value | Source |
| --- | --- | --- |
| Architecture | Causal TCN (`GazeZoneTCN` / `arch=tcn`) | `model/model.py`, checkpoint |
| Input tensor | `[batch, T, F]` float32 | `GazeZoneTCN.forward` |
| Window frames `T` | **5** | `checkpoint["window_size"]` / `args.window_size` |
| Feature count `F` | **22** | `checkpoint["feature_names"]` |
| Normalization | **None** (raw features) | No `feature_mean`/`feature_std` in checkpoint or training |
| Missing values (training CSV only) | empty / NaN → `0.0` when reading CSVs | `pre_process.dataloader.parse_feature_value` |
| Live API values | Must be finite; NaN/Inf rejected | API contract |
| Output | Softmax over 5 classes | `torch.softmax(logits)` |
| Sampling rate | **Not defined** in checkpoint or training code | Set optional `DROWSINESS_SAMPLING_RATE_HZ` to enforce a client rate |

### Exact ordered feature names

```text
face_detected
vision_confidence
yaw
pitch
roll
left_eye_valid
right_eye_valid
left_eye_aspect_ratio
right_eye_aspect_ratio
left_eyelid_gap
right_eyelid_gap
left_pupil_x
left_pupil_y
right_pupil_x
right_pupil_y
mouth_valid
mouth_aspect_ratio
inner_lip_gap
inner_mouth_area
hand_detected
hand_confidence
hand_near_mouth
```

### Exact class mapping

```text
close      -> 0
closing    -> 1
open       -> 2
opening    -> 3
undefined  -> 4
```

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DROWSINESS_CHECKPOINT_PATH` | `saved_weights/best_accuracy.pt` | Resolved relative to this package root |
| `DROWSINESS_API_KEY` | unset | When set, require `X-API-Key` |
| `DROWSINESS_DEVICE` | `auto` | `auto` → CUDA if available else CPU |
| `DROWSINESS_HOST` | `0.0.0.0` | Bind host |
| `DROWSINESS_PORT` | `8001` | Bind port |
| `DROWSINESS_SAMPLING_RATE_HZ` | unset | Optional exact rate enforcement |

See `.env.example`.

## Local setup

```bash
cd /Users/zaimazarnaz/Desktop/IOSApps/DriveSensAI/AIPackages/DrowsinessDetection

python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt

python -m uvicorn api.app:app \
  --host 0.0.0.0 \
  --port 8001
```

Optional auth + sampling-rate lock:

```bash
export DROWSINESS_API_KEY="replace-with-a-long-random-value"
export DROWSINESS_SAMPLING_RATE_HZ=15.0
```

## Health

```bash
curl http://127.0.0.1:8001/health
```

Swagger UI: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

## Example prediction

Requires exactly **5** samples and the exact feature name order.

```bash
curl -s http://127.0.0.1:8001/v1/drowsiness/predict \
  -H 'Content-Type: application/json' \
  -H "X-API-Key: $DROWSINESS_API_KEY" \
  -d '{
    "schema_version": 1,
    "session_id": "demo-session",
    "sequence_id": 1,
    "sent_at_utc": "2026-09-22T18:00:00Z",
    "sampling_rate_hz": 15.0,
    "feature_names": [
      "face_detected", "vision_confidence", "yaw", "pitch", "roll",
      "left_eye_valid", "right_eye_valid", "left_eye_aspect_ratio", "right_eye_aspect_ratio",
      "left_eyelid_gap", "right_eyelid_gap", "left_pupil_x", "left_pupil_y",
      "right_pupil_x", "right_pupil_y", "mouth_valid", "mouth_aspect_ratio",
      "inner_lip_gap", "inner_mouth_area", "hand_detected", "hand_confidence", "hand_near_mouth"
    ],
    "samples": [
      {"timestamp_ms": 1000, "values": [1,0.9,0,0,0,1,1,0.3,0.3,0.1,0.1,0,0,0,0,1,0.4,0.1,0.2,0,0,0]},
      {"timestamp_ms": 1067, "values": [1,0.9,0,0,0,1,1,0.3,0.3,0.1,0.1,0,0,0,0,1,0.4,0.1,0.2,0,0,0]},
      {"timestamp_ms": 1134, "values": [1,0.9,0,0,0,1,1,0.3,0.3,0.1,0.1,0,0,0,0,1,0.4,0.1,0.2,0,0,0]},
      {"timestamp_ms": 1201, "values": [1,0.9,0,0,0,1,1,0.3,0.3,0.1,0.1,0,0,0,0,1,0.4,0.1,0.2,0,0,0]},
      {"timestamp_ms": 1268, "values": [1,0.9,0,0,0,1,1,0.3,0.3,0.1,0.1,0,0,0,0,1,0.4,0.1,0.2,0,0,0]}
    ]
  }'
```

Example response shape:

```json
{
  "session_id": "demo-session",
  "sequence_id": 1,
  "label": "open",
  "label_index": 2,
  "confidence": 0.91,
  "probabilities": {
    "close": 0.02,
    "closing": 0.03,
    "open": 0.91,
    "opening": 0.02,
    "undefined": 0.02
  },
  "model_version": "best_accuracy",
  "inference_latency_ms": 8.4
}
```

## iOS endpoint URL format

Point the native client at your Mac/LAN IP (not `localhost` on device):

```text
http://<mac-lan-ip>:8001/v1/drowsiness/predict
```

Health:

```text
http://<mac-lan-ip>:8001/health
```

## Docker

```bash
cd /Users/zaimazarnaz/Desktop/IOSApps/DriveSensAI/AIPackages/DrowsinessDetection

docker build -t drowsiness-api .
docker run --rm -p 8001:8001 \
  -e DROWSINESS_API_KEY=replace-with-a-long-random-value \
  -e DROWSINESS_SAMPLING_RATE_HZ=15.0 \
  drowsiness-api
```

## Production HTTPS warning

This server is intended for **local / private-network** development. Do not expose port 8001 on the public internet without TLS termination, a strong API key, and network access controls.

## Tests

```bash
cd /Users/zaimazarnaz/Desktop/IOSApps/DriveSensAI/AIPackages/DrowsinessDetection
source .venv/bin/activate
python -m pytest -q
```
