# Deepfake Detection System

Deepfake Detection System is a local deepfake detection system for images and videos.

It provides:

- Tri-state verdicts: REAL, FAKE, SUSPICIOUS
- Explainable outputs (reasons, confidence explanation, heatmap/timeline)
- Metadata and watermark checks for image authenticity signals
- Live backend processing steps shown in the upload UI
- Local scan history and user feedback storage in SQLite

## Current Project Status

This README reflects the current cleaned project structure and active runtime files.

Main active frontend pages:

- frontend/analysis.html
- frontend/video_result.html

Main backend entry:

- backend/app.py

Model checkpoint expected at:

- model/results/checkpoints/Mark-V.safetensors

## End-To-End Flow

1. User opens the app at / and uploads an image or video.
2. Frontend sends file to backend:
   - Image: POST /api/predict
   - Video: POST /api/predict_video
3. Backend initializes process tracking and streams step updates.
4. Media is preprocessed and analyzed.
5. Backend returns verdict, confidence, explanation, and processing details.
6. Result is stored in SQLite history and shown in frontend report page.

## Architecture

### Backend

File: backend/app.py

Responsibilities:

- Serve frontend pages and media history files
- Handle image and video inference requests
- Build explainable result payloads
- Track live backend steps with process IDs
- Persist scan history and feedback data

### Database

File: backend/database.py

Database file:

- backend/database.db

Stores:

- Scan history (filename, prediction, confidence, probabilities, path, timestamp, session)
- User feedback records (correct/incorrect, labels, confidence, optional copied media)

### Model

Core model files:

- model/src/models.py
- model/src/video_inference.py
- model/src/config.py

Checkpoint:

- model/results/checkpoints/Mark-V.safetensors

## Model Used

Model class: DeepfakeDetector (Mark-V family)

Multi-branch architecture in model/src/models.py:

- RGB Branch: EfficientNet V2 Small (spatial signals)
- Frequency Branch: CNN over FFT-derived features
- Patch Branch: local patch consistency analysis
- ViT Branch: Swin V2 Tiny (global context)

The branch features are concatenated and passed to a fusion classifier that outputs a fake score.

## Inference Logic

### Image Pipeline

Flow in backend/app.py:

1. Load and normalize image
2. Run metadata_checker and watermark_checker
3. Run model inference
4. Generate heatmap via model.get_heatmap
5. Build explainable verdict payload

Current image thresholds:

- FAKE when fake_probability >= 0.45
- REAL when fake_probability <= 0.15
- SUSPICIOUS otherwise

### Video Pipeline

Flow in model/src/video_inference.py + backend/app.py:

1. Optional ffmpeg re-encode for playback compatibility
2. Decode frames (Decord if available, else OpenCV fallback)
3. Sample frames and process in batches
4. Aggregate frame-level scores
5. Build verdict + timeline + explanation details

## Authenticity Signals

Image analysis includes additional checks:

- Metadata check (including C2PA/provenance-style signals where available)
- Invisible watermark check (Stable Diffusion style watermark detection support)

These signals can strengthen or adjust the final confidence interpretation.

## Live Backend Transparency

Endpoints:

- GET /api/process/<process_id>

Frontend sends X-Process-ID with upload requests and polls process status to show:

- Current step message
- Progress percentage
- Completion or failure status

## API Endpoints

- GET /api/health
- POST /api/predict
- POST /api/predict_video
- GET /api/process/<process_id>
- GET /api/history
- PATCH /api/history/<scan_id>
- DELETE /api/history/<scan_id>
- DELETE /api/history
- POST /api/feedback
- GET /api/feedback/stats
- GET /api/model-info

## Project Structure

Current top-level structure:

- backend/
- frontend/
- model/
- deepguard_env/
- Documentation/
- PROJECT_ARCHITECTURE.md
- START_PROJECT.md
- README.md
- setup.py
- inference.py

## How To Start

From project root:

PowerShell:

    cd backend
    & "..\\deepguard_env\\Scripts\\python.exe" app.py

Open in browser:

    http://localhost:7860

## Dependencies

See backend/requirements_web.txt

Key runtime packages:

- flask
- flask-cors
- torch
- torchvision
- opencv-python
- albumentations
- Pillow
- numpy
- safetensors
- c2pa-python
- invisible-watermark
- ExifRead

## Common Startup Issues

### ModuleNotFoundError: No module named flask

Cause:

- Running system python instead of deepguard_env python

Fix:

    & "..\\deepguard_env\\Scripts\\python.exe" app.py

If dependencies are missing in deepguard_env:

    & ".\\deepguard_env\\Scripts\\python.exe" -m pip install -r ".\\backend\\requirements_web.txt"

### PowerShell parser error with quoted executable path

Cause:

- Missing call operator when running a quoted executable path

Fix:

    & "..\\deepguard_env\\Scripts\\python.exe" app.py

## Additional Docs

- PROJECT_ARCHITECTURE.md: full architecture and flow details
- START_PROJECT.md: startup-focused quick guide



