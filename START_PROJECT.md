# Deepfake Detection System Start Guide

Use this guide to start the project locally.

## 1. Prerequisites

- Windows
- Python environment already prepared in `deepguard_env`
- Model checkpoint present at `model/results/checkpoints/Mark-V.safetensors`

If you are missing the bundled environment, create your own Python environment and install the dependencies from `backend/requirements_web.txt`.

## 2. Start The Backend

From the project root:

```powershell
cd backend
& "..\deepguard_env\Scripts\python.exe" app.py
```

The backend starts on:

- `http://127.0.0.1:7860`
- `http://localhost:7860`

## 3. Open The App

After the server starts, open the browser at:

```text
http://localhost:7860
```

This loads the upload-first analysis page.

## 4. What To Expect On Startup

When the server starts successfully, it will:

- load the Mark-V checkpoint
- initialize the image transform
- prepare the SQLite database
- expose the API endpoints
- serve the frontend files

## 5. Main Endpoints

- `GET /api/health` - health and model status
- `POST /api/predict` - image analysis
- `POST /api/predict_video` - video analysis
- `GET /api/process/<process_id>` - live progress feed
- `GET /api/history` - saved scans

## 6. Basic Usage Flow

1. Start the backend.
2. Open the upload page.
3. Choose an image or video.
4. Watch the live backend steps while it processes.
5. Review the verdict, explanation, and saved history entry.

## 7. Optional Checks

If you want to verify the backend before uploading media, open:

```text
http://localhost:7860/api/health
```

If the response shows the model as ready, the app is prepared for analysis.

## 8. Common Problems

- If the server cannot find the model, confirm that `model/results/checkpoints/Mark-V.safetensors` exists.
- If the backend cannot import dependencies, reinstall the packages listed in `backend/requirements_web.txt`.
- If a port conflict appears, stop the other process using port 7860 or change the port in [backend/app.py](backend/app.py).

## 9. Shut Down

Stop the running backend terminal when you are finished.

## 10. Short Version

```powershell
cd backend
& "..\deepguard_env\Scripts\python.exe" app.py
```

Then open `http://localhost:7860`.


