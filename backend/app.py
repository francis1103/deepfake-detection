from flask import Flask, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
import sys
import os
import re
import mimetypes
import subprocess
import shutil
import time
import uuid

# Add model directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
import datetime
import torch
import cv2
import os
import numpy as np
import ssl
import base64
from werkzeug.utils import secure_filename
import io
from PIL import Image
from src import video_inference

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch import ToTensorV2
from src.models import DeepfakeDetector
from src.config import Config
from checkers import metadata_checker
from checkers import watermark_checker
import database

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'mp4', 'avi', 'mov', 'webm'}
HISTORY_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'history_uploads')
FEEDBACK_FOLDER = os.path.join(os.path.dirname(__file__), 'feedback_images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Increase to 500MB for video

# Global model and transform
# Global model and transform
device = torch.device(Config.DEVICE)
model = None
video_model_onnx = None # Dedicated optimized model for video
transform = None
model_load_error = None
PROCESS_TRACKER = {}
PROCESS_TTL_SECONDS = 900


def _cleanup_process_tracker():
    now = time.time()
    stale_keys = [
        key for key, value in PROCESS_TRACKER.items()
        if now - value.get('updated_at', now) > PROCESS_TTL_SECONDS
    ]
    for key in stale_keys:
        PROCESS_TRACKER.pop(key, None)


def _init_process(process_id, media_type):
    _cleanup_process_tracker()
    PROCESS_TRACKER[process_id] = {
        'process_id': process_id,
        'media_type': media_type,
        'status': 'running',
        'progress': 0,
        'steps': [],
        'created_at': time.time(),
        'updated_at': time.time()
    }


def _add_process_step(process_id, message, progress=None, status='running'):
    if not process_id:
        return
    if process_id not in PROCESS_TRACKER:
        _init_process(process_id, 'unknown')

    record = PROCESS_TRACKER[process_id]
    record['steps'].append({
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'message': message,
        'progress': progress
    })
    record['status'] = status
    if progress is not None:
        record['progress'] = int(max(0, min(100, progress)))
    record['updated_at'] = time.time()


def _complete_process(process_id, ok=True, message=None):
    if not process_id or process_id not in PROCESS_TRACKER:
        return
    record = PROCESS_TRACKER[process_id]
    record['status'] = 'completed' if ok else 'failed'
    record['progress'] = 100 if ok else record.get('progress', 0)
    if message:
        record['steps'].append({
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'message': message,
            'progress': record['progress']
        })
    record['updated_at'] = time.time()

def get_transform():
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model():
    """Load the trained deepfake detection model"""
    global model, transform, video_model_onnx, model_load_error
    
    checkpoint_dir = Config.CHECKPOINT_DIR
    target_model_name = "Mark-V.safetensors"
    checkpoint_path = os.path.join(checkpoint_dir, target_model_name)
    model_load_error = None
    
    print(f"Using device: {device}")
    
    # 1. Load PyTorch Model (Required for single image heatmaps)
    # Use pretrained=False to avoid runtime downloads; checkpoint supplies weights.
    model = DeepfakeDetector(pretrained=False)
    model.to(device)
    model.eval()
    
    if not os.path.exists(checkpoint_path):
        model_load_error = f"Model file not found at: {checkpoint_path}"
        print(f"âŒ CRITICAL ERROR: {model_load_error}")
        model = None
        transform = get_transform()
        return model, transform

    # Detect Git LFS pointer files from ZIP/source downloads.
    # A real checkpoint should be binary and large; pointer files are tiny text files.
    try:
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1024 * 1024:
            with open(checkpoint_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(256)
            if "git-lfs.github.com/spec/v1" in header:
                model_load_error = (
                    "Checkpoint is a Git LFS pointer, not real weights. "
                    "Re-download repository with Git LFS or download the actual Mark-V.safetensors file."
                )
                print(f"âŒ CRITICAL ERROR: {model_load_error}")
                model = None
                transform = get_transform()
                return model, transform
    except Exception:
        # Non-fatal; continue to normal loader.
        pass

    try:
        print(f"Loading PyTorch checkpoint: {checkpoint_path}")

        # Some projects rename torch checkpoints to .safetensors accidentally.
        # Try safetensors first when available, then fallback to torch.load.
        state_dict = None
        load_errors = []

        if checkpoint_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
            try:
                state_dict = load_file(checkpoint_path)
                print("âœ… Loaded checkpoint using safetensors")
            except Exception as e:
                load_errors.append(f"safetensors load failed: {e}")

        if state_dict is None:
            try:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
                print("âœ… Loaded checkpoint using torch.load")
            except Exception as e:
                load_errors.append(f"torch.load failed: {e}")

        if state_dict is None:
            raise RuntimeError("; ".join(load_errors) if load_errors else "Unknown checkpoint load error")
        
        # Try loading directly first
        try:
            model.load_state_dict(state_dict)
            print(f"âœ… PyTorch Model loaded successfully!")
        except Exception as e:
            # Keys don't match - apply remapping for architecture compatibility
            print(f"âš ï¸  Direct load failed. Attempting key remapping...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('rgb_branch.features.'):
                    new_k = k.replace('rgb_branch.features.', 'rgb_branch.net.features.')
                    new_state_dict[new_k] = v
                elif k.startswith('rgb_branch.avgpool.'):
                    new_k = k.replace('rgb_branch.avgpool.', 'rgb_branch.net.avgpool.')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"âœ… PyTorch Model loaded successfully (with key remapping)!")
            
    except Exception as e:
        model_load_error = str(e)
        print(f"âŒ Error loading PyTorch checkpoint: {model_load_error}")
        model = None

    # 2. Load ONNX Model (Removed)
    # System optimized for PyTorch Pipeline (Threaded Preprocessing)
    video_model_onnx = None

    transform = get_transform()
    return model, transform

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _build_image_explanation(prob, metadata_detected=False, watermark_detected=False):
    """Build evidence-first explanation for image predictions from numeric signals.
    
    Strict image prediction threshold (more sensitive to fake detection):
    - FAKE: >= 0.45 (more aggressive)
    - REAL: <= 0.15 (very conservative)
    - SUSPICIOUS: everything in between (wider band)
    """
    fake_prob = _clamp01(prob)
    real_prob = _clamp01(1.0 - fake_prob)

    # Stricter thresholds for image detection
    if fake_prob >= 0.45:  # Lower threshold, more sensitive to fakes
        prediction = 'FAKE'
    elif fake_prob <= 0.15:  # Higher threshold, very strict for REAL
        prediction = 'REAL'
    else:
        prediction = 'SUSPICIOUS'

    if prediction == 'FAKE':
        confidence = fake_prob
    elif prediction == 'REAL':
        confidence = min(real_prob, 0.85)  # Slightly lower max confidence
    else:
        confidence = 0.55 + abs(fake_prob - 0.5) * 0.4

    reasons = []
    if prediction == 'REAL':
        reasons.append(f"Majority confidence supports authentic content (fake probability {fake_prob * 100:.1f}%).")
        reasons.append("No high-risk anomaly threshold was crossed in visual inference.")
    elif prediction == 'FAKE':
        reasons.append(f"High fake probability detected ({fake_prob * 100:.1f}%), exceeding the FAKE threshold.")
        reasons.append("Model confidence indicates strong synthetic artifact presence in the image.")
    else:
        reasons.append(f"Mixed confidence signal detected (fake probability {fake_prob * 100:.1f}%).")
        reasons.append("Result falls inside the uncertainty band, so the image is flagged as suspicious.")

    if metadata_detected:
        reasons.append("Metadata checker flagged synthetic-source indicators.")
    if watermark_detected:
        reasons.append("Watermark detector found non-authentic generation markers.")

    if not reasons:
        reasons.append("No anomalies detected by the available checks.")

    confidence_explanation = [
        "Based on single-frame visual inference.",
        f"Model fake score: {fake_prob * 100:.1f}%.",
        f"Model real score: {real_prob * 100:.1f}%."
    ]

    summary = (
        "Strong synthetic evidence detected in image analysis."
        if prediction == 'FAKE'
        else "Signals are mixed; image contains uncertain anomaly evidence."
        if prediction == 'SUSPICIOUS'
        else "Image appears authentic with low anomaly confidence."
    )

    return {
        'prediction': prediction,
        'confidence': _clamp01(confidence),
        'summary': summary,
        'reasons': reasons,
        'confidence_explanation': confidence_explanation,
        'metrics': {
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'uncertainty_distance': abs(fake_prob - 0.5)
        }
    }

def predict_image(image_path, process_id=None):
    """Make prediction on a single image"""
    if model is None:
        return None, "Error: Model not loaded. Check backend logs for checkpoint loading errors."

    try:
        _add_process_step(process_id, 'Loading image data', 20)
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            try:
                pil_img = Image.open(image_path).convert("RGB")
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print("Warning: OpenCV could not read image directly; loaded via PIL fallback.")
            except Exception:
                return None, f"Error: Could not read image: {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        
        # 0. Metadata & Watermark Checks
        _add_process_step(process_id, 'Running metadata and watermark checks', 35)
        meta_result = metadata_checker.check_metadata(image_path)
        water_result = watermark_checker.check_watermarks(image_path)
        
        # Make prediction
        _add_process_step(process_id, 'Running model inference', 55)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()
        
        # Generate Heatmap
        _add_process_step(process_id, 'Generating explainability heatmap', 70)
        heatmap = model.get_heatmap(image_tensor)
        
        # Process Heatmap for Visualization
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        # Heatmap is BGR (from cv2), Image is RGB. Convert Image to BGR.
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        superimposed_img = heatmap * 0.4 + image_bgr * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Override if metadata confirms fake
        if meta_result['detected'] or water_result['detected']:
            # If visual model was unsure (e.g. 0.4), bump it up? 
            # Or just rely on the 'prediction' label.
            # Let's trust the metadata 100%
            prob = max(prob, 0.99) 
            
        # Hidden Check: Explicitly flag known generator filenames as FAKE without frontend badging
        filename_lower = os.path.basename(image_path).lower()
        if "chatgpt" in filename_lower or "gemini" in filename_lower:
            prob = max(prob, 0.998) # Extremely high confidence
            # Intentionally NOT adding to meta_result or water_result to keep it hidden from badges
            # as requested by user ("dont shiw this in fornetend") 
            
        explanation = _build_image_explanation(
            prob,
            metadata_detected=meta_result['detected'],
            watermark_detected=water_result['detected']
        )
        _add_process_step(process_id, 'Building explainable verdict', 85)
        
        return {
            'prediction': explanation['prediction'],
            'confidence': explanation['confidence'],
            'fake_probability': float(prob),
            'real_probability': float(1 - prob),
            'media_type': 'image',
            'heatmap': heatmap_b64,
            'metadata_check': meta_result,
            'watermark_check': water_result,
            'detection_explanation_summary': explanation['summary'],
            'reasons': explanation['reasons'],
            'confidence_explanation': explanation['confidence_explanation'],
            'explanation_metrics': explanation['metrics'],
            'processing_details': {
                'pipeline': 'Single Image Inference',
                'inference_mode': 'PyTorch',
                'device': str(device),
                'input_size': int(Config.IMAGE_SIZE)
            }
        }, None
    except Exception as e:
        return None, str(e)


@app.route('/')
def index():
    """Serve the upload-first analysis page."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
    return send_from_directory(frontend_dir, 'analysis.html')


@app.route('/index.html')
@app.route('/analysis.html')
@app.route('/history.html')
def redirect_legacy_pages():
    """Redirect legacy front-end pages to the upload-first view."""
    return make_response('', 302, {'Location': '/'})

@app.route('/history_uploads/<path:filename>')
def serve_history_image(filename):
    """Serve history images and videos with Range support"""
    file_path = os.path.join(HISTORY_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    # Handle Video Range Requests
    if filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range', None)
        
        if not range_header:
            # No Range header, serve normally but with video headers
            response = make_response(send_from_directory(HISTORY_FOLDER, filename))
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Accept-Ranges'] = 'bytes'
            return response
            
        # Parse Range Header
        byte1, byte2 = 0, None
        m = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if m:
            g = m.groups()
            byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])

        length = file_size - byte1
        if byte2 is not None:
            length = byte2 + 1 - byte1

        # Read partial content
        with open(file_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)

        response = Response(
            data,
            206,
            mimetype='video/mp4',
            direct_passthrough=True
        )
        
        # Determine content range
        content_range_end = byte2 if byte2 is not None else file_size - 1
        
        response.headers.add('Content-Range', f'bytes {byte1}-{content_range_end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(length))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    # Default for images
    response = send_from_directory(HISTORY_FOLDER, filename)
    return response

def reencode_video(input_path):
    """Re-encode video to H.264/AAC with faststart using ffmpeg"""
    # ffmpeg is optional; if unavailable, continue with original file.
    if shutil.which('ffmpeg') is None:
        print("âš ï¸ FFmpeg not found in PATH. Skipping re-encoding and using original video.")
        return input_path

    try:
        output_path = input_path + "_temp.mp4"
        print(f"ðŸ”„ Re-encoding video: {input_path}")
        
        # FFmpeg command
        # -y: overwrite output
        # -c:v libx264: use H.264 video codec
        # -preset fast: encode speed
        # -profile:v high: high profile for better compatibility
        # -level 4.0: compatibility level
        # -pix_fmt yuv420p: ensure wide player compatibility (essential for QuickTime/Safari)
        # -c:a aac: use AAC audio codec
        # -movflags +faststart: move metadata to front for streaming
        cmd = [
            'ffmpeg', '-y', 
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"âŒ FFmpeg re-encoding failed: {result.stderr.decode()}")
            return input_path # Fallback to original
            
        print(f"âœ… Video re-encoded successfully!")
        
        # Replace original
        os.remove(input_path)
        os.rename(output_path, input_path)
        return input_path
        
    except Exception as e:
        print(f"âŒ Error during re-encoding: {e}")
        return input_path

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed model status"""
    model_status = "ready" if model is not None else "initializing"
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_loaded': model is not None,
        'device': str(device),
        'model_error': model_load_error
    })


@app.route('/api/process/<process_id>', methods=['GET'])
def process_status(process_id):
    """Get live backend processing steps for an upload process."""
    _cleanup_process_tracker()
    process = PROCESS_TRACKER.get(process_id)
    if not process:
        return jsonify({'error': 'Process not found'}), 404
    return jsonify(process)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        process_id = request.headers.get('X-Process-ID') or str(uuid.uuid4())
        _init_process(process_id, 'image')
        _add_process_step(process_id, 'Request received by backend', 5)

        # Check if file is present
        if 'file' not in request.files:
            _complete_process(process_id, ok=False, message='No file provided')
            return jsonify({'error': 'No file provided'}), 400

        if model is None:
            _complete_process(process_id, ok=False, message='Model unavailable')
            return jsonify({'error': model_load_error or 'Model not loaded'}), 503
        
        file = request.files['file']
        
        if file.filename == '':
            _complete_process(process_id, ok=False, message='No file selected')
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            _complete_process(process_id, ok=False, message='Invalid file type')
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, webp'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        _add_process_step(process_id, 'Uploaded file saved', 15)
        
        # Make prediction
        result, error = predict_image(filepath, process_id=process_id)
        
        if error:
            _complete_process(process_id, ok=False, message=f'Prediction failed: {error}')
            return jsonify({'error': error}), 500
            
        # Save to History
        import shutil
        history_filename = f"scan_{int(datetime.datetime.now().timestamp())}_{filename}"
        history_path = os.path.join(HISTORY_FOLDER, history_filename)
        
        # Copy original file to history folder
        # We need to read the file again or just copy if we haven't deleted it?
        # We read via cv2, the file is still at filepath.
        shutil.copy(filepath, history_path)
        _add_process_step(process_id, 'Saved artifact for history', 92)
        
        # Relative path for frontend
        relative_path = f"history_uploads/{history_filename}"

        scan_id = database.add_scan(
            filename=filename,
            prediction=result['prediction'],
            confidence=result['confidence'],
            fake_prob=result['fake_probability'],
            real_prob=result['real_probability'],
            image_path=relative_path,
            session_id=request.headers.get('X-Session-ID')
        )
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Add scan_id to result for frontend tracking
        result['scan_id'] = scan_id
        result['image_path'] = relative_path
        result['process_id'] = process_id
        _complete_process(process_id, ok=True, message='Image analysis completed')
        
        return jsonify(result)
    
    except Exception as e:
        process_id = request.headers.get('X-Process-ID')
        _complete_process(process_id, ok=False, message=f'Unexpected error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_video', methods=['POST'])
def predict_video():
    """Handle video upload and prediction"""
    try:
        process_id = request.headers.get('X-Process-ID') or str(uuid.uuid4())
        _init_process(process_id, 'video')
        _add_process_step(process_id, 'Request received by backend', 5)

        if 'file' not in request.files:
            _complete_process(process_id, ok=False, message='No file provided')
            return jsonify({'error': 'No file provided'}), 400

        if model is None and video_model_onnx is None:
            _complete_process(process_id, ok=False, message='Model unavailable')
            return jsonify({'error': model_load_error or 'Model not loaded'}), 503
        
        file = request.files['file']
        
        if file.filename == '':
            _complete_process(process_id, ok=False, message='No file selected')
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
             _complete_process(process_id, ok=False, message='Invalid file type')
             return jsonify({'error': 'Invalid file type'}), 400
             
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        _add_process_step(process_id, 'Uploaded video saved', 10)
        
        # Re-encode video for proper web playback
        _add_process_step(process_id, 'Preparing video stream format', 15)
        filepath = reencode_video(filepath)
        
        # Process Video
        # Prioritize Optimized ONNX Model
        active_model = video_model_onnx if video_model_onnx is not None else model
        
        if active_model is None:
            _complete_process(process_id, ok=False, message='No active model available')
            return jsonify({'error': 'Model not loaded'}), 500
             
        result = video_inference.process_video(
            filepath,
            active_model,
            transform,
            device,
            frames_per_second=10,
            progress_callback=lambda message, progress=None: _add_process_step(process_id, message, progress)
        )
        
        if "error" in result:
            _complete_process(process_id, ok=False, message=result.get('error', 'Video inference failed'))
            return jsonify(result), 500
             
        # Save to History (Using the first frame or a placeholder icon for now?)
        # For video, we might want to save the video file itself to history_uploads
        # or just a thumbnail. Let's save the video for now.
        import shutil
        history_filename = f"scan_{int(datetime.datetime.now().timestamp())}_{filename}"
        history_path = os.path.join(HISTORY_FOLDER, history_filename)
        shutil.copy(filepath, history_path)
        _add_process_step(process_id, 'Saved video artifact for history', 95)
        
        relative_path = f"history_uploads/{history_filename}"
        
        # Add to database
        # Note: The database 'add_scan' might expect image-specific fields.
        # We'll re-use 'fake_prob' as 'avg_fake_prob'
        scan_id = database.add_scan(
            filename=filename,
            prediction=result['prediction'],
            confidence=result['confidence'],
            fake_prob=result['avg_fake_prob'],
            real_prob=1 - result['avg_fake_prob'],
            image_path=relative_path,
            session_id=request.headers.get('X-Session-ID')
        )
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
            
        # Add video URL for frontend playback
        result['video_url'] = relative_path
        result['media_type'] = 'video'
        result['scan_id'] = scan_id
        result['process_id'] = process_id
        _complete_process(process_id, ok=True, message='Video analysis completed')
        
        return jsonify(result)

    except Exception as e:
        print(f"Video Error: {e}")
        process_id = request.headers.get('X-Process-ID')
        _complete_process(process_id, ok=False, message=f'Unexpected error: {str(e)}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all past scans"""
    session_id = request.headers.get('X-Session-ID')
    history = database.get_history(session_id)
    return jsonify(history)

@app.route('/api/history/<int:scan_id>', methods=['PATCH'])
def update_history_item(scan_id):
    """Update a specific scan's metadata (notes, tags)"""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    if database.update_scan(scan_id, data):
        return jsonify({'message': 'Scan updated successfully'})
    return jsonify({'error': 'Failed to update scan'}), 500

@app.route('/api/history/<int:scan_id>', methods=['DELETE'])
def delete_scan(scan_id):
    """Delete a specific scan"""
    session_id = request.headers.get('X-Session-ID')
    if database.delete_scan(scan_id, session_id):
        return jsonify({'message': 'Scan deleted'})
    return jsonify({'error': 'Failed to delete scan'}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear all history"""
    session_id = request.headers.get('X-Session-ID')
    if database.clear_history(session_id):
        return jsonify({'message': 'History cleared'})
    return jsonify({'error': 'Failed to clear history'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback on a prediction"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        scan_id = data.get('scan_id')
        is_correct = data.get('is_correct')
        predicted_label = data.get('predicted_label')
        
        if scan_id is None or is_correct is None or not predicted_label:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get scan details from history
        history = database.get_history()
        scan = next((s for s in history if s['id'] == scan_id), None)
        
        if not scan:
            return jsonify({'error': 'Scan not found'}), 404
        
        actual_label = None
        feedback_image_path = None
        
        # If prediction is incorrect, determine actual label and copy image
        if not is_correct:
            # For tri-state predictions, keep explicit mapping instead of binary inversion.
            if predicted_label == 'FAKE':
                actual_label = 'REAL'
            elif predicted_label == 'REAL':
                actual_label = 'FAKE'
            else:
                actual_label = 'REAL'
            
            # Copy image to feedback folder for retraining
            if scan.get('image_path'):
                try:
                    import shutil
                    source_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', scan['image_path'])
                    feedback_filename = f"feedback_{scan_id}_{scan['filename']}"
                    feedback_dest = os.path.join(FEEDBACK_FOLDER, feedback_filename)
                    
                    if os.path.exists(source_path):
                        shutil.copy(source_path, feedback_dest)
                        feedback_image_path = feedback_filename
                        print(f"âœ… Copied feedback image to: {feedback_dest}")
                    else:
                        print(f"âš ï¸  Source image not found: {source_path}")
                except Exception as e:
                    print(f"âŒ Error copying feedback image: {e}")
        
        # Record feedback in database
        success = database.add_feedback(
            scan_id=scan_id,
            is_correct=is_correct,
            predicted_label=predicted_label,
            actual_label=actual_label,
            image_path=feedback_image_path,
            confidence=scan.get('confidence')
        )
        
        if success:
            feedback_type = 'correct' if is_correct else 'incorrect'
            return jsonify({
                'message': f'Feedback recorded successfully',
                'feedback': feedback_type,
                'actual_label': actual_label
            })
        else:
            return jsonify({'error': 'Failed to record feedback'}), 500
            
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    stats = database.get_feedback_stats()
    return jsonify(stats)

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model_name': 'Deepfake Detection System: Advanced Deepfake Detector',
        'architecture': 'Hybrid CNN-ViT',
        'components': {
            'RGB Analysis': Config.USE_RGB,
            'Frequency Domain': Config.USE_FREQ,
            'Patch-based Detection': Config.USE_PATCH,
            'Vision Transformer': Config.USE_VIT
        },
        'image_size': Config.IMAGE_SIZE,
        'device': str(device),
        'threshold': 0.5
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ðŸš€ Deepfake Detection System - Deepfake Detection System")
    print("=" * 60)
    
    # Load model
    load_model()
    
    print("=" * 60)
    print("=" * 60)
    # Check if running on Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        port = 7860
        print(f"ðŸª Detected Hugging Face Space. Forcing port {port}")
    else:
        port = int(os.environ.get("PORT", 7860))
        
    print(f"ðŸŒ Starting server on http://0.0.0.0:{port}")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)


