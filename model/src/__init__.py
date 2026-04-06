import cv2
import torch
import numpy as np
import os
import time
import base64
from queue import Queue, Empty
from threading import Thread, Event
import traceback


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _variance_proxy(values):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.var(arr))


def _build_video_explanation(verdict, avg_prob, max_prob, fake_ratio, processed_frames, suspicious_count, variance_score):
    reasons = []

    if verdict == "FAKE":
        reasons.append(f"Multiple high-risk frames detected (max fake probability {max_prob * 100:.1f}%).")
        reasons.append(f"Suspicious frame ratio is elevated at {fake_ratio * 100:.1f}%.")
        reasons.append(f"Average anomaly confidence is high ({avg_prob * 100:.1f}%).")
    elif verdict == "SUSPICIOUS":
        reasons.append(f"Mixed evidence detected (average fake probability {avg_prob * 100:.1f}%).")
        reasons.append(f"A limited subset of frames is suspicious ({fake_ratio * 100:.1f}% ratio).")
        reasons.append("Anomaly signals are present but not strong enough for a FAKE verdict.")
    else:
        reasons.append(f"Majority of sampled frames are low-risk (average fake probability {avg_prob * 100:.1f}%).")
        reasons.append(f"Suspicious frame ratio remains low at {fake_ratio * 100:.1f}%.")
        reasons.append("No sustained high-confidence anomaly pattern was observed.")

    confidence_explanation = [
        f"Based on {processed_frames} frames analyzed.",
        f"{suspicious_count} frames exceeded the suspicious threshold (>50%).",
        f"Maximum anomaly score observed: {max_prob * 100:.1f}%.",
        f"Temporal variance score: {variance_score:.3f}."
    ]

    summary = (
        "The video is classified as FAKE due to strong and repeated anomaly evidence."
        if verdict == "FAKE"
        else "The video is classified as SUSPICIOUS due to mixed anomaly evidence."
        if verdict == "SUSPICIOUS"
        else "The video is classified as REAL because anomaly evidence stays consistently low."
    )

    return summary, reasons, confidence_explanation

# Try to import decord
try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("âš ï¸ Decord not found. Falling back to OpenCV for video decoding.")



class VideoPipeline:
    def __init__(self, 
                 model, 
                 transform, 
                 device, 
                 batch_size=16, 
                 queue_size=128, 
                 num_workers=2,
                 progress_callback=None):
        """
        Production-Ready Deepfake Detection System Video Analysis Pipeline.
        
        Features:
        - Hardware-Accelerated Decoding (Decord)
        - Producer-Consumer Threading
        - Pinned Memory Transfer (CPU -> GPU)
        - Batch Inference
        - Sparse Sampling
        """
        self.model = model
        self.transform = transform
        self.device = device
        self.batch_size = batch_size
        self.batch_queue = Queue(maxsize=queue_size)
        self.result_queue = Queue()
        self.stop_event = Event()
        self.num_workers = num_workers
        self.progress_callback = progress_callback
        
        # Determine if we are using ONNX or PyTorch
        self.is_onnx = False
        
        # Load Face Detector (Lazy load in workers usually, but init here for simplicity)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _emit_progress(self, message, progress=None):
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception:
                pass

    def _get_pinned_memory_buffer(self, batch_tensor):
        """
        Wraps a tensor in pinned memory for faster CPU-to-GPU transfer.
        """
        if self.device.type == 'cuda' and not self.is_onnx:
            if not batch_tensor.is_pinned():
                return batch_tensor.pin_memory()
        return batch_tensor

    def _producer_worker(self, video_path, step):
        """
        Producer Thread: Decodes frames -> Preprocesses -> Batches -> Pushes to Queue
        """
        from concurrent.futures import ThreadPoolExecutor
        
        try:
            # 1. Open Video
            use_decord = DECORD_AVAILABLE
            if use_decord:
                try:
                    vr = VideoReader(video_path, ctx=cpu(0))
                    total_frames = len(vr)
                    indices = list(range(0, total_frames, step))
                except Exception as e:
                    print(f"âš ï¸ Decord Init Failed: {e}. Fallback to OpenCV.")
                    use_decord = False

            if not use_decord:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = [] # Generate on fly
            
            # Batch Buffers
            batch_imgs = []
            batch_idxs = []
            batch_thumbs = []
            
            # Helper to wrap processing
            def process_wrapper(args):
                frame, idx = args
                return self._process_single_frame(frame, idx)

            # Decord Processing Loop (Parallelized)
            if use_decord:
                chunk_size = self.batch_size
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    for i in range(0, len(indices), chunk_size):
                        if self.stop_event.is_set(): 
                            break
                        
                        chunk_indices = indices[i : i + chunk_size]
                        try:
                            frames = vr.get_batch(chunk_indices).asnumpy() # (N, H, W, C) RGB
                        except Exception as e:
                             print(f"Decord Batch Error: {e}")
                             continue

                        # Parallel Process
                        tasks = zip(frames, chunk_indices)
                        results = list(executor.map(process_wrapper, tasks))
                        
                        # Collect valid results
                        for res in results:
                            if res:
                                img, idx, thumb = res
                                batch_imgs.append(img)
                                batch_idxs.append(idx)
                                batch_thumbs.append(thumb)
                        
                        if len(batch_imgs) >= self.batch_size:
                            self._push_batch(batch_imgs, batch_idxs, batch_thumbs)
                            batch_imgs = []
                            batch_idxs = []
                            batch_thumbs = []
            
            else: 
                # OpenCV Fallback (Sequential read, Parallel process chunk)
                count = 0
                frame_buffer = []
                idx_buffer = []
                
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                   while cap.isOpened() and not self.stop_event.is_set():
                        ret, frame = cap.read()
                        if not ret: 
                            break
                        
                        if count % step == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_buffer.append(frame_rgb)
                            idx_buffer.append(count)
                            
                            # Process when buffer full
                            if len(frame_buffer) >= self.batch_size:
                                tasks = zip(frame_buffer, idx_buffer)
                                results = list(executor.map(process_wrapper, tasks))
                                
                                for res in results:
                                    if res:
                                        img, idx, thumb = res
                                        batch_imgs.append(img)
                                        batch_idxs.append(idx)
                                        batch_thumbs.append(thumb)
                                
                                self._push_batch(batch_imgs, batch_idxs, batch_thumbs)
                                batch_imgs = []
                                batch_idxs = []
                                batch_thumbs = []
                                frame_buffer = []
                                idx_buffer = []

                        count += 1
                   cap.release()
                   
                   # Flush OpenCV Buffer
                   if frame_buffer:
                        tasks = zip(frame_buffer, idx_buffer)
                        results = list(executor.map(process_wrapper, tasks))
                        for res in results:
                            if res:
                                img, idx, thumb = res
                                batch_imgs.append(img)
                                batch_idxs.append(idx)
                                batch_thumbs.append(thumb)

            # Flush remaining batch
            if batch_imgs:
                self._push_batch(batch_imgs, batch_idxs, batch_thumbs)
                
        except Exception as e:
            print(f"âŒ Producer Error: {e}")
            traceback.print_exc()
        finally:
            self.batch_queue.put(None) # Signal End

    def _process_single_frame(self, image_rgb, idx):
        """Helper to face-detect and transform. Returns (processed, idx, thumb)"""
        try:
            # Face Detection Optimization
            face_crop = None
            if self.face_cascade:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                
                # User Request: Keep original quality (No downscaling)
                # This will be slower (CPU heavy) but ensures maximum detection quality
                scale_factor = 1.0 
                small_gray = gray 
                
                try:
                    # Detect on full resolution
                    faces = self.face_cascade.detectMultiScale(
                        small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )
                except:
                    faces = []
                
                if len(faces) > 0:
                    # Find largest
                    largest = max(faces, key=lambda r: r[2] * r[3])
                    sx, sy, sw, sh = largest
                    
                    # No mapping needed since scale_factor is 1.0
                    x = sx
                    y = sy
                    w = sw 
                    h = sh
                    
                    margin = int(max(w, h) * 0.2)
                    x_s, y_s = max(x-margin, 0), max(y-margin, 0)
                    x_e, y_e = min(x+w+margin, image_rgb.shape[1]), min(y+h+margin, image_rgb.shape[0])
                    face_crop = image_rgb[y_s:y_e, x_s:x_e]
            
            input_img = face_crop if face_crop is not None else image_rgb
            
            # Transform
            augmented = self.transform(image=input_img)
            processed = augmented['image'] # Tensor (C, H, W)
            
            # Thumbnail
            thumb = cv2.resize(image_rgb, (160, 90))
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            thumb_b64 = base64.b64encode(buf).decode('utf-8')

            return processed, idx, thumb_b64
                
        except Exception as e:
            print(f"Frame Processing Error: {e}")
            return None

    def _push_batch(self, b_imgs, b_idxs, b_thumbs):
        # Stack
        if not b_imgs: return
        
        # Convert list of Tensors to Stacked Tensor
        batch_tensor = torch.stack(b_imgs) # (B, C, H, W)
        
        # === GPU MEMORY PINNING ===
        # If using PyTorch (CUDA), pin memory before pushing to queue.
        # This allows non-blocking transfer to GPU in the consumer thread.
        # Note: If batch_tensor is already on CPU, pin_memory() returns a copy in pinned memory.
        if torch.cuda.is_available():
             batch_tensor = batch_tensor.pin_memory()
             
        # For ONNX, we usually pass numpy. 
        batch_data = batch_tensor

        self.batch_queue.put({
            'data': batch_data,
            'indices': b_idxs, 
            'thumbnails': b_thumbs
        })

    def run(self, video_path, frames_per_second=5):
        """
        Main execution entry point.
        """
        print(f"ðŸŽ¬ Starting Pipeline for {os.path.basename(video_path)}")
        self._emit_progress("Starting video pipeline", 5)
        
        # 1. Video Info
        if DECORD_AVAILABLE:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
        else:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
        if fps <= 0: fps = 30
        duration = total_frames / fps
        step = int(fps / frames_per_second)
        if step < 1: step = 1

        self._emit_progress(f"Decoder ready ({'Decord' if DECORD_AVAILABLE else 'OpenCV'})", 15)
        
        print(f"   Using { 'Decord' if DECORD_AVAILABLE else 'OpenCV' } decoder.")
        print(f"   Mode: PyTorch (Optimized)")
        print(f"   Batch Size: {self.batch_size}, Sampling Step: {step} ({frames_per_second} fps)")

        # 2. Start Producer
        t_prod = Thread(target=self._producer_worker, args=(video_path, step))
        t_prod.start()
        self._emit_progress("Frame extraction started", 25)
        
        # 3. Consumer Inference Loop (Main Thread)
        probs = []
        frame_indices = []
        suspicious_frames = []
        processed_count = 0
        
        t0 = time.time()
        
        try:
            while True:
                # Get Batch
                item = self.batch_queue.get()
                if item is None: break
                
                batch_data = item['data']
                b_idxs = item['indices']
                b_thumbs = item['thumbnails']
                current_bs = len(b_idxs)
                
                # Inference
                # PyTorch
                with torch.no_grad():
                    # Transfer to GPU (Async if pinned)
                    if self.device.type == 'cuda':
                        input_tensor = batch_data.to(self.device, non_blocking=True)
                    else:
                        input_tensor = batch_data.to(self.device)
                        
                    logits = self.model(input_tensor)
                    batch_probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()

                # Process Results
                if len(batch_probs) != current_bs:
                    print(f"âŒ CRITICAL MISMATCH: Batch input {current_bs}, Output {len(batch_probs)}")
                    print(f"   Logits shape: {logits.shape if hasattr(logits, 'shape') else 'unknown'}")

                for i in range(current_bs):
                    prob = batch_probs[i]
                    idx = b_idxs[i]
                    thumb = b_thumbs[i]
                    
                    probs.append(prob)
                    frame_indices.append({"index": idx, "thumbnail": thumb})
                    
                    if prob > 0.5:
                        suspicious_frames.append({
                            "timestamp": round(idx / fps, 2),
                            "frame_index": idx,
                            "fake_prob": round(prob, 4),
                            "thumbnail": thumb
                        })
                
                processed_count += current_bs
                self.batch_queue.task_done()

                if total_frames > 0:
                    percent = min(90, 25 + int((processed_count / max(1, total_frames / step)) * 60))
                    self._emit_progress(f"Processed {processed_count} sampled frames", percent)
                
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            t_prod.join()
            
        dt = time.time() - t0
        print(f"âœ… Finished in {dt:.2f}s ({processed_count / dt:.1f} fps processed)")
        self._emit_progress("Aggregating model outputs", 94)

        # 4. Aggregation
        if processed_count == 0:
            return {"error": "No frames processed"}

        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)
        fake_frame_count = len([p for p in probs if p > 0.6])
        fake_ratio = fake_frame_count / processed_count
        
        # Verdict Logic (tri-state)
        cond1 = avg_prob > 0.65
        cond2 = fake_ratio > 0.15 and max_prob > 0.7
        cond3 = max_prob > 0.95
        cond_suspicious = avg_prob > 0.45 or fake_ratio > 0.08 or max_prob > 0.8

        if cond1 or cond2 or cond3:
            verdict = "FAKE"
            confidence = max(max_prob, 0.6)
        elif cond_suspicious:
            verdict = "SUSPICIOUS"
            confidence = max(avg_prob, max_prob * 0.8)
        else:
            verdict = "REAL"
            confidence = 1 - avg_prob

        suspicious_count = len([p for p in probs if p > 0.5])
        variance_score = _variance_proxy(probs)
        summary, reasons, confidence_explanation = _build_video_explanation(
            verdict,
            _clamp01(avg_prob),
            _clamp01(max_prob),
            _clamp01(fake_ratio),
            processed_count,
            suspicious_count,
            variance_score
        )

        self._emit_progress("Video inference complete", 100)
        
        return {
            "type": "video",
            "prediction": verdict,
            "confidence": _clamp01(confidence),
            "avg_fake_prob": float(avg_prob),
            "max_fake_prob": float(max_prob),
            "fake_frame_ratio": float(fake_ratio),
            "processed_frames": processed_count,
            "duration": float(duration),
            "detection_explanation_summary": summary,
            "reasons": reasons,
            "confidence_explanation": confidence_explanation,
            "explanation_metrics": {
                "avg_fake_prob": float(avg_prob),
                "max_fake_prob": float(max_prob),
                "fake_frame_ratio": float(fake_ratio),
                "processed_frames": int(processed_count),
                "suspicious_count": int(suspicious_count),
                "variance_score": float(variance_score)
            },
            "processing_details": {
                "decoder": "Decord" if DECORD_AVAILABLE else "OpenCV",
                "inference_mode": "PyTorch (Optimized)",
                "batch_size": int(self.batch_size),
                "sampling_fps": int(frames_per_second),
                "sampling_step": int(step),
                "source_fps": float(fps),
                "processed_fps": float(processed_count / dt) if dt > 0 else 0.0,
                "pipeline_seconds": float(dt)
            },
            "timeline": [
                {
                    "time": round(item["index"] / fps, 2), 
                    "prob": round(p, 3),
                    "thumbnail": item["thumbnail"]
                } 
                for item, p in zip(frame_indices, probs)
            ],
            "suspicious_frames": suspicious_frames[:10]
        }

# Wrapper function for backward compatibility
def process_video(video_path, model, transform, device, frames_per_second=1, batch_size=16, progress_callback=None):
    pipeline = VideoPipeline(model, transform, device, batch_size=batch_size, progress_callback=progress_callback)
    return pipeline.run(video_path, frames_per_second)



