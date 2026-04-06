// Video Result Page - Enhanced Functionality
// ==========================================

let videoPlayer;
let currentResult = null;
let chart = null;

function getVerdictConfig(prediction, mediaType) {
    const noun = mediaType === 'image' ? 'IMAGE' : 'VIDEO';

    if (prediction === 'FAKE') {
        return {
            title: `FAKE ${noun} DETECTED`,
            color: '#ff3333',
            icon: 'fa-exclamation-triangle',
            meterClass: 'fill-fake',
            panelBg: 'rgba(255, 51, 51, 0.1)',
            panelTitle: 'Manipulation Detected',
            summaryFallback: 'Strong anomaly evidence triggered a FAKE classification.'
        };
    }

    if (prediction === 'SUSPICIOUS') {
        return {
            title: `SUSPICIOUS ${noun}`,
            color: '#f59e0b',
            icon: 'fa-question-circle',
            meterClass: 'fill-suspicious',
            panelBg: 'rgba(245, 158, 11, 0.12)',
            panelTitle: 'Suspicious Signals Detected',
            summaryFallback: 'Mixed anomaly evidence triggered a SUSPICIOUS classification.'
        };
    }

    return {
        title: `AUTHENTIC ${noun}`,
        color: '#10B981',
        icon: 'fa-check-circle',
        meterClass: 'fill-real',
        panelBg: 'rgba(16, 185, 129, 0.1)',
        panelTitle: 'Authentic Signals Detected',
        summaryFallback: 'Low anomaly evidence supports a REAL classification.'
    };
}

function toBulletList(items, fallback) {
    const safeItems = Array.isArray(items) && items.length ? items : [fallback];
    return safeItems.map((item) => `<li>${item}</li>`).join('');
}

function formatBackendDetails(details) {
    if (!details || typeof details !== 'object') {
        return '<li>Backend process details were not provided.</li>';
    }

    const entries = Object.entries(details)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => {
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
            const display = typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(3)) : value;
            return `<li><strong>${label}:</strong> ${display}</li>`;
        });

    return entries.length ? entries.join('') : '<li>Backend process details were not provided.</li>';
}

function inferMediaType(result) {
    if (result?.media_type) return result.media_type;
    if (result?.image_path && /\.(png|jpe?g|webp|gif)$/i.test(result.image_path)) return 'image';
    if (result?.video_url) return 'video';
    if (result?.timeline && result.timeline.length > 0) return 'video';
    return 'image';
}

function updatePageContext(mediaType) {
    const headerTitle = document.querySelector('.results-header h1');
    const headerSubtitle = document.querySelector('.results-header p');

    if (headerTitle) {
        headerTitle.textContent = mediaType === 'image' ? 'Image Analysis Report' : 'Video Analysis Report';
    }

    if (headerSubtitle) {
        headerSubtitle.textContent = mediaType === 'image'
            ? 'Comprehensive AI-powered deepfake detection analysis with explainable image evidence'
            : 'Comprehensive AI-powered deepfake detection analysis with frame-by-frame insights';
    }

    const timelineTitle = document.querySelector('.timeline-header h3');
    if (timelineTitle && mediaType === 'image') {
        timelineTitle.innerHTML = '<i class="fas fa-image"></i> Image Confidence Snapshot';
    }
}

function buildPipelineSteps(result, mediaType) {
    const details = result?.processing_details || {};

    if (mediaType === 'image') {
        return [
            `Input accepted as image and normalized to ${details.input_size || 'model'} resolution.`,
            `Inference executed in ${details.inference_mode || 'PyTorch'} on ${details.device || 'configured device'}.`,
            'Heatmap and confidence signals were computed for explainable output.'
        ];
    }

    return [
        `Video decoded using ${details.decoder || 'configured decoder'} at sampling ${details.sampling_fps || 'N/A'} fps.`,
        `Frames were batched (${details.batch_size || 'N/A'} per step) and passed through ${details.inference_mode || 'model inference'}.`,
        `Timeline and suspicious frame metrics were aggregated in ${typeof details.pipeline_seconds === 'number' ? details.pipeline_seconds.toFixed(2) : 'N/A'} seconds.`
    ];
}

function getBorderlineRealWarning(result, mediaType, prediction) {
    if (prediction !== 'REAL') {
        return null;
    }

    const imageFakeProb = Number.isFinite(result.fake_probability) ? result.fake_probability : null;
    const videoAvgFakeProb = Number.isFinite(result.avg_fake_prob) ? result.avg_fake_prob : null;

    // Strict image thresholds: REAL when fake_prob <= 0.15, so warn if >= 0.12 (close to boundary)
    if (mediaType === 'image' && imageFakeProb !== null && imageFakeProb >= 0.12) {
        return `Borderline real classification: fake probability is ${(imageFakeProb * 100).toFixed(1)}%, which is close to the suspicious threshold (0.15).`;
    }

    if (mediaType === 'video' && videoAvgFakeProb !== null && videoAvgFakeProb >= 0.34) {
        return `Borderline real classification: average fake probability is ${(videoAvgFakeProb * 100).toFixed(1)}%, near suspicious range.`;
    }

    return null;
}

document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    videoPlayer = document.getElementById('videoPlayer');

    // Retrieve results from localStorage
    const resultData = localStorage.getItem('video_analysis_result');

    if (!resultData) {
        currentResult = {
            prediction: 'REAL',
            confidence: 0,
            duration: 0,
            processed_frames: 0,
            suspicious_frames: [],
            avg_fake_prob: 0,
            fake_frame_ratio: 0,
            timeline: []
        };

        initializeVideoPlayer();
        setupVideoControls();

        const verdictTitle = document.getElementById('verdictTitle');
        const confidenceValue = document.getElementById('confidenceValue');
        if (verdictTitle) verdictTitle.textContent = 'NO ANALYSIS LOADED';
        if (confidenceValue) confidenceValue.textContent = '0% Confidence';

        const notes = document.getElementById('analysisNotes');
        if (notes) {
            notes.innerHTML = '<p>No saved analysis is loaded yet. Run a video scan to populate the report.</p>';
        }

        const pipelinePanel = document.getElementById('analysisPipelineContent');
        if (pipelinePanel) {
            pipelinePanel.innerHTML = '<p>No pipeline data available yet.</p>';
        }

        const backendPanel = document.getElementById('backendProcessContent');
        if (backendPanel) {
            backendPanel.innerHTML = '<p>No backend process data available yet.</p>';
        }

        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
        }

        return;
    }

    currentResult = JSON.parse(resultData);

    // Initialize everything
    initializeVideoPlayer();
    populateUI(currentResult);
    setupVideoControls();
    setupDownloadButton();

    // Hide loading overlay
    setTimeout(() => {
        document.getElementById('loadingOverlay').classList.add('hidden');
    }, 500);
});

// ==========================================
// VIDEO PLAYER INITIALIZATION
// ==========================================

function initializeVideoPlayer() {
    const videoWrapper = document.querySelector('.video-wrapper');
    const controls = document.querySelector('.video-controls');

    const setImageSource = (src) => {
        if (videoPlayer) videoPlayer.style.display = 'none';
        if (controls) controls.style.display = 'none';

        let mediaImage = document.getElementById('mediaImage');
        if (!mediaImage && videoWrapper) {
            mediaImage = document.createElement('img');
            mediaImage.id = 'mediaImage';
            mediaImage.alt = 'Analyzed media';
            mediaImage.style.cssText = 'width:100%;height:100%;display:block;object-fit:contain;background:#000;';
            videoWrapper.appendChild(mediaImage);
        }
        if (mediaImage) mediaImage.src = src;
    };

    const setVideoSource = (src) => {
        if (videoPlayer) videoPlayer.style.display = 'block';
        if (controls) controls.style.display = 'block';

        const mediaImage = document.getElementById('mediaImage');
        if (mediaImage) mediaImage.remove();
        videoPlayer.src = src;
    };

    const applySource = (rawSrc) => {
        if (!rawSrc) return false;

        let src = rawSrc;
        if (!src.startsWith('http') && !src.startsWith('/')) {
            src = '/' + src;
        }

        const mediaType = currentResult?.media_type || 'video';
        const isImageSource = mediaType === 'image' || /\.(png|jpe?g|webp|gif)$/i.test(src);

        if (isImageSource) {
            setImageSource(src);
        } else {
            setVideoSource(src);
        }

        return true;
    };

    const imageFallback = currentResult && currentResult.heatmap
        ? (currentResult.heatmap.startsWith('data:') ? currentResult.heatmap : `data:image/jpeg;base64,${currentResult.heatmap}`)
        : null;

    if (currentResult && applySource(currentResult.video_url || currentResult.image_path || imageFallback)) {
        // Source applied.
    } else {
        console.warn('No video source available in result data');
        if (videoWrapper) {
            const placeholder = document.createElement('div');
            placeholder.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: rgba(255, 255, 255, 0.6);
                z-index: 5;
            `;
            placeholder.innerHTML = `
                <i class="fas fa-video-slash" style="font-size: 4rem; margin-bottom: 1rem; display: block; color: rgba(227, 245, 20, 0.3);"></i>
                <p style="font-size: 1.1rem;">Media source not available</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">The uploaded file could not be loaded</p>
            `;
            videoWrapper.appendChild(placeholder);
        }
    }

    // Video event listeners
    if (videoPlayer) {
        videoPlayer.addEventListener('loadedmetadata', () => {
            updateDuration();
        });

        videoPlayer.addEventListener('timeupdate', () => {
            updateProgress();
            updateTimeDisplay();
        });

        videoPlayer.addEventListener('ended', () => {
            const playPauseIcon = document.querySelector('#playPauseBtn i');
            if (playPauseIcon) {
                playPauseIcon.className = 'fas fa-redo';
            }
        });

        // Error handling
        videoPlayer.addEventListener('error', (e) => {
            console.error('Video load error:', e);
            console.error('Failed source:', videoPlayer.src);
            if (videoWrapper) {
                const errorMsg = document.createElement('div');
                errorMsg.style.cssText = `
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    text-align: center;
                    color: rgba(255, 51, 51, 0.8);
                    z-index: 5;
                    background: rgba(0,0,0,0.7);
                    padding: 20px;
                    border-radius: 10px;
                `;
                errorMsg.innerHTML = `
                    <i class="fas fa-exclamation-triangle" style="font-size: 4rem; margin-bottom: 1rem; display: block;"></i>
                    <p style="font-size: 1.1rem;">Failed to load video</p>
                    <p style="font-size: 0.9rem; margin-top: 0.5rem;">The video format may not be supported</p>
                    <p style="font-size: 0.8rem; color: #aaa; margin-top: 10px; word-break: break-all;">Source: ${videoPlayer.src}</p>
                `;
                videoWrapper.appendChild(errorMsg);
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case ' ':
                e.preventDefault();
                togglePlay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
                break;
            case 'ArrowRight':
                e.preventDefault();
                videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + 5);
                break;
            case 'f':
                toggleFullscreen();
                break;
            case 'm':
                toggleMute();
                break;
        }
    });
}

// ==========================================
// VIDEO CONTROLS
// ==========================================

function setupVideoControls() {
    // Play/Pause
    const playPauseBtn = document.getElementById('playPauseBtn');
    playPauseBtn.addEventListener('click', togglePlay);

    // Progress bar
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.addEventListener('click', seek);

    // Frame navigation
    document.getElementById('prevFrameBtn').addEventListener('click', () => {
        videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - (1 / 30)); // Assuming 30fps
    });

    document.getElementById('nextFrameBtn').addEventListener('click', () => {
        videoPlayer.currentTime = Math.min(videoPlayer.duration, videoPlayer.currentTime + (1 / 30));
    });

    // Volume
    const volumeSlider = document.getElementById('volumeSlider');
    const muteBtn = document.getElementById('muteBtn');

    volumeSlider.addEventListener('input', (e) => {
        const volume = e.target.value / 100;
        videoPlayer.volume = volume;
        updateVolumeIcon(volume);
    });

    muteBtn.addEventListener('click', toggleMute);

    // Speed control
    const speedBtn = document.getElementById('speedBtn');
    const speedMenu = document.getElementById('speedMenu');

    speedBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        speedMenu.classList.toggle('active');
    });

    document.addEventListener('click', () => {
        speedMenu.classList.remove('active');
    });

    document.querySelectorAll('.speed-option').forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const speed = parseFloat(e.target.dataset.speed);
            videoPlayer.playbackRate = speed;

            document.querySelectorAll('.speed-option').forEach(opt => opt.classList.remove('active'));
            e.target.classList.add('active');
            speedMenu.classList.remove('active');
        });
    });

    // Fullscreen
    document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);

    // Video click to play/pause
    videoPlayer.addEventListener('click', togglePlay);
}

function togglePlay() {
    if (videoPlayer.paused) {
        videoPlayer.play();
        document.querySelector('#playPauseBtn i').className = 'fas fa-pause';
    } else {
        videoPlayer.pause();
        document.querySelector('#playPauseBtn i').className = 'fas fa-play';
    }
}

function seek(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    videoPlayer.currentTime = percent * videoPlayer.duration;
}

function toggleMute() {
    videoPlayer.muted = !videoPlayer.muted;
    updateVolumeIcon(videoPlayer.muted ? 0 : videoPlayer.volume);
}

function updateVolumeIcon(volume) {
    const muteBtn = document.querySelector('#muteBtn i');
    if (volume === 0) {
        muteBtn.className = 'fas fa-volume-mute';
    } else if (volume < 0.5) {
        muteBtn.className = 'fas fa-volume-down';
    } else {
        muteBtn.className = 'fas fa-volume-up';
    }
}

function toggleFullscreen() {
    const container = document.querySelector('.video-player-container');

    if (!document.fullscreenElement) {
        container.requestFullscreen().catch(err => {
            console.error('Fullscreen error:', err);
        });
        document.querySelector('#fullscreenBtn i').className = 'fas fa-compress';
    } else {
        document.exitFullscreen();
        document.querySelector('#fullscreenBtn i').className = 'fas fa-expand';
    }
}

function updateProgress() {
    const percent = (videoPlayer.currentTime / videoPlayer.duration) * 100;
    document.getElementById('progressBar').style.width = percent + '%';
}

function updateTimeDisplay() {
    document.getElementById('currentTime').textContent = formatTime(videoPlayer.currentTime);
}

function updateDuration() {
    document.getElementById('duration').textContent = formatTime(videoPlayer.duration);
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ==========================================
// UI POPULATION
// ==========================================

function populateUI(result) {
    const mediaType = inferMediaType(result);
    const prediction = result.prediction || 'REAL';
    const verdict = getVerdictConfig(prediction, mediaType);
    const suspiciousFrames = Array.isArray(result.suspicious_frames) ? result.suspicious_frames : [];
    const duration = Number.isFinite(result.duration) ? result.duration : 0;
    const processedFrames = Number.isFinite(result.processed_frames) ? result.processed_frames : 1;
    const avgFakeProb = Number.isFinite(result.avg_fake_prob)
        ? result.avg_fake_prob
        : Number.isFinite(result.fake_probability)
        ? result.fake_probability
        : 0;

    updatePageContext(mediaType);

    // Verdict
    const title = document.getElementById('verdictTitle');
    const bar = document.getElementById('confidenceBar');
    const val = document.getElementById('confidenceValue');
    const verdictInfo = document.querySelector('.verdict-info');

    title.textContent = verdict.title;
    title.style.color = verdict.color;

    const conf = (result.confidence * 100).toFixed(1);

    // Animate confidence bar
    setTimeout(() => {
        bar.style.width = `${conf}%`;
    }, 100);

    bar.className = `meter-fill ${verdict.meterClass}`;
    val.textContent = `${conf}% Confidence`;

    // Show caution when REAL is close to suspicious threshold.
    const existingWarning = document.getElementById('confidenceWarningBadge');
    if (existingWarning) {
        existingWarning.remove();
    }

    const warningText = getBorderlineRealWarning(result, mediaType, prediction);
    if (warningText && verdictInfo) {
        const warning = document.createElement('div');
        warning.id = 'confidenceWarningBadge';
        warning.style.cssText = 'margin-top:10px;padding:8px 10px;border-radius:10px;border:1px solid rgba(245,158,11,0.45);background:rgba(245,158,11,0.14);color:#fbbf24;font-size:0.82rem;line-height:1.4;text-align:left;';
        warning.innerHTML = '<strong style="display:block;margin-bottom:2px;">Confidence Warning</strong>' + warningText;
        verdictInfo.appendChild(warning);
    }

    // Update verdict icon
    const verdictIcon = document.querySelector('.verdict-icon');
    verdictIcon.innerHTML = `<i class="fas ${verdict.icon}"></i>`;
    verdictIcon.style.background = `linear-gradient(135deg, ${verdict.color} 0%, ${verdict.color}cc 100%)`;

    // Stats with animation
    animateValue('videoDuration', 0, duration, `${duration.toFixed(1)}s`, 1000);
    animateValue('framesProcessed', 0, processedFrames, processedFrames, 1000);
    animateValue('suspiciousCount', 0, suspiciousFrames.length, suspiciousFrames.length, 1000);
    animateValue('avgProb', 0, avgFakeProb * 100, `${(avgFakeProb * 100).toFixed(1)}%`, 1000);

    // Notes
    const notes = document.getElementById('analysisNotes');
    const summary = result.detection_explanation_summary || verdict.summaryFallback;
    const reasonsList = toBulletList(result.reasons, 'No anomalies were reported by the current checks.');
    const confidenceList = toBulletList(
        result.confidence_explanation,
        `Based on ${processedFrames} frames analyzed with average fake probability ${(avgFakeProb * 100).toFixed(1)}%.`
    );
    const backendProcessList = formatBackendDetails(result.processing_details);
    const pipelineSteps = toBulletList(buildPipelineSteps(result, mediaType), 'Pipeline summary is not available.');

    const pipelinePanel = document.getElementById('analysisPipelineContent');
    if (pipelinePanel) {
        pipelinePanel.innerHTML = `<ul style="margin-left: 1.25rem; line-height: 1.8;">${pipelineSteps}</ul>`;
    }

    const backendPanel = document.getElementById('backendProcessContent');
    if (backendPanel) {
        backendPanel.innerHTML = `<ul style="margin-left: 1.25rem; line-height: 1.8;">${backendProcessList}</ul>`;
    }

    notes.innerHTML = `
        <div style="display: flex; align-items: start; gap: 12px; padding: 1rem; background: ${verdict.panelBg}; border-left: 3px solid ${verdict.color}; border-radius: 8px; margin-bottom: 1rem;">
            <i class="fas ${verdict.icon}" style="color: ${verdict.color}; font-size: 24px; margin-top: 2px;"></i>
            <div>
                <strong style="color: ${verdict.color}; font-size: 1.05rem;">${verdict.panelTitle}</strong><br>
                <span style="margin-top: 8px; display: block;">${summary}</span>
            </div>
        </div>
        <p><strong>Detection Explanation:</strong></p>
        <ul style="margin-left: 1.5rem; margin-top: 0.5rem; line-height: 1.8;">${reasonsList}</ul>
        <p style="margin-top: 1rem;"><strong>Confidence Explanation:</strong></p>
        <ul style="margin-left: 1.5rem; margin-top: 0.5rem; line-height: 1.8;">${confidenceList}</ul>
    `;

    // Chart
    if (Array.isArray(result.timeline) && result.timeline.length > 0) {
        renderChart(result.timeline);
        addFrameMarkers(result.timeline, duration || 1);
    } else {
        const chartSection = document.querySelector('.timeline-container');
        if (chartSection) chartSection.style.display = mediaType === 'image' ? 'none' : 'block';
    }

    // Frame Grid
    if (Array.isArray(result.timeline) && result.timeline.length > 0) {
        renderFrameGrid(result.timeline, duration || 1);
    } else {
        const framesSection = document.querySelector('.dashboard-frames');
        if (framesSection) framesSection.style.display = 'none';
    }
}

// ==========================================
// CHART RENDERING
// ==========================================

function renderChart(timeline) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    const times = timeline.map(t => formatTime(t.time));
    const probs = timeline.map(t => t.prob);

    // Destroy existing chart if any
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Fake Probability',
                data: probs,
                borderColor: '#E3F514',
                backgroundColor: 'rgba(227, 245, 20, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 8,
                pointBackgroundColor: '#E3F514',
                pointBorderColor: '#000',
                pointBorderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const time = timeline[index].time;
                    videoPlayer.currentTime = time;
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888',
                        callback: function (value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Fake Probability',
                        color: '#888'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255,255,255,0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888',
                        maxTicksLimit: 10
                    },
                    title: {
                        display: true,
                        text: 'Time',
                        color: '#888'
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#E3F514',
                    bodyColor: '#fff',
                    borderColor: 'rgba(227, 245, 20, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        title: (context) => `Time: ${context[0].label}`,
                        label: (ctx) => `Probability: ${(ctx.raw * 100).toFixed(1)}%`,
                        afterLabel: (ctx) => {
                            const prob = ctx.raw;
                            if (prob > 0.5) {
                                return 'Status: Suspicious âš ï¸';
                            } else {
                                return 'Status: Clean âœ“';
                            }
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// ==========================================
// FRAME MARKERS
// ==========================================

function addFrameMarkers(timeline, duration) {
    const progressContainer = document.getElementById('progressContainer');

    timeline.forEach(frame => {
        if (frame.prob > 0.5) { // Suspicious frames
            const marker = document.createElement('div');
            marker.className = 'frame-marker';
            marker.style.left = ((frame.time / duration) * 100) + '%';
            marker.title = `Suspicious frame at ${formatTime(frame.time)}`;

            marker.addEventListener('click', (e) => {
                e.stopPropagation();
                videoPlayer.currentTime = frame.time;
            });

            progressContainer.appendChild(marker);
        }
    });
}

// ==========================================
// FRAME GRID
// ==========================================

function renderFrameGrid(timeline, duration) {
    const frameGrid = document.getElementById('frameGrid');
    frameGrid.innerHTML = '';

    timeline.forEach((frame, index) => {
        const frameItem = document.createElement('div');
        frameItem.className = 'frame-item' + (frame.prob > 0.5 ? ' suspicious' : '');

        const thumbContent = frame.thumbnail
            ? `<img src="data:image/jpeg;base64,${frame.thumbnail}" style="width: 100%; height: 100%; object-fit: cover;">`
            : `<i class="fas fa-film" style="font-size: 2rem;"></i>`;

        frameItem.innerHTML = `
            <div class="frame-thumbnail" style="display: flex; align-items: center; justify-content: center; color: #666; font-size: 0.9rem; overflow: hidden; background: #000;">
                ${thumbContent}
            </div>
            <div class="frame-badge ${frame.prob > 0.5 ? 'suspicious' : 'clean'}">
                ${frame.prob > 0.5 ? 'âš ï¸ ' + (frame.prob * 100).toFixed(0) + '%' : 'âœ“ ' + ((1 - frame.prob) * 100).toFixed(0) + '%'}
            </div>
            <div class="frame-info">
                <span class="frame-time">${formatTime(frame.time)}</span>
                <span class="frame-confidence">${(frame.prob * 100).toFixed(1)}%</span>
            </div>
        `;

        frameItem.addEventListener('click', () => {
            videoPlayer.currentTime = frame.time;
            videoPlayer.play();
        });

        frameGrid.appendChild(frameItem);
    });
}

// ==========================================
// ANIMATIONS
// ==========================================

function animateValue(id, start, end, suffix, duration) {
    const element = document.getElementById(id);
    const startTime = performance.now();
    const isPercentage = typeof suffix === 'string' && suffix.includes('%');

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeOut;

        if (isPercentage) {
            element.textContent = suffix;
        } else if (typeof suffix === 'string') {
            element.textContent = suffix;
        } else {
            element.textContent = Math.floor(current);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = suffix;
        }
    }

    requestAnimationFrame(update);
}

// ==========================================
// DOWNLOAD REPORT
// ==========================================

function setupDownloadButton() {
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (!downloadBtn) return;

    downloadBtn.addEventListener('click', async () => {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
        downloadBtn.disabled = true;

        try {
            await generatePDFReport();
            downloadBtn.innerHTML = '<i class="fas fa-check"></i> Report Downloaded!';

            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
                downloadBtn.disabled = false;
            }, 3000);
        } catch (error) {
            console.error('Error generating report:', error);
            downloadBtn.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';

            alert(`Failed to generate report: ${error.message || error}\nPlease check console for details.`);

            setTimeout(() => {
                downloadBtn.innerHTML = originalText;
                downloadBtn.disabled = false;
            }, 3000);
        }
    });
}

async function generatePDFReport() {
    const reportContainer = document.getElementById('reportContainer');
    if (!reportContainer) {
        throw new Error('Report container not found');
    }
    const prediction = currentResult.prediction || 'REAL';
    const verdict = getVerdictConfig(prediction);
    const isFake = prediction === 'FAKE';
    const accentColor = verdict.color;

    // 1. Construct Report HTML
    const dateStr = new Date().toLocaleDateString('en-US', {
        year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
    });

    // Get frames for report (max 8)
    // Prioritize suspicious frames if fake, otherwise spread out frames
    let reportFrames = [];
    const suspiciousFrames = Array.isArray(currentResult.suspicious_frames) ? currentResult.suspicious_frames : [];

    if (isFake && suspiciousFrames.length > 0) {
        // Take up to 8 suspicious frames
        reportFrames = suspiciousFrames
            .slice(0, 8)
            .map(frame => {
                if (frame && frame.thumbnail) {
                    return {
                        time: frame.timestamp || frame.time || 0,
                        prob: frame.fake_prob || frame.prob || 0,
                        thumbnail: frame.thumbnail
                    };
                }

                if (typeof frame === 'number' && currentResult.timeline && currentResult.timeline[frame]) {
                    return currentResult.timeline[frame];
                }

                return null;
            })
            .filter(frame => frame !== null);
    } else if (currentResult.timeline && currentResult.timeline.length > 0) {
        // Take up to 8 evenly spaced frames
        const step = Math.max(1, Math.floor(currentResult.timeline.length / 8));
        for (let i = 0; i < currentResult.timeline.length && reportFrames.length < 8; i += step) {
            if (currentResult.timeline[i]) {
                reportFrames.push(currentResult.timeline[i]);
            }
        }
    }

    const framesHTML = reportFrames.map(frame => {
        if (!frame) return '';
        return `
        <div class="report-frame-item">
            ${frame.thumbnail ? `<img src="data:image/jpeg;base64,${frame.thumbnail}">` : ''}
            <div class="report-frame-badge" style="background: ${frame.prob > 0.5 ? '#ff3333' : '#10B981'}">
                ${(frame.prob * 100).toFixed(0)}%
            </div>
        </div>
    `}).join('');

    // Capture chart as image
    const chartCanvas = document.getElementById('timelineChart');
    const chartImg = chartCanvas ? chartCanvas.toDataURL('image/png') : null;

    // Capture video preview (thumbnail of current frame)
    // We can use the first frame of the timeline if available, or just a placeholder if video element is cross-origin restricted
    // Ideally, we'd capture the video element, but that's often blocked by CORS or returns black.
    // Let's use the thumbnail of the most significant frame from the timeline as the 'video preview'
    // Find the first frame with a valid thumbnail in reportFrames, or fallback to any frame in timeline
    let mainPreviewThumb = '';

    // Helper to check for thumbnail
    const hasThumb = (f) => f && f.thumbnail;

    // 1. Try report frames (suspicious/key frames)
    const timelineItems = Array.isArray(currentResult.timeline) ? currentResult.timeline : [];
    const previewFrame = reportFrames.find(hasThumb) || timelineItems.find(hasThumb);

    if (previewFrame) {
        mainPreviewThumb = `data:image/jpeg;base64,${previewFrame.thumbnail}`;
    }

    reportContainer.innerHTML = `
        <div class="report-header">
            <div class="report-logo">
                <div style="width:36px;height:36px;border-radius:10px;background:#22d3ee;color:#04101a;display:grid;place-items:center;font-weight:900;font-size:0.85rem;">DS</div>
                <span>Deepfake Detection System Analysis Report</span>
            </div>
            <div class="report-meta">
                <p>Generated on: ${dateStr}</p>
                <p>Ref ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 2fr 3fr; gap: 30px; margin-bottom: 30px;">
            <!-- Video Preview Section -->
            <div class="report-card" style="margin: 0; padding: 10px; display: flex; align-items: center; justify-content: center; background: #000; overflow: hidden; height: 200px;">
                ${mainPreviewThumb
            ? `<img src="${mainPreviewThumb}" style="width: 100%; height: 100%; object-fit: contain;">`
            : `<div style="color: #666;">Video Preview</div>`}
            </div>

            <!-- Verdict Section -->
            <div class="report-verdict" style="margin: 0; padding: 20px; border-color: ${accentColor}; background: ${verdict.panelBg}">
                <div class="report-verdict-content" style="width: 100%;">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <div class="report-verdict-icon" style="background: ${accentColor}; color: white; width: 50px; height: 50px; font-size: 24px;">
                            <i class="fas ${verdict.icon}"></i>
                        </div>
                        <div>
                            <h2 style="color: ${accentColor}; font-size: 24px; margin: 0;">${prediction === 'FAKE' ? 'MANIPULATION DETECTED' : prediction === 'SUSPICIOUS' ? 'SUSPICIOUS MEDIA' : 'AUTHENTIC MEDIA'}</h2>
                        </div>
                    </div>
                    
                    <!-- Confidence Meter -->
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px;">
                            <span>Confidence Score</span>
                            <strong>${(currentResult.confidence * 100).toFixed(1)}%</strong>
                        </div>
                        <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden;">
                            <div style="width: ${(currentResult.confidence * 100).toFixed(1)}%; height: 100%; background: ${accentColor};"></div>
                        </div>
                    </div>

                    <p style="font-size: 12px; opacity: 0.8; line-height: 1.4;">
                        ${currentResult.detection_explanation_summary || verdict.summaryFallback}
                    </p>
                </div>
            </div>
        </div>

        <div class="report-grid">
            <div class="report-card">
                <h3>ANALYSIS STATISTICS</h3>
                <div class="report-stats-grid">
                    <div class="report-stat-item">
                        <span class="report-stat-label">Duration</span>
                        <span class="report-stat-value">${(Number.isFinite(currentResult.duration) ? currentResult.duration : 0).toFixed(1)}s</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Frames Scanned</span>
                        <span class="report-stat-value">${Number.isFinite(currentResult.processed_frames) ? currentResult.processed_frames : 1}</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Suspicious Frames</span>
                        <span class="report-stat-value" style="color: ${prediction === 'FAKE' ? '#ff3333' : prediction === 'SUSPICIOUS' ? '#f59e0b' : 'inherit'}">${suspiciousFrames.length}</span>
                    </div>
                    <div class="report-stat-item">
                        <span class="report-stat-label">Avg. Anomaly Score</span>
                        <span class="report-stat-value">${((Number.isFinite(currentResult.avg_fake_prob) ? currentResult.avg_fake_prob : Number.isFinite(currentResult.fake_probability) ? currentResult.fake_probability : 0) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>

            <div class="report-card">
                <h3>DETAILED FINDINGS</h3>
                <div style="font-size: 14px; line-height: 1.6; opacity: 0.9;">
                    ${document.getElementById('analysisNotes').innerHTML}
                </div>
            </div>
        </div>

        ${chartImg ? `
        <div class="report-card" style="margin-bottom: 30px;">
            <h3>TEMPORAL ANALYSIS</h3>
            <img src="${chartImg}" style="width: 100%; height: auto; max-height: 250px; object-fit: contain; margin-top: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
        </div>
        ` : ''}

        <div class="report-card">
            <h3>KEY FRAMES</h3>
            <div class="report-frames">
                ${framesHTML}
            </div>
        </div>

        <div class="report-footer">
            <p>Deepfake Detection System AI Analysis System â€¢ Deepfake Detection Report â€¢ ${dateStr}</p>
        </div>
    `;

    // 2. Generate PDF using html2canvas and jspdf
    // Unhide container temporarily (off-screen but rendered)
    reportContainer.style.opacity = '1';
    reportContainer.style.zIndex = '9999';
    reportContainer.style.background = '#0a0a0a';

    try {
        // Ensure libraries are loaded
        if (!window.html2canvas) {
            throw new Error("html2canvas library not loaded");
        }

        // Check for jspdf in various likely locations
        const jsPDF = window.jspdf?.jsPDF || window.jsPDF;
        if (!jsPDF) {
            console.error("jspdf debug:", window.jspdf);
            throw new Error("jspdf library not loaded properly");
        }

        const canvas = await html2canvas(reportContainer, {
            scale: 2, // Improve quality
            useCORS: true,
            backgroundColor: '#0a0a0a',
            logging: false
        });

        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4'
        });

        const imgWidth = 210; // A4 width in mm
        const pageHeight = 297; // A4 height in mm
        const imgHeight = (canvas.height * imgWidth) / canvas.width;

        pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);

        // If content is longer than one page (unlikely with this layout but good to handle)
        // For now, simpler single page strictly controlled by layout

        pdf.save(`Deepfake Detection System_Report_${Date.now()}.pdf`);

    } finally {
        // Hide again
        reportContainer.style.opacity = '0';
        reportContainer.style.zIndex = '-1000';
    }
}



