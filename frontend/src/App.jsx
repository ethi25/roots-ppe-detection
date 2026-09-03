import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [status, setStatus] = useState({ task: 'idle', progress: 0, message: 'System Idle', logs: [] });
  const [viewMode, setViewMode] = useState('live'); // 'live' or 'rendered'
  const [processedVideoUrl, setProcessedVideoUrl] = useState('');
  
  // Selection and Upload state
  const [selectedVideo, setSelectedVideo] = useState('E:\\New folder (2)\\CCTV 1.mp4');
  const [uploadedVideoPath, setUploadedVideoPath] = useState('');
  const [uploading, setUploading] = useState(false);
  const [demoMode, setDemoMode] = useState(true);
  
  const videoPlayerRef = useRef(null);

  // Poll status from FastAPI
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        setStatus(data);

        if (data.task === 'idle' && data.progress === 100 && data.message.includes('successfully')) {
          fetchLatestProcessedVideo();
        }
      } catch (err) {
        console.error('Error fetching status:', err);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1500);
    return () => clearInterval(interval);
  }, []);

  const fetchLatestProcessedVideo = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/processed-videos`);
      const videos = await res.json();
      if (videos && videos.length > 0) {
        setProcessedVideoUrl(`${API_BASE}${videos[0]}?t=${Date.now()}`);
      }
    } catch (err) {
      console.error('Error fetching processed video:', err);
    }
  };

  useEffect(() => {
    fetchLatestProcessedVideo();
  }, []);

  // Handle Upload
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setUploadedVideoPath(data.video_path);
      setSelectedVideo(data.video_path);
      setViewMode('live');
      alert(`Video uploaded successfully! Click '▶ Stream Live PPE Detection' to view live bounding boxes immediately.`);
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploading(false);
    }
  };

  // Trigger Off-line Detection & Video Saving
  const startInference = async () => {
    setViewMode('rendered');
    setProcessedVideoUrl('');
    try {
      const res = await fetch(`${API_BASE}/api/process-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: selectedVideo,
          strict_mode: true,
          alert_threshold_frames: 5,
          max_frames: demoMode ? 500 : 0,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.detail);
      }
    } catch (err) {
      alert('Request failed: ' + err.message);
    }
  };

  const getFileName = (path) => {
    if (!path) return '';
    return path.split('/').pop().split('\\').pop();
  };

  const liveFeedUrl = `${API_BASE}/api/video-feed?video_path=${encodeURIComponent(selectedVideo)}`;

  return (
    <div className="app-container" style={{ padding: '30px 50px', background: '#0b0f17', minHeight: '100vh' }}>
      
      {/* Header Banner */}
      <header style={{
        marginBottom: '24px',
        padding: '18px 28px',
        background: 'rgba(17, 24, 39, 0.7)',
        borderRadius: '16px',
        border: '1px solid var(--border-color)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backdropFilter: 'blur(10px)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <span style={{ fontSize: '2.2rem' }}>🛡️</span>
          <div>
            <h1 style={{ fontSize: '1.7rem', color: '#fff', margin: 0 }}>PPE Detection & Bounding Box System</h1>
            <p style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: '4px' }}>
              Upload any CCTV video to detect employees, helmets, masks, gloves, and safety shoes in real-time.
            </p>
          </div>
        </div>

        <div>
          {status.task !== 'idle' ? (
            <div className="badge badge-warning" style={{ padding: '10px 18px', fontSize: '0.88rem' }}>
              ⚡ Processing ({status.progress}%)
            </div>
          ) : (
            <div className="badge badge-success" style={{ padding: '10px 18px', fontSize: '0.88rem' }}>
              System Ready
            </div>
          )}
        </div>
      </header>

      {/* Upload & Controls */}
      <div className="glass-card" style={{ marginBottom: '24px' }}>
        <h3 style={{ marginBottom: '14px', color: '#fff' }}>1. Select or Upload CCTV Video</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: '15px', alignItems: 'end' }}>
          
          <div>
            <label className="form-label">Video Source</label>
            <select 
              value={selectedVideo} 
              onChange={(e) => {
                setSelectedVideo(e.target.value);
                setViewMode('live');
              }} 
              className="form-input"
            >
              <option value="E:\New folder (2)\CCTV 1.mp4">CCTV 1.mp4 (Sample Factory CCTV)</option>
              {uploadedVideoPath && (
                <option value={uploadedVideoPath}>Uploaded: {getFileName(uploadedVideoPath)}</option>
              )}
            </select>
          </div>

          <div>
            <label className="form-label">Upload Video</label>
            <input 
              type="file" 
              accept="video/*" 
              onChange={handleUpload} 
              style={{ display: 'none' }} 
              id="video-upload-input"
            />
            <label htmlFor="video-upload-input" className="btn-secondary" style={{ width: '100%', justifyContent: 'center', cursor: 'pointer' }}>
              {uploading ? 'Uploading...' : '📁 Choose File'}
            </label>
          </div>

          <div>
            <button 
              onClick={() => setViewMode('live')} 
              className="btn-secondary"
              style={{ 
                width: '100%', 
                height: '45px', 
                justifyContent: 'center', 
                background: viewMode === 'live' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.03)',
                borderColor: viewMode === 'live' ? 'var(--color-primary)' : 'var(--border-color)',
                color: '#fff',
                fontWeight: 'bold'
              }}
            >
              ▶ Stream Live Detection
            </button>
          </div>

          <div>
            <button 
              onClick={startInference} 
              disabled={status.task !== 'idle' || uploading} 
              className="btn-primary"
              style={{ width: '100%', height: '45px', justifyContent: 'center' }}
            >
              {status.task === 'process_video' ? `Rendering (${status.progress}%)` : '🎬 Render & Save MP4'}
            </button>
          </div>

        </div>

        <div style={{ marginTop: '14px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input 
            type="checkbox" 
            id="demo-mode"
            checked={demoMode} 
            onChange={(e) => setDemoMode(e.target.checked)} 
          />
          <label htmlFor="demo-mode" style={{ fontSize: '0.85rem', color: 'var(--color-warning)', cursor: 'pointer', fontWeight: 500 }}>
            Fast Demo Mode (Render first 500 frames) — Uncheck for full-length MP4 file save
          </label>
        </div>

        {status.task === 'process_video' && (
          <div style={{ marginTop: '16px', background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px' }}>
              <span style={{ color: 'var(--color-primary)' }}>{status.message}</span>
              <span>{status.progress}%</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${status.progress}%`, background: 'var(--color-primary)', height: '100%', transition: 'width 0.3s' }}></div>
            </div>
          </div>
        )}
      </div>

      {/* Main Stream Display Container */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h3 style={{ color: '#fff', margin: 0, display: 'inline-block', marginRight: '15px' }}>
              {viewMode === 'live' ? '⚡ Real-Time Live Detection Stream' : '🎬 Rendered Annotated MP4 Video'}
            </h3>
            <span style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)' }}>
              Source: {getFileName(selectedVideo)}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '15px', fontSize: '0.82rem' }}>
            <span style={{ color: '#10b981', fontWeight: 600 }}>🟩 Green Box = Present Gear / Compliant</span>
            <span style={{ color: '#ef4444', fontWeight: 600 }}>libre Red Box = Missing Gear / Violation</span>
          </div>
        </div>
        
        <div style={{ background: '#000', borderRadius: '14px', overflow: 'hidden', border: '1px solid var(--border-color)', minHeight: '480px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          
          {viewMode === 'live' ? (
            <img 
              src={liveFeedUrl} 
              alt="Real-time live bounding box feed" 
              style={{ width: '100%', maxHeight: '680px', display: 'block', objectFit: 'contain' }} 
            />
          ) : (
            processedVideoUrl ? (
              <video 
                ref={videoPlayerRef}
                src={processedVideoUrl} 
                controls 
                autoPlay
                style={{ width: '100%', maxHeight: '680px', display: 'block' }}
              />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '480px', color: 'var(--color-text-muted)' }}>
                <span style={{ fontSize: '2.5rem', marginBottom: '12px' }}>⚙️</span>
                <p style={{ fontSize: '1.1rem', color: '#fff' }}>Rendering MP4 video with bounding boxes...</p>
                <p style={{ fontSize: '0.85rem' }}>{status.message} ({status.progress}%)</p>
              </div>
            )
          )}

        </div>
      </div>

    </div>
  );
}

export default App;
