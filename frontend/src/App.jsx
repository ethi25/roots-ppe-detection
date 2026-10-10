import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [activeCam, setActiveCam] = useState('d29');
  const [stats, setStats] = useState({
    camera_name: '[D29] POWDER COATING OFFICE AREA',
    active_workers: 0,
    compliant_workers: 0,
    violating_workers: 0,
    compliance_rate: 100,
    fps: 0,
    device_name: 'NVIDIA RTX 4050 GPU',
    recent_events: []
  });
  const [isPlaying, setIsPlaying] = useState(true);
  const [audioAlert, setAudioAlert] = useState(false);
  const [cameras, setCameras] = useState([
    { id: 'd29', name: '[D29] Powder Coating Office Area' },
    { id: 'd3', name: '[D3] PC Spray Booth West Wall' },
    { id: 'd23', name: '[D23] Tool Room' },
    { id: 'webcam', name: 'Live Webcam' },
    { id: 'demo', name: 'Demo Footage' },
  ]);
  const [streamUrl, setStreamUrl] = useState(`${API_BASE}/api/cctv-stream?camera=d29&t=${Date.now()}`);
  const [isStreamLoading, setIsStreamLoading] = useState(true);
  const [streamError, setStreamError] = useState(false);

  const videoContainerRef = useRef(null);
  const lastEventKeyRef = useRef('');

  // Switch camera feed
  const switchCamera = (camId) => {
    setActiveCam(camId);
    setIsStreamLoading(true);
    setStreamError(false);
    setStreamUrl(`${API_BASE}/api/cctv-stream?camera=${camId}&t=${Date.now()}`);
  };

  // Poll live stats every 1 second
  useEffect(() => {
    let isMounted = true;
    const fetchStats = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/live-stats`);
        if (res.ok && isMounted) {
          const data = await res.json();
          setStats(data);

          // Audio chime on new violation if enabled
          if (audioAlert && data.recent_events && data.recent_events.length > 0) {
            const latest = data.recent_events[0];
            if (latest.key !== lastEventKeyRef.current) {
              lastEventKeyRef.current = latest.key;
              playAlertSound();
            }
          }
        }
      } catch (err) {
        // Server initializing
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 1000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [audioAlert]);

  // Audio tone generator
  const playAlertSound = () => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } catch (e) {
      // AudioContext policy
    }
  };

  // Take Snapshot
  const handleSnapshot = () => {
    const link = document.createElement('a');
    link.href = `${API_BASE}/api/snapshot?t=${Date.now()}`;
    link.download = `ppe_snapshot_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Toggle Fullscreen
  const toggleFullscreen = () => {
    if (!videoContainerRef.current) return;
    if (!document.fullscreenElement) {
      videoContainerRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // Calculate compliance color
  const getComplianceColor = (rate) => {
    if (rate >= 90) return 'var(--color-success)';
    if (rate >= 50) return 'var(--color-warning)';
    return 'var(--color-danger)';
  };

  return (
    <div className="app-container">
      {/* 1. Clean App Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon">🛡️</div>
          <div>
            <h1 className="brand-title">ROOTS PPE Compliance Monitor</h1>
            <p className="brand-subtitle">
              Live Factory Industrial Safety & Protective Equipment Detection
            </p>
          </div>
        </div>

        <div className="header-actions">
          <span className="pill-badge live">
            <span className="live-dot"></span>
            Live Detection
          </span>
          <span className="pill-badge" style={{ background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1' }}>
            ⚡ {stats.device_name || 'RTX 4050 GPU'}
          </span>
        </div>
      </header>

      {/* 2. Top Metrics Overview Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Compliance Rate</span>
            <span className="metric-icon">📊</span>
          </div>
          <div className="metric-value" style={{ color: getComplianceColor(stats.compliance_rate) }}>
            {stats.compliance_rate}%
          </div>
          <div className="metric-sub">
            {stats.compliant_workers} of {Math.max(stats.active_workers, 1)} workers compliant
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Workers Monitored</span>
            <span className="metric-icon">👥</span>
          </div>
          <div className="metric-value">
            {stats.active_workers}
          </div>
          <div className="metric-sub">
            Tracked in current camera zone
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Face Protection</span>
            <span className="metric-icon">😷</span>
          </div>
          <div className="metric-value" style={{ fontSize: '1.25rem', color: '#0284c7' }}>
            Active
          </div>
          <div className="metric-sub">
            Pink / Blue / White N95 & Respirators
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Hand Protection</span>
            <span className="metric-icon">🧤</span>
          </div>
          <div className="metric-value" style={{ fontSize: '1.25rem', color: '#d97706' }}>
            Palm Analysis
          </div>
          <div className="metric-sub">
            Protective Gloves vs Bare Skin Palm
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span className="metric-title">Inference Speed</span>
            <span className="metric-icon">⚡</span>
          </div>
          <div className="metric-value" style={{ color: 'var(--color-primary)' }}>
            {stats.fps > 0 ? `${stats.fps} FPS` : 'Ready'}
          </div>
          <div className="metric-sub">
            Hardware Accelerated on RTX 4050
          </div>
        </div>
      </div>

      {/* 3. Camera Selector Bar */}
      <div className="camera-selector-bar">
        <div className="camera-tabs">
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginRight: 6 }}>
            CCTV Channels:
          </span>
          {cameras.map((c) => (
            <button
              key={c.id}
              className={`camera-btn ${activeCam === c.id ? 'active' : ''}`}
              onClick={() => switchCamera(c.id)}
            >
              📹 {c.name}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="btn-action"
            onClick={() => setAudioAlert(!audioAlert)}
            title="Toggle audio alert chime on violation"
          >
            {audioAlert ? '🔔 Sound: ON' : '🔕 Sound: OFF'}
          </button>
          <button
            className="btn-action"
            onClick={handleSnapshot}
            title="Save high-resolution snapshot"
          >
            📸 Snapshot
          </button>
        </div>
      </div>

      {/* 4. Main Video Stream Display */}
      <div className="stream-container-card" ref={videoContainerRef}>
        <div className="stream-card-header">
          <div className="stream-card-title">
            <span>🔴 {stats.camera_name || 'Factory CCTV Stream'}</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 400 }}>
              (Fine-tuned YOLOv8m + ByteTrack)
            </span>
          </div>

          <div className="stream-actions">
            <button
              className="btn-action"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? '⏸ Pause' : '▶ Resume'}
            </button>
            <button
              className="btn-action"
              onClick={toggleFullscreen}
            >
              ⛶ Fullscreen
            </button>
          </div>
        </div>

        <div className="video-frame-wrapper">
          {isPlaying ? (
            <>
              {isStreamLoading && (
                <div className="stream-loading-badge">
                  <div className="stream-spinner"></div>
                  <span>Loading feed...</span>
                </div>
              )}
              {streamError ? (
                <div style={{ padding: '60px 20px', color: '#ffffff', textAlign: 'center' }}>
                  <p style={{ fontSize: '1.1rem', marginBottom: '8px', color: '#f87171' }}>
                    Camera feed temporarily unavailable
                  </p>
                  <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '16px' }}>
                    Switch to recorded factory demo footage for testing
                  </p>
                  <button
                    className="btn-action"
                    style={{ background: 'var(--color-primary)', color: '#ffffff', border: 'none' }}
                    onClick={() => switchCamera('demo')}
                  >
                    ▶ Switch to Demo Footage
                  </button>
                </div>
              ) : (
                <img
                  key={streamUrl}
                  src={streamUrl}
                  alt="Live PPE Compliance Feed"
                  className="stream-img"
                  onLoad={() => {
                    setIsStreamLoading(false);
                    setStreamError(false);
                  }}
                  onError={() => {
                    setIsStreamLoading(false);
                    setStreamError(true);
                  }}
                />
              )}
            </>
          ) : (
            <div style={{ padding: '80px 20px', color: '#ffffff', textAlign: 'center' }}>
              <p style={{ fontSize: '1.2rem', marginBottom: '12px' }}>Stream Paused</p>
              <button
                className="btn-action"
                style={{ background: 'var(--color-primary)', color: '#ffffff', border: 'none' }}
                onClick={() => setIsPlaying(true)}
              >
                ▶ Resume Feed
              </button>
            </div>
          )}

          <div className="stream-hud-overlay">
            <span className="hud-pill">{stats.fps} FPS</span>
            <span className="hud-pill">{stats.active_workers} Workers</span>
            <span
              className="hud-pill"
              style={{
                background: stats.compliance_rate >= 90 ? 'rgba(16, 185, 129, 0.85)' : 'rgba(239, 68, 68, 0.85)'
              }}
            >
              {stats.compliance_rate}% Compliant
            </span>
          </div>
        </div>
      </div>

      {/* 5. Live Activity & Violation Log */}
      <div className="activity-card">
        <div className="activity-header">
          <h2 className="activity-title">Live Safety Activity & Violations Log</h2>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Real-time feed across monitored zones
          </span>
        </div>

        <div className="table-wrapper">
          {stats.recent_events && stats.recent_events.length > 0 ? (
            <table className="clean-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Camera Zone</th>
                  <th>Worker ID</th>
                  <th>Violation Details</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_events.map((ev, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 600, color: 'var(--text-muted)' }}>{ev.time}</td>
                    <td>{ev.camera}</td>
                    <td>
                      <strong>Worker #{ev.worker_id}</strong>
                    </td>
                    <td>
                      {ev.missing && ev.missing.length > 0 ? (
                        <span className="tag-violation">
                          ⚠️ Missing {ev.missing.join(', ')}
                        </span>
                      ) : (
                        <span className="tag-compliant">Compliant</span>
                      )}
                    </td>
                    <td>
                      <span className="tag-violation">Needs Action</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">✅</div>
              <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: 4 }}>
                All Workers Fully Compliant
              </p>
              <p style={{ fontSize: '0.85rem' }}>
                All detected workers are currently wearing both face protection (masks/respirators) and protective gloves.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 6. Footer */}
      <footer className="app-footer">
        ROOTS Industrial PPE Detection Engine • Powered by Fine-tuned YOLOv8m on NVIDIA RTX 4050 • ByteTrack Multi-Worker Tracking
      </footer>
    </div>
  );
}

export default App;
