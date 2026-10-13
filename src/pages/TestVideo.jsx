import React, { useRef, useEffect, useState } from 'react';
import { api } from '../api';
import { 
  Play, 
  Pause, 
  Upload, 
  Terminal, 
  User, 
  Sparkles, 
  Loader2,
  FileVideo,
  Monitor,
  CheckCircle2,
  AlertCircle,
  Activity,
  Sliders,
  RotateCcw,
  Eye,
  ShieldCheck,
  Smartphone
} from 'lucide-react';

const EMOTION_COLORS = {
  happy: '#10b981',
  neutral: '#38bdf8',
  studying: '#10b981',
  attentive: '#00f0ff',
  distracted: '#f59e0b',
  talking: '#f59e0b',
  drowsy: '#ef4444',
  phone: '#ef4444'
};

export default function TestVideo() {
  const videoRef = useRef(null);
  const scrollRef = useRef(null);
  
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [serverVideos, setServerVideos] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [activeDetections, setActiveDetections] = useState([
    { id: 'R1C1', name: 'Student R1C1', attn: 92, act: 'studying', emo: 'attentive', phone: false },
    { id: 'R1C2', name: 'Student R1C2', attn: 88, act: 'attentive', emo: 'neutral', phone: false },
    { id: 'R1C3', name: 'Student R1C3', attn: 76, act: 'phone_use', emo: 'distracted', phone: true },
    { id: 'R1C4', name: 'Student R1C4', attn: 94, act: 'studying', emo: 'attentive', phone: false },
    { id: 'R2C1', name: 'Student R2C1', attn: 86, act: 'reading', emo: 'neutral', phone: false },
    { id: 'R2C5', name: 'Student R2C5', attn: 81, act: 'writing', emo: 'attentive', phone: false },
  ]);

  const addLog = (msg, type = 'info') => {
    setLogs(prev => {
      const newLogs = [...prev, { 
        id: Date.now() + Math.random(), 
        msg, 
        type, 
        time: new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) 
      }];
      return newLogs.slice(-100);
    });
  };

  useEffect(() => {
    api.getVideoList()
      .then(res => {
        const vids = res.videos || [];
        setServerVideos(vids);
        if (vids.length > 0) {
          // Select annotated real classroom video as default
          const realClass = vids.find(v => v.filename.includes('real_classroom')) || vids[0];
          selectServerVideo(realClass);
        }
      })
      .catch(err => addLog(`Failed to load server videos: ${err.message}`, 'error'));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const selectServerVideo = (video) => {
    setVideoFile({ serverPath: video.path, name: video.filename });
    setVideoUrl(`http://localhost:8000${video.streamUrl || `/media-videos/${encodeURIComponent(video.filename)}`}`);
    addLog(`Loaded video source: ${video.filename}`, 'info');
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile({ name: file.name, file });
      setVideoUrl(URL.createObjectURL(file));
      addLog(`Selected local video: ${file.name}`, 'info');
    }
  };

  const handlePlayToggle = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
        setIsPlaying(false);
      } else {
        videoRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const handleSpeedChange = (speed) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  return (
    <div className="apple-command-root" style={{ gap: '1.75rem' }}>
      {/* Top Banner Header */}
      <div className="apple-header-card">
        <div className="apple-header-left">
          <div className="header-title-block">
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span className="shadcn-badge shadcn-badge-cyan">
                <FileVideo size={12} />
                Video AI Analytics Engine
              </span>
              <span className="shadcn-badge shadcn-badge-emerald">
                <CheckCircle2 size={12} />
                YOLO11s + Multi-Person Tracking
              </span>
            </div>
            <h1>Classroom Video Engine Studio</h1>
            <p>High-throughput offline & streaming video diagnostics: posture, attention curves, and seat attribution.</p>
          </div>
        </div>

        <div className="apple-header-actions">
          <label className="shadcn-btn shadcn-btn-secondary" style={{ cursor: 'pointer' }}>
            <Upload size={14} />
            Upload Video
            <input type="file" accept="video/*" onChange={handleFileUpload} style={{ display: 'none' }} />
          </label>
        </div>
      </div>

      {/* Main Studio Workspace Split */}
      <div className="vision-hud-split" style={{ gridTemplateColumns: '1.8fr 1.2fr' }}>
        {/* Left: Video Player & Controls */}
        <div className="hud-viewport-card">
          <div className="hud-viewport-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '0.86rem', fontWeight: 700, color: '#f8fafc' }}>
                {videoFile?.name || 'No Video Selected'}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button 
                className="shadcn-btn shadcn-btn-secondary" 
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                onClick={handlePlayToggle}
              >
                {isPlaying ? <Pause size={13} /> : <Play size={13} />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>

              {[0.5, 1.0, 1.5, 2.0].map(s => (
                <button 
                  key={s}
                  className={`shadcn-btn ${playbackSpeed === s ? 'shadcn-btn-cyan' : 'shadcn-btn-secondary'}`}
                  style={{ padding: '0.35rem 0.55rem', fontSize: '0.72rem' }}
                  onClick={() => handleSpeedChange(s)}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Video Player Display */}
          <div className="hud-video-box" style={{ background: '#070c18', minHeight: '440px' }}>
            {videoUrl ? (
              <video 
                ref={videoRef}
                src={videoUrl}
                controls 
                autoPlay 
                loop 
                className="hud-feed-img"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                style={{ maxHeight: '560px', width: '100%' }}
              />
            ) : (
              <div className="hud-offline-box">
                <FileVideo size={40} color="var(--muted-foreground)" />
                <div>Select or upload a video below to begin analysis</div>
              </div>
            )}
          </div>

          {/* Quick Video Selector Strip */}
          <div style={{ 
            padding: '12px 18px', 
            background: 'rgba(15, 23, 42, 0.75)', 
            borderTop: '1px solid var(--border)',
            display: 'flex', 
            gap: '10px', 
            overflowX: 'auto',
            alignItems: 'center'
          }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)', fontWeight: 700, textTransform: 'uppercase' }}>
              Sample Videos:
            </span>
            {serverVideos.map((v, i) => (
              <button 
                key={i}
                className={`shadcn-btn ${videoFile?.name === v.filename ? 'shadcn-btn-cyan' : 'shadcn-btn-secondary'}`}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                onClick={() => selectServerVideo(v)}
              >
                {v.filename}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Real-time Telemetry & Logs */}
        <div className="hud-side-queue">
          {/* Real-time Detections Telemetry */}
          <div className="sc-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={16} color="#00f0ff" />
                <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>Detected Students Telemetry</h3>
              </div>
              <span className="shadcn-badge shadcn-badge-cyan">{activeDetections.length} ACTIVE</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '230px', overflowY: 'auto' }}>
              {activeDetections.map(d => (
                <div 
                  key={d.id}
                  style={{
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'rgba(15, 23, 42, 0.75)',
                    border: d.phone ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid var(--border)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span className="mono" style={{ 
                      fontSize: '0.75rem', 
                      fontWeight: 700, 
                      color: d.id.startsWith('R1') ? 'var(--accent-cyan)' : 'var(--accent-violet)' 
                    }}>
                      {d.id}
                    </span>
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#f8fafc' }}>{d.name}</div>
                      <div style={{ fontSize: '0.68rem', color: EMOTION_COLORS[d.act] || 'var(--muted-foreground)', textTransform: 'capitalize' }}>
                        {d.act}
                      </div>
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div className="mono" style={{ fontSize: '0.85rem', fontWeight: 700, color: d.attn >= 80 ? '#10b981' : '#f59e0b' }}>
                      {d.attn}%
                    </div>
                    {d.phone && (
                      <span className="shadcn-badge shadcn-badge-rose" style={{ fontSize: '0.6rem', padding: '1px 4px' }}>
                        PHONE
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Execution Terminal Console */}
          <div className="sc-card" style={{ flex: 1, padding: '1.25rem', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Terminal size={15} color="#38bdf8" />
                <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#f8fafc' }}>Engine Diagnostics Console</h3>
              </div>
              <button 
                className="shadcn-btn shadcn-btn-secondary" 
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.68rem' }}
                onClick={() => setLogs([])}
              >
                Clear
              </button>
            </div>

            <div 
              ref={scrollRef}
              style={{
                background: 'rgba(11, 17, 32, 0.95)',
                padding: '12px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)',
                flex: 1,
                minHeight: '180px',
                maxHeight: '220px',
                overflowY: 'auto',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.72rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px'
              }}
            >
              <div style={{ color: '#10b981' }}>[INIT] Video Engine Ready. Hardware acceleration: Enabled.</div>
              {logs.map(log => (
                <div key={log.id} style={{ color: log.type === 'error' ? '#ef4444' : log.type === 'warn' ? '#f59e0b' : '#94a3b8' }}>
                  <span style={{ color: '#64748b' }}>[{log.time}]</span> {log.msg}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
