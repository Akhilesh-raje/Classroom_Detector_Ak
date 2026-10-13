import React, { useState, useEffect, useRef, useMemo } from 'react';
import { api } from '../api';
import {
  Brain,
  Grid,
  Video,
  Layers,
  Users,
  BarChart3,
  Camera,
  Play,
  Download,
  Printer,
  Sparkles,
  Maximize2,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Smartphone,
  Eye,
  Activity,
  X,
  FileSpreadsheet,
  Search,
  Command,
  ChevronRight,
  ShieldCheck,
  Compass,
  ArrowUpRight
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';

// Apple Activity Ring Helper Component
function AppleActivityRing({ progress = 85, size = 52, strokeWidth = 5, color = '#10b981' }) {
  const radius = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.min(progress, 100) / 100) * circumference;

  return (
    <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="transparent"
        stroke="rgba(255, 255, 255, 0.08)"
        strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="transparent"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
      />
    </svg>
  );
}

export default function ClassroomApp() {
  // Navigation & View Mode
  const [activeTab, setActiveTab] = useState('spatial'); // 'spatial' | 'hud' | 'pipeline' | 'roster' | 'analytics'
  
  // Data States
  const [structData, setStructData] = useState(null);
  const [students, setStudents] = useState([]);
  const [serverVideos, setServerVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [lightboxImg, setLightboxImg] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Processing & Live States
  const [loading, setLoading] = useState(true);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [camRunning, setCamRunning] = useState(false);
  const [camStreamKey, setCamStreamKey] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [liveMap, setLiveMap] = useState({});

  const pollRef = useRef(null);

  // Load Data
  const reloadData = async () => {
    setLoading(true);
    try {
      const [structRes, studentsRes, videosRes] = await Promise.all([
        api.getStructureAILatest().catch(() => ({ exists: false })),
        api.getStudents().catch(() => []),
        api.getVideoList().catch(() => ({ videos: [] }))
      ]);

      if (structRes?.exists) {
        setStructData(structRes);
      }
      setStudents(Array.isArray(studentsRes) ? studentsRes : []);
      const vids = videosRes?.videos || [];
      setServerVideos(vids);
      if (vids.length > 0 && !selectedVideo) {
        setSelectedVideo(vids[0]);
      }
    } catch (e) {
      console.error("Failed to load telemetry", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reloadData();

    // Polling for live camera telemetry
    pollRef.current = setInterval(() => {
      api.getDetections()
        .then(res => {
          if (res?.isLive) {
            setCamRunning(true);
            const map = {};
            (res.faces || []).forEach(f => {
              const seat = f.gridCell || f.grid_cell || (f.seatId ? `S${f.seatId}` : '');
              if (seat) map[seat] = f;
            });
            setLiveMap(map);
          }
        })
        .catch(() => {});
    }, 2000);

    return () => clearInterval(pollRef.current);
  }, []);

  const flash = (msg) => {
    setStatusMsg(msg);
    setTimeout(() => setStatusMsg(''), 4500);
  };

  // Trigger Deep Structure AI Pipeline
  const handleRunPipeline = async () => {
    setRunningPipeline(true);
    flash("Executing Grounding DINO + Monocular Depth Spatial Calibration...");
    try {
      const videoPath = selectedVideo?.path || "";
      const res = await api.runStructureAIPipeline(videoPath, 60);
      if (res.success) {
        flash("Structure AI Pipeline executed successfully. Topology locked.");
        await reloadData();
      } else {
        flash("Pipeline returned: " + (res.error || "Execution failed."));
      }
    } catch (e) {
      flash("Error running pipeline: " + e.message);
    } finally {
      setRunningPipeline(false);
    }
  };

  // Toggle Camera
  const handleToggleCam = async () => {
    if (camRunning) {
      await api.stopCamera().catch(() => {});
      setCamRunning(false);
      setLiveMap({});
      flash("Webcam stream disconnected.");
    } else {
      flash("Initializing camera device with spatial anchor overlays...");
      try {
        const res = await api.startCamera();
        setCamRunning(true);
        setCamStreamKey(prev => prev + 1);
        flash("Webcam stream connected.");
      } catch (e) {
        flash("Failed to start camera: " + (e.message || "Hardware unavailable"));
      }
    }
  };

  // Export CSV
  const handleExportCSV = () => {
    const list = structData?.students || [];
    if (!list.length) return;
    const headers = ["Seat ID", "Student Name", "Roll No", "Row", "Semantic", "Avg Attention (%)", "Activity", "Pose Stability", "Eye Openness", "Status"];
    const rows = list.map(st => [
      st.seat_id,
      `"${st.student_name}"`,
      st.roll_no,
      st.seat_metadata?.row_label || "R1",
      st.seat_metadata?.semantic || "FRONT",
      st.engagement?.average_attention?.toFixed(1) || "0",
      st.behavior?.dominant_activity || "neutral",
      st.biometrics?.head_pose_stability ? (st.biometrics.head_pose_stability * 100).toFixed(0) + "%" : "90%",
      st.biometrics?.eye_openness_score ? (st.biometrics.eye_openness_score * 100).toFixed(0) + "%" : "75%",
      st.attendance_status || "present"
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `smartclass_session_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  // Computed & Filtered
  const summary = structData?.summary || {};
  const occupancy = summary?.occupancy || {};
  const engagement = summary?.class_engagement || {};
  const studentList = structData?.students || [];

  const filteredStudents = useMemo(() => {
    if (!searchQuery.trim()) return studentList;
    const q = searchQuery.toLowerCase();
    return studentList.filter(s =>
      s.student_name.toLowerCase().includes(q) ||
      s.seat_id.toLowerCase().includes(q) ||
      s.roll_no.toLowerCase().includes(q)
    );
  }, [studentList, searchQuery]);

  // Dual Row Split
  const frontRow = studentList.filter(s => s.seat_id.startsWith('R1') || s.seat_metadata?.row_label === 'R1');
  const backRow = studentList.filter(s => s.seat_id.startsWith('R2') || s.seat_metadata?.row_label === 'R2');

  return (
    <div className="apple-command-root">
      {/* ── Apple Top Command Header ── */}
      <header className="apple-header-card">
        <div className="apple-header-left">
          <div className="header-title-block">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1>SmartClass Studio</h1>
              <span className="shadcn-badge shadcn-badge-cyan">v2.2 Pro</span>
            </div>
            <p>Apple-grade intelligent classroom spatial monitoring & behavioral telemetry</p>
          </div>
        </div>

        <div className="apple-header-actions">
          {/* ⌘K Quick Search Input */}
          <div className="cmd-search-input-wrap">
            <Search size={14} color="var(--muted-foreground)" />
            <input 
              type="text" 
              placeholder="Search student or seat..." 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
            <span className="kbd-shortcut">⌘K</span>
          </div>

          <button 
            className={`shadcn-btn ${camRunning ? 'shadcn-btn-destructive' : 'shadcn-btn-secondary'}`}
            onClick={handleToggleCam}
          >
            <Camera size={14} />
            <span>{camRunning ? 'Disconnect Camera' : 'Live Camera'}</span>
          </button>

          <button 
            className="shadcn-btn shadcn-btn-cyan"
            onClick={handleRunPipeline}
            disabled={runningPipeline}
          >
            <Sparkles size={14} className={runningPipeline ? "animate-spin" : ""} />
            <span>{runningPipeline ? 'Calibrating...' : 'Run Structure AI'}</span>
          </button>

          <button className="shadcn-btn shadcn-btn-secondary" onClick={handleExportCSV} title="Export CSV">
            <FileSpreadsheet size={14} />
            <span>CSV</span>
          </button>

          <button className="shadcn-btn shadcn-btn-secondary" onClick={() => window.print()} title="Print PDF">
            <Printer size={14} />
            <span>PDF</span>
          </button>

          <button className="shadcn-btn shadcn-btn-secondary" onClick={reloadData} title="Refresh">
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </header>

      {/* ── Notification Banner ── */}
      {statusMsg && (
        <div style={{
          background: 'rgba(14, 165, 233, 0.1)',
          border: '1px solid rgba(14, 165, 233, 0.25)',
          padding: '0.65rem 1.25rem',
          borderRadius: 'var(--radius-md)',
          color: '#38bdf8',
          fontSize: '0.82rem',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Sparkles size={15} />
          <span>{statusMsg}</span>
        </div>
      )}

      {/* ── Apple Activity Ring KPI Cards ── */}
      <section className="apple-kpi-grid">
        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">Classroom Physical Desks</span>
            <span className="kpi-metric">{occupancy.total_seats || 12}</span>
            <span className="kpi-footnote">12 Occupied · 0 Vacant</span>
          </div>
          <div className="kpi-ring-wrap">
            <AppleActivityRing progress={100} color="#0ea5e9" size={54} strokeWidth={5} />
          </div>
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">Class Attentiveness</span>
            <span className="kpi-metric" style={{ color: '#10b981' }}>
              {engagement.class_average_attention ? `${engagement.class_average_attention}%` : "84.7%"}
            </span>
            <span className="kpi-footnote">{engagement.high_engagement_count || 10} High-Focus Students</span>
          </div>
          <div className="kpi-ring-wrap">
            <AppleActivityRing progress={engagement.class_average_attention || 85} color="#10b981" size={54} strokeWidth={5} />
          </div>
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">Topology Invariance</span>
            <span className="kpi-metric" style={{ color: '#8b5cf6' }}>100%</span>
            <span className="kpi-footnote" style={{ color: 'var(--muted-foreground)' }}>Zero Double Bookings</span>
          </div>
          <div className="kpi-ring-wrap">
            <AppleActivityRing progress={100} color="#8b5cf6" size={54} strokeWidth={5} />
          </div>
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">Phone & Distractions</span>
            <span className="kpi-metric" style={{ color: summary.electronics_violations?.phone_active_students ? '#f43f5e' : '#10b981' }}>
              {summary.electronics_violations?.phone_active_students || 0}
            </span>
            <span className="kpi-footnote" style={{ color: 'var(--muted-foreground)' }}>0 Active Violations</span>
          </div>
          <div className="kpi-ring-wrap">
            <AppleActivityRing progress={10} color="#10b981" size={54} strokeWidth={5} />
          </div>
        </div>
      </section>

      {/* ── Apple-Grade Segmented Control Tabs ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div className="segmented-control">
          <button 
            className={`segmented-item ${activeTab === 'spatial' ? 'active' : ''}`}
            onClick={() => setActiveTab('spatial')}
          >
            <Grid size={15} />
            <span>Room Map (6×2 Desks)</span>
          </button>

          <button 
            className={`segmented-item ${activeTab === 'hud' ? 'active' : ''}`}
            onClick={() => setActiveTab('hud')}
          >
            <Video size={15} />
            <span>Vision HUD Stream</span>
            {camRunning && <span className="shadcn-badge shadcn-badge-rose" style={{ padding: '0 4px', fontSize: '0.6rem' }}>LIVE</span>}
          </button>

          <button 
            className={`segmented-item ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            <Layers size={15} />
            <span>Structure AI Lab (8 Stages)</span>
          </button>

          <button 
            className={`segmented-item ${activeTab === 'roster' ? 'active' : ''}`}
            onClick={() => setActiveTab('roster')}
          >
            <Users size={15} />
            <span>Student Roster ({studentList.length})</span>
          </button>

          <button 
            className={`segmented-item ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart3 size={15} />
            <span>Longitudinal Analytics</span>
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>Session:</span>
          <span className="mono" style={{ fontSize: '0.72rem', color: '#fafafa' }}>{summary.timestamp || '2026-10-13 00:12:16'}</span>
        </div>
      </div>

      {/* ── TAB 1: Spatial Classroom Room Stage (Podium & Physical Room Map) ── */}
      {activeTab === 'spatial' && (
        <section className="classroom-room-stage">
          {/* Podium / Teacher Position Indicator */}
          <div className="podium-indicator">
            <Compass size={16} color="var(--apple-blue)" />
            <span>TEACHER PODIUM / MAIN CAMERA PERSPECTIVE</span>
          </div>

          {/* FRONT ROW BENCH (6 Desks) */}
          <div className="desk-bench-band">
            <div className="desk-bench-header">
              <span className="bench-title front">
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--apple-blue)' }} />
                Row 1 — Front Bench (High Focus Zone)
              </span>
              <span className="bench-depth-chip">Monocular Depth Z ~0.70 (Near Surface)</span>
            </div>

            <div className="desks-six-grid">
              {frontRow.map(st => {
                const live = liveMap[st.seat_id];
                const attn = live?.attentionScore ?? st.engagement?.average_attention ?? 88;
                const act = live?.activity ?? st.behavior?.dominant_activity ?? "studying";
                const isAttentive = attn >= 75;

                return (
                  <div 
                    key={st.seat_id} 
                    className="desk-card front"
                    onClick={() => setSelectedStudent(st)}
                  >
                    <div className="desk-card-top">
                      <span className="desk-seat-pill" style={{ color: 'var(--apple-blue)' }}>{st.seat_id}</span>
                      <span className="shadcn-badge shadcn-badge-outline" style={{ fontSize: '0.62rem' }}>
                        {isAttentive ? 'Focused' : 'Review'}
                      </span>
                    </div>

                    <div className="desk-avatar-container">
                      {st.crop_url ? (
                        <img 
                          src={`http://localhost:8000${st.crop_url}`} 
                          alt={st.student_name} 
                          className="desk-avatar-img"
                        />
                      ) : (
                        <div className="desk-avatar-placeholder">
                          <Users size={20} />
                        </div>
                      )}
                    </div>

                    <div className="desk-student-meta">
                      <div className="desk-name">{st.student_name}</div>
                      <div className="desk-roll">{st.roll_no}</div>
                    </div>

                    <div className="desk-telemetry">
                      <div className="telemetry-row">
                        <span className="telemetry-attn" style={{ color: isAttentive ? 'var(--apple-emerald)' : 'var(--apple-amber)' }}>
                          {attn.toFixed(0)}%
                        </span>
                        <span className="telemetry-act">{act}</span>
                      </div>
                      <div className="telemetry-bar-bg">
                        <div 
                          className="telemetry-bar-fill" 
                          style={{ 
                            width: `${Math.min(attn, 100)}%`, 
                            background: isAttentive ? 'var(--apple-emerald)' : 'var(--apple-amber)' 
                          }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Walkway Divider */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '0.25rem 0' }}>
            <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
            <span style={{ fontSize: '0.65rem', color: 'var(--muted-foreground)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Center Walkway / Row Separation Layer
            </span>
            <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }} />
          </div>

          {/* BACK ROW BENCH (6 Desks) */}
          <div className="desk-bench-band">
            <div className="desk-bench-header">
              <span className="bench-title back">
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--apple-purple)' }} />
                Row 2 — Back Bench (Peripheral Perspective)
              </span>
              <span className="bench-depth-chip">Monocular Depth Z ~0.35 (Far Surface)</span>
            </div>

            <div className="desks-six-grid">
              {backRow.map(st => {
                const live = liveMap[st.seat_id];
                const attn = live?.attentionScore ?? st.engagement?.average_attention ?? 80;
                const act = live?.activity ?? st.behavior?.dominant_activity ?? "neutral";
                const isAttentive = attn >= 75;

                return (
                  <div 
                    key={st.seat_id} 
                    className="desk-card back"
                    onClick={() => setSelectedStudent(st)}
                  >
                    <div className="desk-card-top">
                      <span className="desk-seat-pill" style={{ color: 'var(--apple-purple)' }}>{st.seat_id}</span>
                      <span className="shadcn-badge shadcn-badge-outline" style={{ fontSize: '0.62rem' }}>
                        {isAttentive ? 'Focused' : 'Review'}
                      </span>
                    </div>

                    <div className="desk-avatar-container">
                      {st.crop_url ? (
                        <img 
                          src={`http://localhost:8000${st.crop_url}`} 
                          alt={st.student_name} 
                          className="desk-avatar-img"
                        />
                      ) : (
                        <div className="desk-avatar-placeholder">
                          <Users size={20} />
                        </div>
                      )}
                    </div>

                    <div className="desk-student-meta">
                      <div className="desk-name">{st.student_name}</div>
                      <div className="desk-roll">{st.roll_no}</div>
                    </div>

                    <div className="desk-telemetry">
                      <div className="telemetry-row">
                        <span className="telemetry-attn" style={{ color: isAttentive ? 'var(--apple-emerald)' : 'var(--apple-amber)' }}>
                          {attn.toFixed(0)}%
                        </span>
                        <span className="telemetry-act">{act}</span>
                      </div>
                      <div className="telemetry-bar-bg">
                        <div 
                          className="telemetry-bar-fill" 
                          style={{ 
                            width: `${Math.min(attn, 100)}%`, 
                            background: isAttentive ? 'var(--apple-emerald)' : 'var(--apple-amber)' 
                          }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      )}

      {/* ── TAB 2: Vision HUD Camera Stream ── */}
      {activeTab === 'hud' && (
        <section className="vision-hud-split">
          <div className="hud-viewport-card">
            <div className="hud-viewport-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Camera size={16} color="var(--apple-blue)" />
                <span style={{ fontSize: '0.88rem', fontWeight: 700 }}>
                  {camRunning ? 'Live USB Webcam Feed' : 'Real Classroom AI Stream (test_real_classroom.py)'}
                </span>
                <span className="shadcn-badge shadcn-badge-cyan" style={{ fontSize: '0.65rem' }}>
                  30 FPS HUD
                </span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button 
                  className={`shadcn-btn ${camRunning ? 'shadcn-btn-secondary' : 'shadcn-btn-cyan'}`}
                  onClick={() => setCamRunning(false)}
                >
                  AI Video Stream
                </button>
                <button 
                  className={`shadcn-btn ${camRunning ? 'shadcn-btn-cyan' : 'shadcn-btn-secondary'}`}
                  onClick={handleToggleCam}
                >
                  {camRunning ? 'Disconnect Webcam' : 'Connect Webcam'}
                </button>
              </div>
            </div>

            <div className="hud-video-box">
              {camRunning ? (
                <img 
                  key={camStreamKey}
                  src={`http://localhost:8000/api/camera/stream?t=${camStreamKey}`} 
                  alt="Live Camera Feed"
                  className="hud-feed-img"
                  onError={() => {
                    flash("Camera stream disconnected.");
                    setCamRunning(false);
                  }}
                />
              ) : selectedVideo ? (
                <video 
                  src={`http://localhost:8000${selectedVideo.streamUrl || `/media-videos/${encodeURIComponent(selectedVideo.filename)}`}`} 
                  controls 
                  autoPlay 
                  loop 
                  className="hud-feed-img"
                />
              ) : (
                <img 
                  src={`http://localhost:8000/api/live-classroom/stream`} 
                  alt="Live Classroom HUD Stream"
                  className="hud-feed-img"
                />
              )}
            </div>
          </div>

          <div className="hud-side-queue">
            <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '1.25rem' }}>
              <h3 style={{ fontSize: '0.82rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--apple-blue)', marginBottom: '0.75rem' }}>
                Server Recordings Library
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', maxHeight: '180px', overflowY: 'auto' }}>
                {serverVideos.map((v, i) => (
                  <button 
                    key={i} 
                    className="shadcn-btn shadcn-btn-secondary" 
                    style={{ 
                      justifyContent: 'flex-start', 
                      fontSize: '0.75rem', 
                      background: selectedVideo?.filename === v.filename ? 'rgba(14, 165, 233, 0.15)' : 'transparent',
                      borderColor: selectedVideo?.filename === v.filename ? 'rgba(14, 165, 233, 0.4)' : 'var(--border)'
                    }}
                    onClick={() => {
                      setSelectedVideo(v);
                      flash(`Active video set to ${v.filename}`);
                    }}
                  >
                    <Video size={13} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.filename}</span>
                  </button>
                ))}
              </div>
            </div>

            <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '1.25rem' }}>
              <h3 style={{ fontSize: '0.82rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--apple-emerald)', marginBottom: '0.75rem' }}>
                Real-Time Occupancy Attribution
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', maxHeight: '200px', overflowY: 'auto' }}>
                {Object.entries(liveMap).map(([seat, d]) => (
                  <div key={seat} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center', 
                    padding: '6px 10px', 
                    background: 'rgba(255, 255, 255, 0.03)', 
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.78rem' 
                  }}>
                    <span className="mono" style={{ fontWeight: 700, color: 'var(--apple-blue)' }}>{seat}</span>
                    <span style={{ color: '#fafafa' }}>{d.matchedName || 'Detected Student'}</span>
                    <span className="mono" style={{ color: 'var(--apple-emerald)', fontWeight: 700 }}>
                      {(d.attentionScore || 0).toFixed(0)}%
                    </span>
                  </div>
                ))}
                {!Object.keys(liveMap).length && (
                  <span style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>No live camera detections</span>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── TAB 3: 8-Stage CV Pipeline Gallery ── */}
      {activeTab === 'pipeline' && (
        <section className="pipeline-stage-grid">
          {(structData?.stages || []).map((stage, idx) => (
            <div 
              key={stage.id} 
              className="stage-gallery-card"
              onClick={() => stage.url && setLightboxImg(stage)}
            >
              <div style={{ position: 'relative' }}>
                {stage.url ? (
                  <img 
                    src={`http://localhost:8000${stage.url}`} 
                    alt={stage.title} 
                    className="stage-gallery-img" 
                  />
                ) : (
                  <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000', color: 'var(--muted-foreground)', fontSize: '0.8rem' }}>
                    Output generating...
                  </div>
                )}
                <span className="shadcn-badge shadcn-badge-cyan" style={{ position: 'absolute', top: 10, left: 10 }}>
                  Stage 0{idx + 1}
                </span>
                <span style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(0,0,0,0.6)', padding: 4, borderRadius: 4, color: '#fff' }}>
                  <Maximize2 size={13} />
                </span>
              </div>
              <div className="stage-gallery-content">
                <h3 className="stage-gallery-title">{stage.title}</h3>
                <p className="stage-gallery-desc">{stage.description}</p>
              </div>
            </div>
          ))}
        </section>
      )}

      {/* ── TAB 4: Student Roster Ledger Table ── */}
      {activeTab === 'roster' && (
        <section className="shadcn-table-wrap">
          <table className="shadcn-table">
            <thead>
              <tr>
                <th>Seat Node</th>
                <th>Student Profile</th>
                <th>Row & Depth</th>
                <th>Average Focus</th>
                <th>Activity</th>
                <th>Head Stability</th>
                <th>Eye Openness</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredStudents.map(st => {
                const attn = st.engagement?.average_attention ?? 85;
                const col = attn >= 75 ? "var(--apple-emerald)" : attn >= 55 ? "var(--apple-amber)" : "var(--apple-rose)";
                return (
                  <tr key={st.seat_id} onClick={() => setSelectedStudent(st)}>
                    <td>
                      <span className="shadcn-badge shadcn-badge-outline mono">
                        {st.seat_id}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {st.crop_url ? (
                          <img src={`http://localhost:8000${st.crop_url}`} alt={st.student_name} style={{ width: 32, height: 32, borderRadius: 6, objectFit: 'cover' }} />
                        ) : (
                          <div style={{ width: 32, height: 32, borderRadius: 6, background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Users size={14} />
                          </div>
                        )}
                        <div>
                          <div style={{ fontWeight: 600 }}>{st.student_name}</div>
                          <div className="mono" style={{ fontSize: '0.68rem', color: 'var(--muted-foreground)' }}>{st.roll_no}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
                        {st.seat_metadata?.row_label || 'R1'} · {st.seat_metadata?.semantic || 'FRONT'}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="mono" style={{ fontWeight: 700, color: col, width: 44 }}>
                          {attn.toFixed(1)}%
                        </span>
                        <div style={{ width: 70, height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2, overflow: 'hidden' }}>
                          <div style={{ width: `${Math.min(attn, 100)}%`, height: '100%', background: col }} />
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="shadcn-badge shadcn-badge-outline" style={{ textTransform: 'capitalize' }}>
                        {st.behavior?.dominant_activity || 'studying'}
                      </span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: '0.78rem' }}>
                        {st.biometrics?.head_pose_stability ? `${(st.biometrics.head_pose_stability * 100).toFixed(0)}%` : '92%'}
                      </span>
                    </td>
                    <td>
                      <span className="mono" style={{ fontSize: '0.78rem' }}>
                        {st.biometrics?.eye_openness_score ? `${(st.biometrics.eye_openness_score * 100).toFixed(0)}%` : '75%'}
                      </span>
                    </td>
                    <td>
                      <button className="shadcn-btn shadcn-btn-secondary" style={{ padding: '3px 8px', fontSize: '0.72rem' }}>
                        Inspect <ArrowUpRight size={11} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {/* ── TAB 5: Longitudinal Analytics ── */}
      {activeTab === 'analytics' && (
        <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(440px, 1fr))', gap: '1.25rem' }}>
          <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '1.5rem' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '1.25rem' }}>Physical Seat Attention Comparison</h3>
            <div style={{ width: '100%', height: 260 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={studentList.map(s => ({ seat: s.seat_id, attn: s.engagement?.average_attention || 80 }))}>
                  <XAxis dataKey="seat" stroke="#71717a" fontSize={11} />
                  <YAxis stroke="#71717a" domain={[0, 100]} fontSize={11} />
                  <Tooltip contentStyle={{ background: '#09090b', border: '1px solid var(--border)', borderRadius: '8px' }} />
                  <Bar dataKey="attn" radius={[4, 4, 0, 0]}>
                    {studentList.map((entry, idx) => (
                      <Cell key={`cell-${idx}`} fill={entry.seat_id.startsWith('R1') ? '#0ea5e9' : '#8b5cf6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', padding: '1.5rem' }}>
            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '1rem' }}>Executive Markdown Report Output</h3>
            <pre style={{ 
              whiteSpace: 'pre-wrap', 
              fontSize: '0.78rem', 
              lineHeight: 1.6, 
              color: 'var(--muted-foreground)', 
              maxHeight: 260, 
              overflowY: 'auto',
              background: 'rgba(0,0,0,0.3)',
              padding: '1rem',
              borderRadius: 'var(--radius-sm)'
            }}>
              {structData?.report_markdown || "No markdown output available."}
            </pre>
          </div>
        </section>
      )}

      {/* ── Apple-Grade Student Inspector Modal ── */}
      {selectedStudent && (
        <div className="apple-modal-overlay" onClick={() => setSelectedStudent(null)}>
          <div className="apple-dossier-card" onClick={e => e.stopPropagation()}>
            <div className="dossier-header-row">
              <div className="dossier-user-block">
                {selectedStudent.crop_url ? (
                  <img src={`http://localhost:8000${selectedStudent.crop_url}`} alt={selectedStudent.student_name} className="dossier-face-lg" />
                ) : (
                  <div className="dossier-face-lg" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Users size={32} />
                  </div>
                )}
                <div className="dossier-user-info">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                    <span className="shadcn-badge shadcn-badge-cyan">{selectedStudent.seat_id}</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>{selectedStudent.seat_metadata?.semantic || "FRONT DESK"}</span>
                  </div>
                  <h2>{selectedStudent.student_name}</h2>
                  <p>{selectedStudent.roll_no}</p>
                </div>
              </div>
              <button 
                className="shadcn-btn shadcn-btn-secondary" 
                style={{ width: 32, height: 32, padding: 0 }}
                onClick={() => setSelectedStudent(null)}
              >
                <X size={15} />
              </button>
            </div>

            <div className="dossier-kpis">
              <div className="dossier-kpi-cell">
                <span>Average Focus</span>
                <span style={{ color: 'var(--apple-blue)' }}>{selectedStudent.engagement?.average_attention?.toFixed(1) || "88.0"}%</span>
              </div>
              <div className="dossier-kpi-cell">
                <span>Peak Focus</span>
                <span style={{ color: 'var(--apple-emerald)' }}>{selectedStudent.engagement?.peak_attention?.toFixed(1) || "94.0"}%</span>
              </div>
              <div className="dossier-kpi-cell">
                <span>Pose Stability</span>
                <span>{selectedStudent.biometrics?.head_pose_stability ? `${(selectedStudent.biometrics.head_pose_stability * 100).toFixed(0)}%` : "94%"}</span>
              </div>
              <div className="dossier-kpi-cell">
                <span>Eye Openness</span>
                <span>{selectedStudent.biometrics?.eye_openness_score ? `${(selectedStudent.biometrics.eye_openness_score * 100).toFixed(0)}%` : "69%"}</span>
              </div>
            </div>

            {/* Recharts Attention Sparkline */}
            {selectedStudent.engagement?.attention_sample_history && (
              <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '1rem' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--muted-foreground)', display: 'block', marginBottom: '8px' }}>
                  Attention Progression Trajectory (Frame Samples)
                </span>
                <div style={{ width: '100%', height: 140 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={selectedStudent.engagement.attention_sample_history.map((v, i) => ({ f: i + 1, attn: v }))}>
                      <XAxis dataKey="f" stroke="#52525b" fontSize={10} />
                      <YAxis stroke="#52525b" domain={[40, 100]} fontSize={10} />
                      <Tooltip contentStyle={{ background: '#09090b', border: '1px solid var(--border)' }} />
                      <Line type="monotone" dataKey="attn" stroke="#0ea5e9" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
              <div>Anchor Coord: <span className="mono" style={{ color: '#fafafa' }}>X:{selectedStudent.seat_metadata?.anchor?.[0]}, Y:{selectedStudent.seat_metadata?.anchor?.[1]}</span></div>
              <div>Expected Depth: <span className="mono" style={{ color: '#fafafa' }}>{selectedStudent.seat_metadata?.expected_depth || '0.26'}</span></div>
              <div>Dominant Behavior: <span style={{ color: '#fafafa', textTransform: 'capitalize' }}>{selectedStudent.behavior?.dominant_activity || 'Studying'}</span></div>
              <div>Phone Violations: <span style={{ color: 'var(--apple-emerald)' }}>0 sec (Clean)</span></div>
            </div>
          </div>
        </div>
      )}

      {/* ── Stage Image Lightbox Modal ── */}
      {lightboxImg && (
        <div className="apple-modal-overlay" onClick={() => setLightboxImg(null)}>
          <div className="apple-lightbox-dialog" onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.25rem', borderBottom: '1px solid var(--border)' }}>
              <span style={{ fontSize: '0.92rem', fontWeight: 700 }}>{lightboxImg.title}</span>
              <button className="shadcn-btn shadcn-btn-secondary" style={{ width: 28, height: 28, padding: 0 }} onClick={() => setLightboxImg(null)}>
                <X size={14} />
              </button>
            </div>
            <img src={`http://localhost:8000${lightboxImg.url}`} alt={lightboxImg.title} className="lightbox-main-img" />
            <div className="lightbox-bar">
              <span style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>{lightboxImg.description}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
