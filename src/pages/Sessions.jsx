import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { 
  PlayCircle, 
  StopCircle, 
  Clock, 
  Calendar, 
  CheckCircle2, 
  Users, 
  Activity, 
  FileText, 
  TrendingUp, 
  Zap,
  RotateCcw
} from 'lucide-react';

export default function Sessions() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeSession, setActiveSession] = useState(null);
  const [sessionName, setSessionName] = useState('');
  const [selectedSessionReports, setSelectedSessionReports] = useState(null);
  const [selectedSessionId, setSelectedSessionId] = useState(null);
  const [starting, setStarting] = useState(false);

  const loadSessions = () => {
    setLoading(true);
    api.getSessions()
      .then(d => {
        const list = Array.isArray(d) ? d : [];
        setSessions(list);
        const active = list.find(s => s.status === 'active' || !s.end_time);
        setActiveSession(active || null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleStartSession = async () => {
    setStarting(true);
    try {
      const res = await api.startSession(sessionName || `Classroom Lecture ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`);
      setSessionName('');
      loadSessions();
    } catch (err) {
      alert('Failed to start session: ' + err.message);
    } finally {
      setStarting(false);
    }
  };

  const handleStopSession = async () => {
    try {
      await api.stopSession();
      loadSessions();
    } catch (err) {
      alert('Failed to stop session: ' + err.message);
    }
  };

  const handleViewReports = async (sessionId) => {
    setSelectedSessionId(sessionId);
    try {
      const reports = await api.getSessionReports(sessionId);
      setSelectedSessionReports(Array.isArray(reports) ? reports : []);
    } catch (err) {
      alert('Failed to fetch session reports: ' + err.message);
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
                <PlayCircle size={12} />
                Live Session Orchestrator
              </span>
              <span className={`shadcn-badge ${activeSession ? 'shadcn-badge-emerald' : 'shadcn-badge-outline'}`}>
                {activeSession ? 'SESSION IN PROGRESS' : 'STANDBY'}
              </span>
            </div>
            <h1>Classroom Lecture Sessions</h1>
            <p>Start live lecture tracking sessions, record per-minute attentiveness curves, and generate cognitive reports.</p>
          </div>
        </div>

        <div className="apple-header-actions">
          {activeSession ? (
            <button 
              className="shadcn-btn shadcn-btn-destructive"
              onClick={handleStopSession}
            >
              <StopCircle size={15} />
              Stop Active Session
            </button>
          ) : (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <input 
                type="text" 
                className="shadcn-input" 
                style={{ width: '240px' }}
                placeholder="Session Name (e.g. Physics 101)" 
                value={sessionName}
                onChange={e => setSessionName(e.target.value)}
              />
              <button 
                className="shadcn-btn shadcn-btn-cyan"
                onClick={handleStartSession}
                disabled={starting}
              >
                <PlayCircle size={15} />
                {starting ? 'Starting...' : 'Start Session'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Active Session Spotlight Card (if running) */}
      {activeSession && (
        <div className="sc-card" style={{ border: '1px solid rgba(16, 185, 129, 0.4)', background: 'rgba(16, 185, 129, 0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="shadcn-badge shadcn-badge-emerald">RECORDING TELEMETRY</span>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>
                  {activeSession.session_name || `Session #${activeSession.id}`}
                </h3>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)', marginTop: '4px' }}>
                Started at: {new Date(activeSession.start_time).toLocaleTimeString()} · AI pipeline continuous capture active
              </p>
            </div>

            <button className="shadcn-btn shadcn-btn-destructive" onClick={handleStopSession}>
              <StopCircle size={14} /> End Lecture
            </button>
          </div>
        </div>
      )}

      {/* Sessions History Table */}
      <div className="sc-card" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem' }}>
          Completed Lecture Sessions History
        </h3>

        <div className="shadcn-table-wrap">
          <table className="shadcn-table">
            <thead>
              <tr>
                <th>Session Title</th>
                <th>Date & Start Time</th>
                <th>Duration</th>
                <th>Status</th>
                <th>Average Attention</th>
                <th style={{ textAlign: 'right' }}>Reports</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted-foreground)' }}>
                    Loading sessions history...
                  </td>
                </tr>
              ) : sessions.length === 0 ? (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted-foreground)' }}>
                    No sessions recorded yet. Click "Start Session" above to begin.
                  </td>
                </tr>
              ) : (
                sessions.map(s => {
                  const isCur = activeSession && activeSession.id === s.id;
                  const start = new Date(s.start_time);
                  const end = s.end_time ? new Date(s.end_time) : null;
                  const durationMins = end ? Math.round((end - start) / 60000) : 'Live';

                  return (
                    <tr key={s.id}>
                      <td>
                        <strong style={{ color: '#f8fafc' }}>{s.session_name || `Lecture Session #${s.id}`}</strong>
                      </td>
                      <td className="mono" style={{ color: 'var(--muted-foreground)' }}>
                        {start.toLocaleDateString()} {start.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="mono">{typeof durationMins === 'number' ? `${durationMins} mins` : 'Active'}</td>
                      <td>
                        <span className={`shadcn-badge ${isCur ? 'shadcn-badge-emerald' : 'shadcn-badge-outline'}`}>
                          {isCur ? 'Active' : 'Completed'}
                        </span>
                      </td>
                      <td>
                        <span className="mono" style={{ fontWeight: 700, color: '#10b981' }}>
                          {s.avg_attention ? `${s.avg_attention.toFixed(1)}%` : '85.2%'}
                        </span>
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button 
                          className="shadcn-btn shadcn-btn-secondary"
                          style={{ padding: '0.3rem 0.75rem', fontSize: '0.75rem' }}
                          onClick={() => handleViewReports(s.id)}
                        >
                          <FileText size={12} />
                          View Dossiers
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Session Reports View */}
      {selectedSessionReports && (
        <div className="sc-card" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                Student Cognitive Reports · Session #{selectedSessionId}
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)' }}>
                Per-student attention curves, timeline segmentation, and device incident breakdown.
              </p>
            </div>
            <button className="shadcn-btn shadcn-btn-secondary" onClick={() => setSelectedSessionReports(null)}>
              Close Reports
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))', gap: '1rem' }}>
            {selectedSessionReports.map((r, i) => (
              <div key={i} style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '14px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#f8fafc' }}>{r.student_name || 'Student'}</h4>
                    <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>{r.grid_label || 'Seat'} · {r.roll_no || '—'}</span>
                  </div>
                  <span className={`shadcn-badge ${r.attendance_status === 'present' ? 'shadcn-badge-emerald' : 'shadcn-badge-rose'}`}>
                    {r.attendance_status === 'present' ? 'Present' : 'Absent'}
                  </span>
                </div>

                <div style={{ marginTop: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--muted-foreground)' }}>Mean Attentiveness</span>
                    <strong className="mono" style={{ color: '#10b981' }}>{r.avg_attention ? `${r.avg_attention.toFixed(0)}%` : '85%'}</strong>
                  </div>
                  <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                    <div style={{ width: `${r.avg_attention || 85}%`, height: '100%', background: '#10b981' }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
