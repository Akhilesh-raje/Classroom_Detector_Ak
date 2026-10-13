import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { 
  CheckCircle2, 
  AlertCircle, 
  HelpCircle, 
  Users, 
  Clock, 
  Download, 
  RotateCcw, 
  UserCheck, 
  UserX, 
  ShieldCheck, 
  Sparkles, 
  Search, 
  Check, 
  X, 
  Eye, 
  ChevronRight,
  Filter,
  Sliders,
  Radio
} from 'lucide-react';

export default function Attendance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('ALL'); // 'ALL' | 'VERIFIED' | 'REVIEW' | 'ABSENT'
  const [search, setSearch] = useState('');
  const [dryRun, setDryRun] = useState(false);
  const [selectedReview, setSelectedReview] = useState(null);
  const [studentsList, setStudentsList] = useState([]);

  const loadData = () => {
    setLoading(true);
    Promise.all([
      api.getAttendance(),
      api.getStudents()
    ])
      .then(([att, studs]) => {
        setData(att);
        setDryRun(att?.dryRunMode ?? false);
        setStudentsList(Array.isArray(studs) ? studs : []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => {
      api.getAttendance().then(setData).catch(() => {});
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  const handleReviewAction = async (recordId, decision, assignedId = null) => {
    try {
      await api.reviewAttendance(recordId, decision, assignedId);
      setSelectedReview(null);
      loadData();
    } catch (e) {
      alert('Review action failed: ' + e.message);
    }
  };

  const handleManualToggle = async (studentId, currentStatus) => {
    try {
      const nextStatus = currentStatus === 'VERIFIED' ? 'absent' : 'present';
      await api.markManual(studentId, nextStatus);
      loadData();
    } catch (e) {
      alert('Manual update failed: ' + e.message);
    }
  };

  const [syncing, setSyncing] = useState(false);

  const handleSyncLiveFeed = async () => {
    setSyncing(true);
    try {
      await api.syncLiveFeed(5, dryRun);
      loadData();
    } catch (e) {
      alert('Live sync failed: ' + e.message);
    } finally {
      setSyncing(false);
    }
  };

  const handleResetAttendance = async () => {
    if (!window.confirm('Reset today\'s attendance records for testing?')) return;
    try {
      await api.resetAttendance(null, dryRun);
      loadData();
    } catch (e) {
      alert('Reset failed: ' + e.message);
    }
  };

  const counts = data?.counts || {
    detected: 0,
    verified: 0,
    review: 0,
    absent: 0,
    totalEnrolled: 12
  };

  const records = data?.records || [];
  const reviewQueue = data?.reviewQueue || [];
  const recentEvents = data?.recentEvents || [];

  const filteredRecords = records.filter(r => {
    if (filter === 'VERIFIED' && r.status !== 'VERIFIED') return false;
    if (filter === 'REVIEW' && r.status !== 'PENDING_REVIEW') return false;
    if (filter === 'ABSENT' && r.status !== 'ABSENT') return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        r.name.toLowerCase().includes(q) ||
        (r.rollNo && r.rollNo.toLowerCase().includes(q)) ||
        (r.assignedSeat && r.assignedSeat.toLowerCase().includes(q))
      );
    }
    return true;
  });

  // Physical 12-seat map lookup
  const seatMap = {};
  records.forEach(r => {
    if (r.detectedSeat) seatMap[r.detectedSeat] = r;
    else if (r.assignedSeat && !seatMap[r.assignedSeat]) seatMap[r.assignedSeat] = r;
  });
  reviewQueue.forEach(rq => {
    if (rq.seatId) {
      seatMap[rq.seatId] = {
        name: rq.candidateName !== 'None' ? `Candidate: ${rq.candidateName}` : 'Unknown Occupant',
        status: 'PENDING_REVIEW',
        detectedSeat: rq.seatId,
        reviewItem: rq
      };
    }
  });

  const R1_SEATS = ['R1C1', 'R1C2', 'R1C3', 'R1C4', 'R1C5', 'R1C6'];
  const R2_SEATS = ['R2C1', 'R2C2', 'R2C3', 'R2C4', 'R2C5', 'R2C6'];

  return (
    <div className="apple-command-root" style={{ gap: '1.75rem' }}>
      {/* Top Banner Header */}
      <div className="apple-header-card">
        <div className="apple-header-left">
          <div className="header-title-block">
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span className="shadcn-badge shadcn-badge-cyan">
                <ShieldCheck size={12} />
                Attendance Decision Engine v1
              </span>
              <span className={`shadcn-badge ${dryRun ? 'shadcn-badge-amber' : 'shadcn-badge-emerald'}`}>
                {dryRun ? 'DRY RUN MODE' : 'LIVE PERSISTENCE ACTIVE'}
              </span>
            </div>
            <h1>Live Classroom Attendance Command Center</h1>
            <p>Identity verification, temporal confirmation, candidate seat correlation, and teacher review audit trail.</p>
          </div>
        </div>

        <div className="apple-header-actions">
          <button 
            className="shadcn-btn shadcn-btn-primary"
            onClick={handleSyncLiveFeed}
            disabled={syncing}
            title="Scan active live classroom video stream to detect students"
          >
            <Radio size={14} className={syncing ? "animate-pulse" : ""} />
            {syncing ? 'Scanning Live...' : 'Sync Video Detections'}
          </button>

          <button 
            className={`shadcn-btn ${dryRun ? 'shadcn-btn-amber' : 'shadcn-btn-secondary'}`}
            onClick={() => setDryRun(!dryRun)}
            title="Toggle between dry run testing and official attendance persistence"
          >
            <Sliders size={14} />
            Mode: {dryRun ? 'Dry Run' : 'Live Official'}
          </button>

          <a 
            href={api.exportAttendanceUrl()} 
            target="_blank" 
            rel="noreferrer"
            className="shadcn-btn shadcn-btn-secondary"
            download
          >
            <Download size={14} />
            Export CSV
          </a>

          <button 
            className="shadcn-btn shadcn-btn-secondary"
            onClick={handleResetAttendance}
            title="Reset today's attendance for a fresh lecture session"
          >
            <RotateCcw size={14} />
            Reset Day
          </button>
        </div>
      </div>

      {/* Recommended 3-Metric KPI Grid */}
      <div className="apple-kpi-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">TOTAL DETECTED PRESENCE</span>
            <span className="kpi-metric" style={{ color: '#38bdf8' }}>{counts.detected}</span>
            <span className="kpi-footnote">Physical classroom presence</span>
          </div>
          <Eye size={28} color="#38bdf8" opacity={0.6} />
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">VERIFIED CHECK-INS</span>
            <span className="kpi-metric" style={{ color: '#10b981' }}>{counts.verified}</span>
            <span className="kpi-footnote" style={{ color: '#10b981' }}>Biometrically & temporally confirmed</span>
          </div>
          <CheckCircle2 size={28} color="#10b981" opacity={0.6} />
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">NEEDS REVIEW (CANDIDATES)</span>
            <span className="kpi-metric" style={{ color: '#f59e0b' }}>{counts.review}</span>
            <span className="kpi-footnote" style={{ color: '#f59e0b' }}>Seat candidate / unknown occupant</span>
          </div>
          <AlertCircle size={28} color="#f59e0b" opacity={0.6} />
        </div>

        <div className="apple-kpi-card">
          <div className="kpi-info">
            <span className="kpi-label">ABSENT ENROLLED</span>
            <span className="kpi-metric" style={{ color: '#ef4444' }}>{counts.absent}</span>
            <span className="kpi-footnote" style={{ color: 'var(--muted-foreground)' }}>Out of {counts.totalEnrolled} enrolled</span>
          </div>
          <UserX size={28} color="#ef4444" opacity={0.6} />
        </div>
      </div>

      {/* Main Split: 2x6 Physical Seating Map & Recent Events Ticker */}
      <div className="vision-hud-split" style={{ gridTemplateColumns: '1.8fr 1.2fr' }}>
        {/* Physical Seating Map (2x6 Grid) */}
        <div className="sc-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
                Physical Classroom Seating Map (2×6 Matrix)
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
                Green = Verified identity · Amber = Needs Review (Candidate/Unknown) · Grey = Vacant
              </p>
            </div>
            <span className="shadcn-badge shadcn-badge-outline">▲ PODIUM / CAMERA</span>
          </div>

          {/* Row 1: Front Bench */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-cyan)', letterSpacing: '0.04em' }}>
              ROW 1 (FRONT BENCH · R1C1 - R1C6)
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '10px' }}>
              {R1_SEATS.map(seatId => {
                const item = seatMap[seatId];
                const isVerified = item?.status === 'VERIFIED';
                const isReview   = item?.status === 'PENDING_REVIEW';

                return (
                  <div 
                    key={seatId}
                    style={{
                      padding: '12px 8px',
                      borderRadius: 'var(--radius-md)',
                      background: isVerified ? 'rgba(16, 185, 129, 0.12)' : isReview ? 'rgba(245, 158, 11, 0.12)' : 'rgba(15, 23, 42, 0.7)',
                      border: isVerified ? '1.5px solid #10b981' : isReview ? '1.5px solid #f59e0b' : '1px solid var(--border)',
                      textAlign: 'center',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      cursor: isReview ? 'pointer' : 'default',
                      boxShadow: isReview ? '0 0 12px rgba(245, 158, 11, 0.2)' : 'none'
                    }}
                    onClick={() => item?.reviewItem && setSelectedReview(item.reviewItem)}
                  >
                    <span className="mono" style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
                      {seatId}
                    </span>
                    {item?.faceImage ? (
                      <img 
                        src={item.faceImage} 
                        alt={item.name} 
                        style={{ width: '28px', height: '28px', borderRadius: '50%', objectFit: 'cover', margin: '2px auto', border: '1px solid rgba(255,255,255,0.2)' }} 
                      />
                    ) : (
                      <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'rgba(255,255,255,0.06)', margin: '2px auto' }} />
                    )}
                    <strong style={{ fontSize: '0.75rem', color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {item ? item.name : 'Vacant'}
                    </strong>
                    <span style={{ 
                      fontSize: '0.62rem', 
                      fontWeight: 700,
                      color: isVerified ? '#10b981' : isReview ? '#f59e0b' : 'var(--muted-foreground)'
                    }}>
                      {isVerified ? 'VERIFIED' : isReview ? 'REVIEW' : 'VACANT'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Row 2: Back Bench */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-violet)', letterSpacing: '0.04em' }}>
              ROW 2 (BACK BENCH · R2C1 - R2C6)
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '10px' }}>
              {R2_SEATS.map(seatId => {
                const item = seatMap[seatId];
                const isVerified = item?.status === 'VERIFIED';
                const isReview   = item?.status === 'PENDING_REVIEW';

                return (
                  <div 
                    key={seatId}
                    style={{
                      padding: '12px 8px',
                      borderRadius: 'var(--radius-md)',
                      background: isVerified ? 'rgba(16, 185, 129, 0.12)' : isReview ? 'rgba(245, 158, 11, 0.12)' : 'rgba(15, 23, 42, 0.7)',
                      border: isVerified ? '1.5px solid #10b981' : isReview ? '1.5px solid #f59e0b' : '1px solid var(--border)',
                      textAlign: 'center',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                      cursor: isReview ? 'pointer' : 'default',
                      boxShadow: isReview ? '0 0 12px rgba(245, 158, 11, 0.2)' : 'none'
                    }}
                    onClick={() => item?.reviewItem && setSelectedReview(item.reviewItem)}
                  >
                    <span className="mono" style={{ fontSize: '0.72rem', fontWeight: 800, color: 'var(--accent-violet)' }}>
                      {seatId}
                    </span>
                    {item?.faceImage ? (
                      <img 
                        src={item.faceImage} 
                        alt={item.name} 
                        style={{ width: '28px', height: '28px', borderRadius: '50%', objectFit: 'cover', margin: '2px auto', border: '1px solid rgba(255,255,255,0.2)' }} 
                      />
                    ) : (
                      <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'rgba(255,255,255,0.06)', margin: '2px auto' }} />
                    )}
                    <strong style={{ fontSize: '0.75rem', color: '#f8fafc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {item ? item.name : 'Vacant'}
                    </strong>
                    <span style={{ 
                      fontSize: '0.62rem', 
                      fontWeight: 700,
                      color: isVerified ? '#10b981' : isReview ? '#f59e0b' : 'var(--muted-foreground)'
                    }}>
                      {isVerified ? 'VERIFIED' : isReview ? 'REVIEW' : 'VACANT'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right: Recent Events Ticker */}
        <div className="sc-card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.02rem', fontWeight: 700, color: '#f8fafc' }}>
              Recent Check-In Events
            </h3>
            <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--muted-foreground)' }}>
              Real-time audit
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto', maxHeight: '310px' }}>
            {recentEvents.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted-foreground)', fontSize: '0.82rem' }}>
                No events recorded today yet.
              </div>
            ) : (
              recentEvents.map((evt, idx) => {
                const isVerified = evt.status === 'VERIFIED';
                return (
                  <div 
                    key={idx}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-sm)',
                      background: isVerified ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
                      border: isVerified ? '1px solid rgba(16, 185, 129, 0.25)' : '1px solid rgba(245, 158, 11, 0.25)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px'
                    }}
                  >
                    {isVerified ? (
                      <CheckCircle2 size={16} color="#10b981" />
                    ) : (
                      <AlertCircle size={16} color="#f59e0b" />
                    )}

                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <strong style={{ fontSize: '0.8rem', color: isVerified ? '#f8fafc' : '#fbbf24' }}>
                          {isVerified ? 'Verified check-in' : 'Identity needs review'}
                        </strong>
                        <span className="mono" style={{ fontSize: '0.68rem', color: 'var(--muted-foreground)' }}>
                          {evt.time}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)', marginTop: '2px' }}>
                        {isVerified 
                          ? `${evt.studentName} · Seat ${evt.seat} (${evt.method || 'Biometric'})`
                          : `Person detected at Seat ${evt.seat} · No high-confidence student match`
                        }
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Review Queue Drawer (if items pending review) */}
      {reviewQueue.length > 0 && (
        <div className="sc-card" style={{ border: '1px solid rgba(245, 158, 11, 0.35)', background: 'rgba(245, 158, 11, 0.05)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={18} color="#f59e0b" />
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
                Pending Review Queue ({reviewQueue.length} Unresolved Presences)
              </h3>
            </div>
            <span style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
              Teacher confirmation required before marking official attendance
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
            {reviewQueue.map(rq => (
              <div 
                key={rq.id} 
                style={{ 
                  background: 'rgba(15, 23, 42, 0.85)', 
                  border: '1px solid var(--border)', 
                  borderRadius: 'var(--radius-md)', 
                  padding: '14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <strong style={{ fontSize: '0.88rem', color: '#f8fafc' }}>
                      Desk {rq.seatId || 'Unknown'} Occupant
                    </strong>
                    <div style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>
                      Detected at: {rq.detectedAt ? new Date(rq.detectedAt).toLocaleTimeString() : 'Recent'}
                    </div>
                  </div>
                  <span className="shadcn-badge shadcn-badge-amber">REVIEW REQUIRED</span>
                </div>

                <div style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
                  Candidate Student: <strong style={{ color: '#38bdf8' }}>{rq.candidateName}</strong>
                </div>

                <div style={{ display: 'flex', gap: '6px', marginTop: '6px' }}>
                  <button 
                    className="shadcn-btn shadcn-btn-emerald"
                    style={{ flex: 1, padding: '0.35rem 0.6rem', fontSize: '0.75rem' }}
                    onClick={() => handleReviewAction(rq.id, 'APPROVE')}
                  >
                    <Check size={12} /> Confirm Candidate
                  </button>
                  <button 
                    className="shadcn-btn shadcn-btn-destructive"
                    style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem' }}
                    onClick={() => handleReviewAction(rq.id, 'REJECT')}
                  >
                    <X size={12} /> Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Full Student Roster & Attendance Status Table */}
      <div className="sc-card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
              Enrolled Students Attendance Register
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)' }}>
              Official presence log with biometric verification confidence and check-in method.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div className="cmd-search-input-wrap">
              <Search size={14} color="var(--muted-foreground)" />
              <input 
                type="text" 
                placeholder="Search student or seat..." 
                value={search} 
                onChange={e => setSearch(e.target.value)} 
              />
            </div>

            <div className="segmented-control">
              {['ALL', 'VERIFIED', 'REVIEW', 'ABSENT'].map(st => (
                <button 
                  key={st}
                  className={`segmented-item ${filter === st ? 'active' : ''}`}
                  onClick={() => setFilter(st)}
                >
                  {st}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="shadcn-table-wrap">
          <table className="shadcn-table">
            <thead>
              <tr>
                <th>Student</th>
                <th>Roll Number</th>
                <th>Assigned Seat</th>
                <th>Detected Seat</th>
                <th>Verification State</th>
                <th>Method</th>
                <th>Confidence</th>
                <th>Check-in Time</th>
                <th style={{ textAlign: 'right' }}>Manual Override</th>
              </tr>
            </thead>
            <tbody>
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan="9" style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--muted-foreground)' }}>
                    No students matching filter.
                  </td>
                </tr>
              ) : (
                filteredRecords.map(r => {
                  const isVerified = r.status === 'VERIFIED';
                  const isReview   = r.status === 'PENDING_REVIEW';

                  return (
                    <tr key={r.studentId}>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          {r.faceImage ? (
                            <img 
                              src={r.faceImage} 
                              alt={r.name} 
                              style={{ width: '32px', height: '32px', borderRadius: '50%', objectFit: 'cover', border: '1.5px solid rgba(56, 189, 248, 0.4)' }} 
                            />
                          ) : (
                            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '0.75rem' }}>
                              {r.name.slice(0, 2).toUpperCase()}
                            </div>
                          )}
                          <div>
                            <strong style={{ color: '#f8fafc', display: 'block', fontSize: '0.85rem' }}>{r.name}</strong>
                            <span style={{ fontSize: '0.7rem', color: 'var(--muted-foreground)' }}>Section {r.section || 'CS-A'}</span>
                          </div>
                        </div>
                      </td>
                      <td className="mono" style={{ color: 'var(--muted-foreground)' }}>{r.rollNo}</td>
                      <td className="mono" style={{ color: 'var(--accent-cyan)' }}>{r.assignedSeat || '—'}</td>
                      <td className="mono" style={{ color: r.detectedSeat ? 'var(--accent-violet)' : 'var(--muted-foreground)' }}>
                        {r.detectedSeat || '—'}
                      </td>
                      <td>
                        <span className={`shadcn-badge ${isVerified ? 'shadcn-badge-emerald' : isReview ? 'shadcn-badge-amber' : 'shadcn-badge-rose'}`}>
                          {isVerified ? 'VERIFIED' : isReview ? 'PENDING REVIEW' : 'ABSENT'}
                        </span>
                      </td>
                      <td style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)' }}>
                        {r.method === 'biometric_verified' ? 'Biometric Face' : r.method === 'seat_candidate' ? 'Seat Candidate' : r.method === 'teacher_approved' ? 'Teacher Approved' : r.method || '—'}
                      </td>
                      <td className="mono" style={{ color: isVerified ? '#10b981' : 'var(--muted-foreground)' }}>
                        {r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="mono" style={{ fontSize: '0.75rem', color: 'var(--muted-foreground)' }}>
                        {r.markedAt ? new Date(r.markedAt).toLocaleTimeString() : '—'}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        <button 
                          className={`shadcn-btn ${isVerified ? 'shadcn-btn-secondary' : 'shadcn-btn-cyan'}`}
                          style={{ padding: '0.25rem 0.65rem', fontSize: '0.72rem' }}
                          onClick={() => handleManualToggle(r.studentId, r.status)}
                        >
                          {isVerified ? 'Mark Absent' : 'Mark Present'}
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
    </div>
  );
}
