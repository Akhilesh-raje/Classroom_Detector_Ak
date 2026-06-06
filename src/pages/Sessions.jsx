import React, { useState, useEffect } from 'react';
import { api } from '../api';

const S = {
  page:   { padding:'24px', color:'#e2e8f0' },
  header: { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'24px' },
  title:  { fontSize:'1.4rem', fontWeight:700, color:'#f1f5f9' },
  row:    { display:'flex', gap:'12px', alignItems:'center' },
  btn:    { padding:'8px 18px', borderRadius:'8px', border:'none', cursor:'pointer',
            background:'#3b82f6', color:'#fff', fontWeight:600, fontSize:'0.85rem' },
  btnRed: { padding:'8px 18px', borderRadius:'8px', border:'none', cursor:'pointer',
            background:'#ef4444', color:'#fff', fontWeight:600, fontSize:'0.85rem' },
  btnGray:{ padding:'8px 18px', borderRadius:'8px', border:'none', cursor:'pointer',
            background:'#334155', color:'#94a3b8', fontWeight:600, fontSize:'0.85rem' },
  card:   { background:'#1e293b', borderRadius:'12px', padding:'20px', marginBottom:'16px' },
  grid:   { display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))', gap:'16px' },
  rcard:  { background:'#0f172a', borderRadius:'10px', padding:'16px', border:'1px solid #1e293b' },
  badge:  { padding:'2px 8px', borderRadius:'999px', fontSize:'0.72rem', fontWeight:600 },
  bar:    { height:'6px', borderRadius:'3px', background:'#1e293b', overflow:'hidden', marginTop:'4px' },
  input:  { padding:'8px 12px', borderRadius:'8px', border:'1px solid #334155',
            background:'#0f172a', color:'#e2e8f0', fontSize:'0.85rem' },
};

function AttnBar({ value }) {
  const col = value >= 75 ? '#22c55e' : value >= 55 ? '#eab308' : value >= 35 ? '#f97316' : '#ef4444';
  return (
    <div style={S.bar}>
      <div style={{ width:`${Math.min(value,100)}%`, height:'100%', background:col, transition:'width 0.5s' }} />
    </div>
  );
}

function ReportCard({ report }) {
  const elec = report.electronics || {};
  const tl   = report.activity_timeline || [];
  const actCounts = tl.reduce((acc, seg) => {
    acc[seg.activity] = (acc[seg.activity] || 0) + seg.duration;
    return acc;
  }, {});
  const totalDur = Object.values(actCounts).reduce((a,b) => a+b, 0) || 1;

  const ACT_COL = {
    studying:'#22c55e', attentive:'#22c55e', neutral:'#eab308',
    distracted:'#f97316', drowsy:'#ef4444', phone:'#ef4444',
    laptop:'#06b6d4', talking:'#eab308', fidgeting:'#f97316',
  };

  return (
    <div style={S.rcard}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start',marginBottom:'10px'}}>
        <div>
          <div style={{fontWeight:700,color:'#f1f5f9',fontSize:'0.95rem'}}>{report.student_name || 'Unknown'}</div>
          <div style={{fontSize:'0.75rem',color:'#64748b'}}>{report.grid_label} · {report.roll_no || '—'}</div>
        </div>
        <span style={{...S.badge,
          background: report.attendance_status === 'present' ? '#16a34a22' : '#dc262622',
          color:       report.attendance_status === 'present' ? '#4ade80'   : '#f87171',
          border:      `1px solid ${report.attendance_status === 'present' ? '#16a34a' : '#dc2626'}`,
        }}>
          {report.attendance_status === 'present' ? '✓ Present' : '✗ Absent'}
        </span>
      </div>

      {/* Attention */}
      <div style={{marginBottom:'10px'}}>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.78rem',color:'#94a3b8'}}>
          <span>Attention</span>
          <span style={{color:'#f1f5f9',fontWeight:600}}>{report.avg_attention?.toFixed(1)}%</span>
        </div>
        <AttnBar value={report.avg_attention || 0} />
        <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.7rem',color:'#475569',marginTop:'2px'}}>
          <span>Peak: {report.peak_attention?.toFixed(1)}%</span>
          <span>Low: {report.low_attention?.toFixed(1)}%</span>
        </div>
      </div>

      {/* Activity breakdown */}
      {Object.keys(actCounts).length > 0 && (
        <div style={{marginBottom:'10px'}}>
          <div style={{fontSize:'0.75rem',color:'#64748b',marginBottom:'6px'}}>Activity Breakdown</div>
          {Object.entries(actCounts).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([act,dur]) => (
            <div key={act} style={{marginBottom:'4px'}}>
              <div style={{display:'flex',justifyContent:'space-between',fontSize:'0.72rem'}}>
                <span style={{color: ACT_COL[act] || '#94a3b8'}}>{act}</span>
                <span style={{color:'#64748b'}}>{(dur/totalDur*100).toFixed(0)}% · {dur.toFixed(0)}s</span>
              </div>
              <div style={{height:'3px',borderRadius:'2px',background:'#1e293b',overflow:'hidden'}}>
                <div style={{width:`${dur/totalDur*100}%`,height:'100%',background: ACT_COL[act] || '#334155'}} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Electronics */}
      <div style={{display:'flex',gap:'8px',flexWrap:'wrap'}}>
        {elec.phone_duration_seconds > 0 && (
          <span style={{...S.badge,background:'#dc262622',color:'#f87171',border:'1px solid #dc2626'}}>
            📱 {elec.phone_duration_seconds.toFixed(0)}s
          </span>
        )}
        {elec.laptop_duration_seconds > 0 && (
          <span style={{...S.badge,background:'#0891b222',color:'#22d3ee',border:'1px solid #0891b2'}}>
            💻 {elec.laptop_duration_seconds.toFixed(0)}s
          </span>
        )}
        {!elec.phone_duration_seconds && !elec.laptop_duration_seconds && (
          <span style={{...S.badge,background:'#33415522',color:'#64748b'}}>No electronics</span>
        )}
      </div>
    </div>
  );
}

export default function Sessions() {
  const [sessions, setSessions]     = useState([]);
  const [selected, setSelected]     = useState(null);
  const [reports, setReports]       = useState([]);
  const [loading, setLoading]       = useState(false);
  const [sessionName, setName]      = useState('');
  const [activeSession, setActive]  = useState(null);
  const [msg, setMsg]               = useState('');

  const flash = (m, err=false) => { setMsg({text:m,err}); setTimeout(()=>setMsg(''),4000); };

  const loadSessions = () => {
    api.getSessions().then(d => {
      const arr = Array.isArray(d) ? d : [];
      setSessions(arr);
      const act = arr.find(s => s.status === 'active');
      setActive(act || null);
    }).catch(() => {});
  };

  useEffect(() => { loadSessions(); }, []);

  const handleStart = async () => {
    const r = await api.startSession(sessionName);
    if (r.session_id) { flash('Session started!'); setName(''); loadSessions(); }
    else flash(r.detail || 'Error', true);
  };

  const handleStop = async () => {
    const r = await api.stopSession();
    if (r.session_id) { flash('Session stopped. Reports generated.'); loadSessions(); }
    else flash(r.detail || 'Error', true);
  };

  const handleSelect = async (session) => {
    setSelected(session);
    setLoading(true);
    try {
      const r = await api.getSessionReports(session.session_id);
      setReports(Array.isArray(r) ? r : []);
    } catch { setReports([]); }
    setLoading(false);
  };

  const fmt = (iso) => iso ? new Date(iso).toLocaleString() : '—';
  const dur = (s) => {
    if (!s.start_time || !s.end_time) return '—';
    const d = (new Date(s.end_time) - new Date(s.start_time)) / 1000;
    return d < 60 ? `${d.toFixed(0)}s` : `${(d/60).toFixed(1)}m`;
  };

  return (
    <div style={S.page}>
      {msg && (
        <div style={{background: msg.err ? '#dc262622':'#22c55e22', border:`1px solid ${msg.err?'#dc2626':'#22c55e'}`,
          borderRadius:'8px', padding:'10px 16px', marginBottom:'16px', color: msg.err?'#f87171':'#22c55e'}}>
          {msg.text}
        </div>
      )}

      <div style={S.header}>
        <h1 style={S.title}>Sessions</h1>
        <div style={S.row}>
          {!activeSession ? (
            <>
              <input style={S.input} placeholder="Session name (optional)" value={sessionName}
                onChange={e => setName(e.target.value)} />
              <button style={S.btn} onClick={handleStart}>▶ Start Session</button>
            </>
          ) : (
            <>
              <span style={{...S.badge,background:'#16a34a22',color:'#4ade80',border:'1px solid #16a34a',padding:'6px 12px'}}>
                ● LIVE: {activeSession.session_name}
              </span>
              <button style={S.btnRed} onClick={handleStop}>■ Stop Session</button>
            </>
          )}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'320px 1fr',gap:'20px'}}>
        {/* Session list */}
        <div>
          <div style={{fontSize:'0.75rem',color:'#64748b',marginBottom:'10px',textTransform:'uppercase',letterSpacing:'0.05em'}}>
            Past Sessions ({sessions.length})
          </div>
          {sessions.length === 0 && (
            <div style={{...S.card,textAlign:'center',color:'#64748b',padding:'30px'}}>No sessions yet</div>
          )}
          {sessions.map(s => (
            <div key={s.session_id}
              style={{...S.card, cursor:'pointer', border: selected?.session_id === s.session_id ? '1px solid #3b82f6' : '1px solid transparent',
                transition:'border 0.15s'}}
              onClick={() => handleSelect(s)}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
                <div style={{fontWeight:600,color:'#f1f5f9',fontSize:'0.9rem'}}>{s.session_name}</div>
                <span style={{...S.badge,
                  background: s.status==='active' ? '#16a34a22':'#33415522',
                  color:       s.status==='active' ? '#4ade80':'#64748b',
                }}>
                  {s.status === 'active' ? '● LIVE' : 'Done'}
                </span>
              </div>
              <div style={{fontSize:'0.75rem',color:'#64748b',marginTop:'6px'}}>{fmt(s.start_time)}</div>
              <div style={{display:'flex',gap:'12px',marginTop:'8px',fontSize:'0.78rem',color:'#94a3b8'}}>
                <span>👥 {s.student_count || 0}</span>
                <span>⏱ {dur(s)}</span>
                <span>📊 {s.avg_class_attention?.toFixed(1) || '—'}%</span>
              </div>
            </div>
          ))}
        </div>

        {/* Reports panel */}
        <div>
          {!selected ? (
            <div style={{...S.card,textAlign:'center',color:'#64748b',padding:'60px'}}>
              Select a session to view student reports
            </div>
          ) : loading ? (
            <div style={{...S.card,textAlign:'center',color:'#64748b',padding:'60px'}}>Loading reports...</div>
          ) : reports.length === 0 ? (
            <div style={{...S.card,textAlign:'center',color:'#64748b',padding:'60px'}}>No reports for this session</div>
          ) : (
            <>
              <div style={{fontSize:'0.75rem',color:'#64748b',marginBottom:'12px',textTransform:'uppercase',letterSpacing:'0.05em'}}>
                {selected.session_name} — {reports.length} student(s)
              </div>
              <div style={S.grid}>
                {reports.map((r,i) => <ReportCard key={i} report={r} />)}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
