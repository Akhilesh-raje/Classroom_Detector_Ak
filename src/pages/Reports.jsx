import React, { useState, useEffect } from 'react';
import { api } from '../api';

const S = {
  page:    { padding:'24px', color:'#e2e8f0' },
  header:  { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'24px' },
  title:   { fontSize:'1.4rem', fontWeight:700, color:'#f1f5f9' },
  row:     { display:'flex', gap:'10px', alignItems:'center', flexWrap:'wrap' },
  btn:     { padding:'8px 16px', borderRadius:'8px', border:'none', cursor:'pointer',
             background:'#3b82f6', color:'#fff', fontWeight:600, fontSize:'0.82rem' },
  btnGray: { padding:'8px 16px', borderRadius:'8px', border:'none', cursor:'pointer',
             background:'#334155', color:'#94a3b8', fontWeight:600, fontSize:'0.82rem' },
  btnGreen:{ padding:'8px 16px', borderRadius:'8px', border:'none', cursor:'pointer',
             background:'#16a34a', color:'#fff', fontWeight:600, fontSize:'0.82rem' },
  card:    { background:'#1e293b', borderRadius:'12px', padding:'20px', marginBottom:'16px' },
  select:  { padding:'8px 12px', borderRadius:'8px', border:'1px solid #334155',
             background:'#0f172a', color:'#e2e8f0', fontSize:'0.85rem', minWidth:'220px' },
  badge:   { padding:'2px 8px', borderRadius:'999px', fontSize:'0.72rem', fontWeight:600 },
  grid:    { display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(300px,1fr))', gap:'16px' },
};

const ACT_COL = {
  studying:'#22c55e', attentive:'#22c55e', neutral:'#eab308',
  distracted:'#f97316', drowsy:'#ef4444', phone:'#ef4444',
  laptop:'#06b6d4', talking:'#eab308', fidgeting:'#f97316', music:'#a855f7',
};

function CircleGauge({ value, size = 80 }) {
  const r   = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const fill = circ * Math.min(value, 100) / 100;
  const col  = value >= 75 ? '#22c55e' : value >= 55 ? '#eab308' : value >= 35 ? '#f97316' : '#ef4444';
  return (
    <svg width={size} height={size} style={{ transform:'rotate(-90deg)' }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1e293b" strokeWidth={8} />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={col} strokeWidth={8}
        strokeDasharray={`${fill} ${circ}`} strokeLinecap="round"
        style={{ transition:'stroke-dasharray 0.6s ease' }} />
      <text x={size/2} y={size/2} textAnchor="middle" dominantBaseline="middle"
        fill={col} fontSize="14" fontWeight="700"
        style={{ transform:`rotate(90deg) translate(0px, -${size}px)` }}>
        {value?.toFixed(0)}%
      </text>
    </svg>
  );
}

function Timeline({ segments, duration }) {
  if (!segments || segments.length === 0) return null;
  return (
    <div style={{ marginTop:'8px' }}>
      <div style={{ fontSize:'0.72rem', color:'#64748b', marginBottom:'4px' }}>Activity Timeline</div>
      <div style={{ display:'flex', height:'16px', borderRadius:'4px', overflow:'hidden', gap:'1px' }}>
        {segments.map((seg, i) => {
          const w = ((seg.duration || 0) / Math.max(duration, 1)) * 100;
          const col = ACT_COL[seg.activity] || '#334155';
          return (
            <div key={i} title={`${seg.activity} (${seg.duration?.toFixed(0)}s)`}
              style={{ width:`${w}%`, background:col, minWidth:'2px', transition:'width 0.3s' }} />
          );
        })}
      </div>
      <div style={{ display:'flex', gap:'8px', flexWrap:'wrap', marginTop:'6px' }}>
        {[...new Set(segments.map(s => s.activity))].map(act => (
          <div key={act} style={{ display:'flex', alignItems:'center', gap:'4px', fontSize:'0.68rem', color:'#94a3b8' }}>
            <div style={{ width:8, height:8, borderRadius:'2px', background: ACT_COL[act]||'#334155' }} />
            {act}
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportCard({ report, onAiAnalyze }) {
  const [expanded, setExpanded] = useState(false);
  const elec = report.electronics || {};
  const tl   = report.activity_timeline || [];
  const dur  = report.session_duration_seconds || 1;

  const actCounts = tl.reduce((acc, seg) => {
    acc[seg.activity] = (acc[seg.activity] || 0) + (seg.duration || 0);
    return acc;
  }, {});

  return (
    <div style={{ background:'#0f172a', borderRadius:'10px', padding:'16px',
      border:'1px solid #1e293b', transition:'border 0.15s' }}>

      {/* Header */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:'12px' }}>
        <div>
          <div style={{ fontWeight:700, color:'#f1f5f9', fontSize:'0.95rem' }}>
            {report.student_name || 'Unknown'}
          </div>
          <div style={{ fontSize:'0.72rem', color:'#64748b', marginTop:'2px' }}>
            {report.grid_label} · {report.roll_no || '—'} · {report.session_date}
          </div>
        </div>
        <span style={{ ...S.badge,
          background: report.attendance_status === 'present' ? '#16a34a22' : '#dc262622',
          color:       report.attendance_status === 'present' ? '#4ade80'   : '#f87171',
          border:      `1px solid ${report.attendance_status === 'present' ? '#16a34a' : '#dc2626'}`,
        }}>
          {report.attendance_status === 'present' ? '✓ Present' : '✗ Absent'}
        </span>
      </div>

      {/* Gauge + stats */}
      <div style={{ display:'flex', gap:'16px', alignItems:'center', marginBottom:'12px' }}>
        <CircleGauge value={report.avg_attention || 0} />
        <div style={{ flex:1 }}>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'6px' }}>
            {[['Peak', report.peak_attention], ['Low', report.low_attention],
              ['Duration', `${(dur/60).toFixed(1)}m`],
              ['Activities', tl.length]].map(([k,v]) => (
              <div key={k} style={{ background:'#1e293b', borderRadius:'6px', padding:'6px 10px' }}>
                <div style={{ fontSize:'0.68rem', color:'#64748b' }}>{k}</div>
                <div style={{ fontSize:'0.85rem', fontWeight:600, color:'#f1f5f9' }}>
                  {typeof v === 'number' ? `${v.toFixed(1)}%` : v}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Activity breakdown */}
      {Object.keys(actCounts).length > 0 && (
        <div style={{ marginBottom:'10px' }}>
          {Object.entries(actCounts).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([act,d]) => (
            <div key={act} style={{ marginBottom:'4px' }}>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.72rem' }}>
                <span style={{ color: ACT_COL[act]||'#94a3b8' }}>{act}</span>
                <span style={{ color:'#64748b' }}>{(d/dur*100).toFixed(0)}% · {d.toFixed(0)}s</span>
              </div>
              <div style={{ height:'3px', borderRadius:'2px', background:'#1e293b', overflow:'hidden' }}>
                <div style={{ width:`${d/dur*100}%`, height:'100%', background: ACT_COL[act]||'#334155' }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Electronics */}
      <div style={{ display:'flex', gap:'6px', flexWrap:'wrap', marginBottom:'10px' }}>
        {elec.phone_duration_seconds > 0 && (
          <span style={{ ...S.badge, background:'#dc262622', color:'#f87171', border:'1px solid #dc2626' }}>
            📱 {elec.phone_duration_seconds.toFixed(0)}s
          </span>
        )}
        {elec.laptop_duration_seconds > 0 && (
          <span style={{ ...S.badge, background:'#0891b222', color:'#22d3ee', border:'1px solid #0891b2' }}>
            💻 {elec.laptop_duration_seconds.toFixed(0)}s
          </span>
        )}
        {!elec.phone_duration_seconds && !elec.laptop_duration_seconds && (
          <span style={{ ...S.badge, background:'#33415522', color:'#64748b' }}>No electronics</span>
        )}
      </div>

      {/* Timeline */}
      {expanded && <Timeline segments={tl} duration={dur} />}

      {/* Actions */}
      <div style={{ display:'flex', gap:'6px', marginTop:'10px' }}>
        <button style={{ ...S.btnGray, fontSize:'0.72rem', padding:'4px 10px' }}
          onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Hide Timeline' : 'Show Timeline'}
        </button>
        <button style={{ ...S.btnGreen, fontSize:'0.72rem', padding:'4px 10px' }}
          onClick={() => onAiAnalyze(report.grid_label)}>
          🤖 AI Insight
        </button>
      </div>
    </div>
  );
}

export default function Reports() {
  const [sessions, setSessions]   = useState([]);
  const [selSession, setSelSess]  = useState('');
  const [reports, setReports]     = useState([]);
  const [loading, setLoading]     = useState(false);
  const [analysis, setAnalysis]   = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [aiInsight, setAiInsight] = useState({ seat:'', text:'' });

  useEffect(() => {
    api.getSessions().then(d => {
      const arr = Array.isArray(d) ? d : [];
      setSessions(arr);
      if (arr.length > 0) setSelSess(arr[0].session_id);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selSession) return;
    setLoading(true);
    api.getSessionReports(selSession)
      .then(d => { setReports(Array.isArray(d) ? d : []); setLoading(false); })
      .catch(() => { setReports([]); setLoading(false); });
  }, [selSession]);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setAnalysis('');
    const r = await api.ollamaAnalyze().catch(() => ({ summary: 'Ollama not available.' }));
    setAnalysis(r.summary || 'No analysis.');
    setAnalyzing(false);
  };

  const handleAiInsight = async (seatId) => {
    setAiInsight({ seat: seatId, text: 'Analyzing...' });
    const r = await api.ollamaAnalyzeStudent(seatId).catch(() => ({ summary: 'Ollama not available.' }));
    setAiInsight({ seat: seatId, text: r.summary || 'No insight available.' });
  };

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(reports, null, 2)], { type:'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `session_${selSession?.slice(0,8)}_reports.json`;
    a.click();
  };

  const classAvg = reports.length
    ? (reports.reduce((s,r) => s + (r.avg_attention||0), 0) / reports.length).toFixed(1)
    : null;

  return (
    <div style={S.page}>
      <div style={S.header}>
        <h1 style={S.title}>Reports</h1>
        <div style={S.row}>
          <select style={S.select} value={selSession} onChange={e => setSelSess(e.target.value)}>
            <option value="">Select session...</option>
            {sessions.map(s => (
              <option key={s.session_id} value={s.session_id}>
                {s.session_name} — {s.start_time ? new Date(s.start_time).toLocaleDateString() : ''}
              </option>
            ))}
          </select>
          <button style={S.btnGreen} onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? 'Analyzing...' : '🤖 AI Class Analysis'}
          </button>
          {reports.length > 0 && (
            <button style={S.btnGray} onClick={exportJSON}>⬇ Export JSON</button>
          )}
        </div>
      </div>

      {/* Class summary bar */}
      {classAvg && (
        <div style={{ ...S.card, display:'flex', gap:'24px', alignItems:'center', flexWrap:'wrap' }}>
          <div>
            <div style={{ fontSize:'0.72rem', color:'#64748b' }}>Class Average</div>
            <div style={{ fontSize:'1.6rem', fontWeight:800, color:'#f1f5f9' }}>{classAvg}%</div>
          </div>
          <div>
            <div style={{ fontSize:'0.72rem', color:'#64748b' }}>Students</div>
            <div style={{ fontSize:'1.6rem', fontWeight:800, color:'#f1f5f9' }}>{reports.length}</div>
          </div>
          <div>
            <div style={{ fontSize:'0.72rem', color:'#64748b' }}>Studying</div>
            <div style={{ fontSize:'1.6rem', fontWeight:800, color:'#22c55e' }}>
              {reports.filter(r => r.activity_timeline?.slice(-1)[0]?.activity === 'studying').length}
            </div>
          </div>
          <div>
            <div style={{ fontSize:'0.72rem', color:'#64748b' }}>Phone Usage</div>
            <div style={{ fontSize:'1.6rem', fontWeight:800, color:'#ef4444' }}>
              {reports.filter(r => (r.electronics?.phone_duration_seconds||0) > 5).length}
            </div>
          </div>
          {/* Class attention bar */}
          <div style={{ flex:1, minWidth:'200px' }}>
            <div style={{ fontSize:'0.72rem', color:'#64748b', marginBottom:'4px' }}>Class Attention</div>
            <div style={{ height:'8px', borderRadius:'4px', background:'#0f172a', overflow:'hidden' }}>
              <div style={{ width:`${classAvg}%`, height:'100%',
                background: classAvg >= 75 ? '#22c55e' : classAvg >= 55 ? '#eab308' : '#ef4444',
                transition:'width 0.5s' }} />
            </div>
          </div>
        </div>
      )}

      {/* AI Analysis */}
      {analysis && (
        <div style={{ ...S.card, background:'#0f172a', border:'1px solid #1d4ed8', marginBottom:'20px' }}>
          <div style={{ fontSize:'0.75rem', color:'#60a5fa', marginBottom:'8px', fontWeight:600 }}>
            🤖 AI Classroom Analysis
          </div>
          <div style={{ fontSize:'0.88rem', color:'#cbd5e1', lineHeight:1.6 }}>{analysis}</div>
        </div>
      )}

      {/* AI Student Insight */}
      {aiInsight.text && (
        <div style={{ ...S.card, background:'#0f172a', border:'1px solid #7c3aed', marginBottom:'20px' }}>
          <div style={{ fontSize:'0.75rem', color:'#a78bfa', marginBottom:'8px', fontWeight:600 }}>
            🤖 AI Insight — Seat {aiInsight.seat}
          </div>
          <div style={{ fontSize:'0.88rem', color:'#cbd5e1', lineHeight:1.6 }}>{aiInsight.text}</div>
          <button style={{ ...S.btnGray, fontSize:'0.72rem', padding:'4px 10px', marginTop:'8px' }}
            onClick={() => setAiInsight({ seat:'', text:'' })}>Dismiss</button>
        </div>
      )}

      {/* Report cards */}
      {loading ? (
        <div style={{ ...S.card, textAlign:'center', color:'#64748b', padding:'60px' }}>Loading reports...</div>
      ) : !selSession ? (
        <div style={{ ...S.card, textAlign:'center', color:'#64748b', padding:'60px' }}>Select a session above</div>
      ) : reports.length === 0 ? (
        <div style={{ ...S.card, textAlign:'center', color:'#64748b', padding:'60px' }}>No reports for this session</div>
      ) : (
        <div style={S.grid}>
          {reports.map((r, i) => (
            <ReportCard key={i} report={r} onAiAnalyze={handleAiInsight} />
          ))}
        </div>
      )}
    </div>
  );
}
