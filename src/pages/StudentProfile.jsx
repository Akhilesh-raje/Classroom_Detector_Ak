import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, PieChart, Pie, Legend,
} from 'recharts';
import { api } from '../api';

// ── Palette & helpers ─────────────────────────────────────────────────────────
const ACT_COL = {
  studying:'#22c55e', attentive:'#4ade80', neutral:'#eab308',
  distracted:'#f97316', drowsy:'#ef4444', phone:'#dc2626',
  laptop:'#06b6d4', talking:'#eab308', fidgeting:'#f97316', music:'#a855f7',
  unknown:'#334155',
};

const attnColor = v => v >= 75 ? '#22c55e' : v >= 55 ? '#eab308' : v >= 35 ? '#f97316' : '#ef4444';

const S = {
  page:    { padding:'24px', color:'#f8fafc', maxWidth:'1200px', margin:'0 auto' },
  back:    { background:'none', border:'none', color:'#38bdf8', cursor:'pointer',
             fontSize:'0.85rem', padding:'0 0 16px', display:'flex', alignItems:'center', gap:'6px' },
  card:    { background:'rgba(19, 30, 54, 0.85)', border:'1px solid rgba(148, 163, 184, 0.16)', backdropFilter:'blur(20px)', borderRadius:'14px', padding:'20px', marginBottom:'16px' },
  h2:      { fontSize:'1rem', fontWeight:700, color:'#f8fafc', marginBottom:'14px' },
  badge:   { padding:'2px 8px', borderRadius:'999px', fontSize:'0.72rem', fontWeight:600 },
  grid2:   { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'16px' },
  grid3:   { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'16px' },
  stat:    { textAlign:'center' },
  statVal: { fontSize:'1.8rem', fontWeight:800, lineHeight:1.1, color:'#f8fafc' },
  statLbl: { fontSize:'0.72rem', color:'#94a3b8', marginTop:'2px' },
};

// ── Circular gauge ────────────────────────────────────────────────────────────
function Gauge({ value = 0, size = 90, label }) {
  const r    = (size - 10) / 2;
  const circ = 2 * Math.PI * r;
  const fill = circ * Math.min(value, 100) / 100;
  const col  = attnColor(value);
  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', gap:'4px' }}>
      <svg width={size} height={size} style={{ transform:'rotate(-90deg)' }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#0f172a" strokeWidth={9} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={col} strokeWidth={9}
          strokeDasharray={`${fill} ${circ}`} strokeLinecap="round"
          style={{ transition:'stroke-dasharray 0.6s ease' }} />
        <text x={size/2} y={size/2} textAnchor="middle" dominantBaseline="middle"
          fill={col} fontSize="15" fontWeight="800"
          style={{ transform:`rotate(90deg) translate(0,-${size}px)` }}>
          {value?.toFixed(0)}%
        </text>
      </svg>
      {label && <div style={{ fontSize:'0.7rem', color:'#64748b' }}>{label}</div>}
    </div>
  );
}

// ── Attendance calendar strip ─────────────────────────────────────────────────
function AttendanceStrip({ history }) {
  if (!history?.length) return <div style={{ color:'#475569', fontSize:'0.8rem' }}>No attendance data yet</div>;

  const dateSet = new Set(history.map(h => h.date));
  // Show last 30 entries ordered by date
  const dates = [...dateSet].sort().slice(-30);

  return (
    <div>
      <div style={{ display:'flex', gap:'4px', flexWrap:'wrap' }}>
        {dates.map(d => (
          <div key={d} title={d}
            style={{ width:22, height:22, borderRadius:'4px', background:'#22c55e',
                     display:'flex', alignItems:'center', justifyContent:'center',
                     fontSize:'0.55rem', color:'#fff', fontWeight:700 }}>
            {new Date(d).getDate()}
          </div>
        ))}
      </div>
      <div style={{ fontSize:'0.68rem', color:'#475569', marginTop:'6px' }}>
        Last 30 attended days shown · each box = one day present
      </div>
    </div>
  );
}

// ── Activity breakdown bar ────────────────────────────────────────────────────
function ActivityBar({ breakdown }) {
  if (!breakdown?.length) return <div style={{ color:'#475569', fontSize:'0.8rem' }}>No activity data</div>;
  const top = breakdown.slice(0, 7);
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'8px' }}>
      {top.map(({ activity, percent, seconds }) => (
        <div key={activity}>
          <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.78rem',
                        marginBottom:'3px' }}>
            <span style={{ color: ACT_COL[activity] || '#94a3b8', fontWeight:600 }}>
              {activity}
            </span>
            <span style={{ color:'#64748b' }}>
              {percent}% · {(seconds / 60).toFixed(0)} min
            </span>
          </div>
          <div style={{ height:6, borderRadius:3, background:'#0f172a', overflow:'hidden' }}>
            <div style={{ width:`${percent}%`, height:'100%',
                          background: ACT_COL[activity] || '#334155',
                          transition:'width 0.5s ease' }} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Custom tooltip for attention trend ───────────────────────────────────────
function TrendTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{ background:'#0f172a', border:'1px solid #1e293b', borderRadius:'8px',
                  padding:'10px 14px', fontSize:'0.78rem' }}>
      <div style={{ color:'#94a3b8', marginBottom:'4px' }}>{d?.session_date || label}</div>
      <div style={{ color: attnColor(d?.avg_attention), fontWeight:700 }}>
        Avg {d?.avg_attention?.toFixed(1)}%
      </div>
      {d?.peak_attention != null && (
        <div style={{ color:'#64748b' }}>Peak {d.peak_attention.toFixed(1)}%</div>
      )}
      {d?.activity && (
        <div style={{ color: ACT_COL[d.activity] || '#94a3b8', marginTop:'2px' }}>
          {d.activity}
        </div>
      )}
      <div style={{ color:'#475569', marginTop:'2px' }}>{d?.duration_min} min session</div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function StudentProfile() {
  const { id }    = useParams();
  const navigate  = useNavigate();
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  useEffect(() => {
    api.getStudentProfile(id)
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message || 'Failed to load profile'); setLoading(false); });
  }, [id]);

  if (loading) return (
    <div style={{ padding:'40px', textAlign:'center', color:'#64748b' }}>Loading profile…</div>
  );
  if (error) return (
    <div style={{ padding:'40px', textAlign:'center', color:'#ef4444' }}>{error}</div>
  );
  if (!data) return null;

  const { student, attendance, performance } = data;
  const trend  = performance.trend || [];
  const elec   = performance.electronics_totals || {};
  const noData = trend.length === 0;

  // Derive improving/declining from first vs last 3 sessions
  let trendLabel = '—';
  if (trend.length >= 4) {
    const firstAvg = trend.slice(0, 3).reduce((s, t) => s + t.avg_attention, 0) / 3;
    const lastAvg  = trend.slice(-3).reduce((s, t) => s + t.avg_attention, 0) / 3;
    const delta    = lastAvg - firstAvg;
    trendLabel = delta >= 2 ? `↑ +${delta.toFixed(1)}%` : delta <= -2 ? `↓ ${delta.toFixed(1)}%` : '→ Stable';
  }

  // Activity pie data (top 5)
  const pieData = (performance.activity_breakdown || []).slice(0, 5).map(a => ({
    name: a.activity, value: a.percent,
  }));

  return (
    <div style={S.page}>
      {/* Back button */}
      <button style={S.back} onClick={() => navigate('/students')}>
        ← Back to Students
      </button>

      {/* ── Header card ──────────────────────────────────── */}
      <div style={{ ...S.card, display:'flex', gap:'20px', alignItems:'center', flexWrap:'wrap' }}>
        {student.faceImage
          ? <img src={`data:image/jpeg;base64,${student.faceImage}`}
                 style={{ width:72, height:72, borderRadius:'50%', objectFit:'cover',
                          border:'3px solid #22c55e', flexShrink:0 }} alt="" />
          : <div style={{ width:72, height:72, borderRadius:'50%', background:'#334155',
                          display:'flex', alignItems:'center', justifyContent:'center',
                          fontSize:'1.5rem', flexShrink:0 }}>?</div>
        }
        <div style={{ flex:1, minWidth:180 }}>
          <div style={{ fontSize:'1.3rem', fontWeight:800, color:'#f1f5f9' }}>{student.name}</div>
          <div style={{ fontSize:'0.82rem', color:'#64748b', marginTop:'3px' }}>
            Roll {student.rollNo} · Section {student.section}
            {student.email && ` · ${student.email}`}
          </div>
          <div style={{ display:'flex', gap:'8px', marginTop:'8px', flexWrap:'wrap' }}>
            {student.gridCell && (
              <span style={{ ...S.badge, background:'#8b5cf622', color:'#c4b5fd',
                             border:'1px solid #8b5cf6' }}>
                📍 {student.gridCell}
              </span>
            )}
            <span style={{ ...S.badge,
                           background: student.hasFace ? '#16a34a22' : '#dc262622',
                           color:       student.hasFace ? '#4ade80'   : '#f87171',
                           border: `1px solid ${student.hasFace ? '#16a34a' : '#dc2626'}` }}>
              {student.hasFace ? '✓ Face enrolled' : '✗ No face'}
            </span>
          </div>
        </div>

        {/* Quick stats */}
        <div style={{ display:'flex', gap:'24px', flexWrap:'wrap' }}>
          <Gauge value={performance.overall_avg_attention} label="Overall avg" />
          <div style={{ display:'flex', flexDirection:'column', gap:'14px', justifyContent:'center' }}>
            <div style={S.stat}>
              <div style={{ ...S.statVal, color:'#60a5fa' }}>{attendance.total_present}</div>
              <div style={S.statLbl}>Sessions present</div>
            </div>
            <div style={S.stat}>
              <div style={{ ...S.statVal, color: attendance.rate_percent >= 75 ? '#22c55e' : '#f97316' }}>
                {attendance.rate_percent}%
              </div>
              <div style={S.statLbl}>Attendance rate</div>
            </div>
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:'14px', justifyContent:'center' }}>
            <div style={S.stat}>
              <div style={{ ...S.statVal, color:'#a78bfa' }}>{performance.sessions_tracked}</div>
              <div style={S.statLbl}>Sessions tracked</div>
            </div>
            <div style={S.stat}>
              <div style={{ ...S.statVal,
                            color: trendLabel.startsWith('↑') ? '#22c55e'
                                 : trendLabel.startsWith('↓') ? '#ef4444' : '#eab308' }}>
                {trendLabel}
              </div>
              <div style={S.statLbl}>Trend vs first 3</div>
            </div>
          </div>
        </div>
      </div>

      {noData ? (
        <div style={{ ...S.card, color:'#64748b', textAlign:'center', padding:'40px' }}>
          No session data yet. Start a session with this student to see performance history.
        </div>
      ) : (
        <>
          {/* ── Attention trend chart ─────────────────────── */}
          <div style={S.card}>
            <div style={S.h2}>Attention Trend — per session</div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trend} margin={{ top:4, right:8, bottom:0, left:-20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="session_date" tick={{ fontSize:10, fill:'#475569' }}
                  tickFormatter={d => d?.slice(5)} />
                <YAxis domain={[0, 100]} tick={{ fontSize:10, fill:'#475569' }} />
                <Tooltip content={<TrendTooltip />} />
                {/* Peak band */}
                <Line type="monotone" dataKey="peak_attention" stroke="#334155"
                  strokeWidth={1} strokeDasharray="4 2" dot={false} />
                {/* Average */}
                <Line type="monotone" dataKey="avg_attention" stroke="#3b82f6"
                  strokeWidth={2.5} dot={{ r:4, fill:'#3b82f6', strokeWidth:0 }}
                  activeDot={{ r:6 }} />
                {/* Low band */}
                <Line type="monotone" dataKey="low_attention" stroke="#1e3a5f"
                  strokeWidth={1} strokeDasharray="4 2" dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ display:'flex', gap:'16px', marginTop:'8px', fontSize:'0.7rem',
                          color:'#475569', flexWrap:'wrap' }}>
              <span>── Average</span>
              <span style={{ color:'#334155' }}>- - Peak / Low</span>
              {performance.best_session && (
                <span>Best: {performance.best_session.session_date}
                  &nbsp;({performance.best_session.avg_attention?.toFixed(0)}%)</span>
              )}
              {performance.worst_session && (
                <span>Lowest: {performance.worst_session.session_date}
                  &nbsp;({performance.worst_session.avg_attention?.toFixed(0)}%)</span>
              )}
            </div>
          </div>

          {/* ── Activity breakdown + attendance ──────────── */}
          <div style={S.grid2}>
            {/* Activity breakdown */}
            <div style={S.card}>
              <div style={S.h2}>Activity Breakdown — all sessions</div>
              <ActivityBar breakdown={performance.activity_breakdown} />
            </div>

            {/* Activity pie */}
            <div style={S.card}>
              <div style={S.h2}>Activity Split</div>
              {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name"
                      cx="50%" cy="50%" outerRadius={65} paddingAngle={2}>
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={ACT_COL[entry.name] || '#334155'} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v, n) => [`${v}%`, n]} />
                    <Legend iconType="circle" iconSize={8}
                      formatter={v => <span style={{ fontSize:'0.72rem', color:'#94a3b8' }}>{v}</span>} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color:'#475569', fontSize:'0.8rem' }}>No data</div>
              )}
            </div>
          </div>

          {/* ── Electronics + session history ────────────── */}
          <div style={S.grid2}>
            {/* Electronics usage */}
            <div style={S.card}>
              <div style={S.h2}>Electronics Usage — total across sessions</div>
              <div style={{ display:'flex', flexDirection:'column', gap:'12px' }}>
                {[
                  { key:'phone',    label:'📱 Phone',    color:'#ef4444' },
                  { key:'laptop',   label:'💻 Laptop',   color:'#06b6d4' },
                  { key:'earphone', label:'🎧 Earphone', color:'#a855f7' },
                ].map(({ key, label, color }) => {
                  const secs = elec[key] || 0;
                  const mins = (secs / 60).toFixed(1);
                  return (
                    <div key={key} style={{ display:'flex', alignItems:'center', gap:'12px' }}>
                      <div style={{ width:80, fontSize:'0.8rem', color:'#94a3b8' }}>{label}</div>
                      <div style={{ flex:1, height:8, borderRadius:4,
                                    background:'#0f172a', overflow:'hidden' }}>
                        {/* Max bar = 60 min */}
                        <div style={{ width:`${Math.min(secs / 3600 * 100, 100)}%`,
                                      height:'100%', background:color,
                                      transition:'width 0.5s ease' }} />
                      </div>
                      <div style={{ width:50, textAlign:'right', fontSize:'0.8rem',
                                    color: secs > 120 ? color : '#64748b', fontWeight:600 }}>
                        {mins} min
                      </div>
                    </div>
                  );
                })}
                {elec.phone === 0 && elec.laptop === 0 && elec.earphone === 0 && (
                  <div style={{ color:'#475569', fontSize:'0.78rem' }}>
                    No electronics usage detected
                  </div>
                )}
              </div>
            </div>

            {/* Session attention bar chart */}
            <div style={S.card}>
              <div style={S.h2}>Session Avg Attention</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={trend.slice(-10)} margin={{ top:4, right:4, bottom:0, left:-20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="session_date" tick={{ fontSize:9, fill:'#475569' }}
                    tickFormatter={d => d?.slice(5)} />
                  <YAxis domain={[0, 100]} tick={{ fontSize:9, fill:'#475569' }} />
                  <Tooltip formatter={v => [`${v?.toFixed(1)}%`, 'Avg attention']}
                    contentStyle={{ background:'#0f172a', border:'1px solid #1e293b',
                                    borderRadius:'6px', fontSize:'0.78rem' }} />
                  <Bar dataKey="avg_attention" radius={[3, 3, 0, 0]}>
                    {trend.slice(-10).map((t, i) => (
                      <Cell key={i} fill={attnColor(t.avg_attention)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize:'0.68rem', color:'#475569', marginTop:'4px' }}>
                Last {Math.min(trend.length, 10)} sessions
              </div>
            </div>
          </div>

          {/* ── Attendance strip ──────────────────────────── */}
          <div style={S.card}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center',
                          marginBottom:'12px' }}>
              <div style={S.h2}>Attendance History</div>
              <div style={{ fontSize:'0.82rem', color:'#64748b' }}>
                {attendance.total_present} of {attendance.total_sessions} sessions
                &nbsp;·&nbsp;
                <span style={{ color: attendance.rate_percent >= 75 ? '#22c55e' : '#f97316',
                               fontWeight:700 }}>
                  {attendance.rate_percent}%
                </span>
              </div>
            </div>
            <AttendanceStrip history={attendance.history} />
          </div>

          {/* ── Per-session table ─────────────────────────── */}
          <div style={S.card}>
            <div style={S.h2}>Session Log</div>
            <div style={{ overflowX:'auto' }}>
              <table style={{ width:'100%', borderCollapse:'collapse', fontSize:'0.82rem' }}>
                <thead>
                  <tr>
                    {['Date', 'Duration', 'Avg Attn', 'Peak', 'Low', 'Activity'].map(h => (
                      <th key={h} style={{ padding:'8px 12px', textAlign:'left',
                                           color:'#64748b', fontSize:'0.72rem',
                                           borderBottom:'1px solid #1e293b',
                                           textTransform:'uppercase', letterSpacing:'0.04em' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...trend].reverse().map((t, i) => (
                    <tr key={i}
                      style={{ transition:'background 0.1s' }}
                      onMouseEnter={e => e.currentTarget.style.background = '#0f172a'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                      <td style={{ padding:'9px 12px', color:'#94a3b8' }}>{t.session_date}</td>
                      <td style={{ padding:'9px 12px', color:'#64748b' }}>{t.duration_min} min</td>
                      <td style={{ padding:'9px 12px', fontWeight:700,
                                   color: attnColor(t.avg_attention) }}>
                        {t.avg_attention?.toFixed(1)}%
                      </td>
                      <td style={{ padding:'9px 12px', color:'#64748b' }}>
                        {t.peak_attention?.toFixed(1)}%
                      </td>
                      <td style={{ padding:'9px 12px', color:'#64748b' }}>
                        {t.low_attention?.toFixed(1)}%
                      </td>
                      <td style={{ padding:'9px 12px' }}>
                        <span style={{ padding:'2px 7px', borderRadius:'999px',
                                       fontSize:'0.7rem', fontWeight:600,
                                       background: (ACT_COL[t.activity] || '#334155') + '22',
                                       color: ACT_COL[t.activity] || '#94a3b8' }}>
                          {t.activity || '—'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
