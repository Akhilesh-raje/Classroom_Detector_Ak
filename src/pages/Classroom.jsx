import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api';

const S = {
  page:    { padding: '24px', color: '#e2e8f0' },
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
  badge:   { padding:'2px 8px', borderRadius:'999px', fontSize:'0.72rem', fontWeight:600 },
};

const ACT_COL = {
  studying:'#22c55e', attentive:'#22c55e', neutral:'#eab308',
  distracted:'#f97316', drowsy:'#ef4444', phone:'#ef4444',
  laptop:'#06b6d4', talking:'#eab308', fidgeting:'#f97316', music:'#a855f7',
};

function GridCell({ cell, student, liveData, seatLabel }) {
  const attn    = liveData?.attentionScore;
  const act     = liveData?.activity;
  const present = liveData != null;
  const col     = attn != null ? (attn >= 75 ? '#22c55e' : attn >= 55 ? '#eab308' : '#ef4444') : '#334155';
  const actCol  = ACT_COL[act] || '#64748b';

  return (
    <div style={{
      background: '#0f172a',
      border: `2px solid ${present ? col : '#1e293b'}`,
      borderRadius: '10px',
      padding: '14px',
      minHeight: '110px',
      display: 'flex',
      flexDirection: 'column',
      gap: '6px',
      transition: 'border-color 0.3s',
      position: 'relative',
    }}>
      {/* Cell label */}
      <div style={{ fontSize:'0.7rem', color:'#475569', fontWeight:600, letterSpacing:'0.05em' }}>
        {cell.label}
      </div>

      {/* Student name / seat label */}
      <div style={{ fontWeight:700, color:'#f1f5f9', fontSize:'0.9rem', lineHeight:1.2 }}>
        {seatLabel || student?.name || <span style={{color:'#334155'}}>Empty</span>}
      </div>

      {student?.rollNo && (
        <div style={{ fontSize:'0.72rem', color:'#64748b' }}>{student.rollNo}</div>
      )}

      {/* Live data */}
      {present ? (
        <>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginTop:'auto' }}>
            <span style={{ ...S.badge, background: actCol + '22', color: actCol, border:`1px solid ${actCol}` }}>
              {act?.toUpperCase() || '—'}
            </span>
            <span style={{ fontSize:'0.85rem', fontWeight:700, color: col }}>
              {attn?.toFixed(0)}%
            </span>
          </div>
          {/* Attention bar */}
          <div style={{ height:'4px', borderRadius:'2px', background:'#1e293b', overflow:'hidden' }}>
            <div style={{ width:`${Math.min(attn||0,100)}%`, height:'100%', background:col, transition:'width 0.5s' }} />
          </div>
        </>
      ) : (
        <div style={{ marginTop:'auto', fontSize:'0.75rem', color:'#334155' }}>
          {student ? '● Registered' : '○ Unassigned'}
        </div>
      )}
    </div>
  );
}

export default function Classroom() {
  const [students, setStudents]     = useState([]);
  const [liveMap, setLiveMap]       = useState({});   // gridCell → face data
  const [seatLabels, setSeatLabels] = useState({});   // gridCell → display name from Ollama
  const [ollamaStatus, setOStatus]  = useState(null);
  const [analysis, setAnalysis]     = useState('');
  const [analyzing, setAnalyzing]   = useState(false);
  const [labeling, setLabeling]     = useState(false);
  const [msg, setMsg]               = useState('');
  const pollRef = useRef(null);

  // Build grid from 2×3 default or detected layout
  const [gridCells, setGridCells] = useState(() => {
    const cells = [];
    for (let r = 1; r <= 2; r++)
      for (let c = 1; c <= 3; c++)
        cells.push({ label: `R${r}C${c}`, row: r, col: c });
    return cells;
  });

  const flash = (m) => { setMsg(m); setTimeout(() => setMsg(''), 4000); };

  useEffect(() => {
    api.getStudents().then(d => setStudents(Array.isArray(d) ? d : []));
    api.ollamaStatus().then(d => setOStatus(d)).catch(() => setOStatus({ available: false }));

    // Poll live detections every 2s
    pollRef.current = setInterval(() => {
      api.getDetections().then(d => {
        const map = {};
        (d.faces || []).forEach(f => {
          const gc = f.gridCell || f.grid_cell;
          if (gc) map[gc] = f;
        });
        setLiveMap(map);
      }).catch(() => {});
    }, 2000);

    return () => clearInterval(pollRef.current);
  }, []);

  const handleAutoLabel = async () => {
    setLabeling(true);
    try {
      const r = await api.ollamaLabelSeats();
      if (r.mapping) {
        setSeatLabels(r.mapping);
        flash('Seats labeled by AI!');
        // Expand grid if needed
        if (r.grid_cells?.length > gridCells.length) {
          const cells = r.grid_cells.map(lbl => {
            const m = lbl.match(/R(\d+)C(\d+)/);
            return m ? { label: lbl, row: parseInt(m[1]), col: parseInt(m[2]) } : { label: lbl, row:1, col:1 };
          });
          setGridCells(cells);
        }
      }
    } catch { flash('Labeling failed'); }
    setLabeling(false);
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setAnalysis('');
    try {
      const r = await api.ollamaAnalyze();
      setAnalysis(r.summary || 'No analysis available.');
    } catch { setAnalysis('Analysis failed. Make sure Ollama is running.'); }
    setAnalyzing(false);
  };

  // Build student map by grid_cell
  const studentByCell = {};
  students.forEach(s => { if (s.gridCell) studentByCell[s.gridCell] = s; });

  // Group cells by row
  const rows = {};
  gridCells.forEach(c => {
    if (!rows[c.row]) rows[c.row] = [];
    rows[c.row].push(c);
  });

  const liveCount  = Object.keys(liveMap).length;
  const studyCount = Object.values(liveMap).filter(f => ['studying','attentive'].includes(f.activity)).length;

  return (
    <div style={S.page}>
      {msg && (
        <div style={{ background:'#22c55e22', border:'1px solid #22c55e', borderRadius:'8px',
          padding:'10px 16px', marginBottom:'16px', color:'#22c55e' }}>{msg}</div>
      )}

      <div style={S.header}>
        <div>
          <h1 style={S.title}>Classroom Grid</h1>
          <div style={{ fontSize:'0.8rem', color:'#64748b', marginTop:'2px' }}>
            {liveCount > 0
              ? `${liveCount} students live · ${studyCount} studying`
              : 'No live session — start webcam to see live data'}
          </div>
        </div>
        <div style={S.row}>
          <button style={S.btnGreen} onClick={handleAutoLabel} disabled={labeling}>
            {labeling ? 'Labeling...' : '🤖 AI Label Seats'}
          </button>
          <button style={S.btn} onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? 'Analyzing...' : '📊 AI Analyze'}
          </button>
          <div style={{ fontSize:'0.75rem', color: ollamaStatus?.available ? '#22c55e' : '#64748b',
            padding:'6px 10px', background:'#0f172a', borderRadius:'6px' }}>
            Ollama: {ollamaStatus?.available ? `✓ ${ollamaStatus.model || 'ready'}` : '✗ offline'}
          </div>
        </div>
      </div>

      {/* AI Analysis box */}
      {analysis && (
        <div style={{ ...S.card, background:'#0f172a', border:'1px solid #1d4ed8', marginBottom:'20px' }}>
          <div style={{ fontSize:'0.75rem', color:'#60a5fa', marginBottom:'8px', fontWeight:600 }}>
            🤖 AI Classroom Analysis
          </div>
          <div style={{ fontSize:'0.88rem', color:'#cbd5e1', lineHeight:1.6 }}>{analysis}</div>
        </div>
      )}

      {/* Grid */}
      {Object.entries(rows).sort((a,b) => a[0]-b[0]).map(([rowNum, cells]) => (
        <div key={rowNum} style={{ marginBottom:'16px' }}>
          <div style={{ fontSize:'0.72rem', color:'#475569', marginBottom:'8px',
            textTransform:'uppercase', letterSpacing:'0.05em' }}>Row {rowNum}</div>
          <div style={{ display:'grid', gridTemplateColumns:`repeat(${cells.length}, 1fr)`, gap:'12px' }}>
            {cells.sort((a,b) => a.col-b.col).map(cell => (
              <GridCell
                key={cell.label}
                cell={cell}
                student={studentByCell[cell.label]}
                liveData={liveMap[cell.label]}
                seatLabel={seatLabels[cell.label]}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Legend */}
      <div style={{ ...S.card, display:'flex', gap:'16px', flexWrap:'wrap', marginTop:'8px' }}>
        <div style={{ fontSize:'0.75rem', color:'#64748b', fontWeight:600 }}>Legend:</div>
        {[['#22c55e','High Attention (≥75%)'],['#eab308','Medium (55-74%)'],['#ef4444','Low (<55%)'],
          ['#334155','No live data']].map(([col,lbl]) => (
          <div key={lbl} style={{ display:'flex', alignItems:'center', gap:'6px', fontSize:'0.75rem', color:'#94a3b8' }}>
            <div style={{ width:10, height:10, borderRadius:'50%', background:col }} />
            {lbl}
          </div>
        ))}
      </div>
    </div>
  );
}
