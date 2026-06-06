import React, { useState, useEffect, useRef } from 'react';
import { api } from '../api';

const GRID_CELLS = ['R1C1','R1C2','R1C3','R1C4','R2C1','R2C2','R2C3','R2C4',
                    'R3C1','R3C2','R3C3','R3C4','R4C1','R4C2','R4C3','R4C4'];

const S = {
  page:    { padding: '24px', color: '#e2e8f0' },
  header:  { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'24px' },
  title:   { fontSize:'1.4rem', fontWeight:700, color:'#f1f5f9' },
  btn:     { padding:'8px 16px', borderRadius:'8px', border:'none', cursor:'pointer',
             background:'#3b82f6', color:'#fff', fontWeight:600, fontSize:'0.85rem' },
  btnRed:  { padding:'6px 12px', borderRadius:'6px', border:'none', cursor:'pointer',
             background:'#ef4444', color:'#fff', fontSize:'0.78rem' },
  btnGray: { padding:'6px 12px', borderRadius:'6px', border:'none', cursor:'pointer',
             background:'#334155', color:'#94a3b8', fontSize:'0.78rem' },
  table:   { width:'100%', borderCollapse:'collapse' },
  th:      { padding:'10px 14px', textAlign:'left', fontSize:'0.75rem', color:'#64748b',
             borderBottom:'1px solid #1e293b', textTransform:'uppercase', letterSpacing:'0.05em' },
  td:      { padding:'12px 14px', fontSize:'0.85rem', borderBottom:'1px solid #1e293b' },
  card:    { background:'#1e293b', borderRadius:'12px', overflow:'hidden' },
  badge:   { padding:'2px 8px', borderRadius:'999px', fontSize:'0.72rem', fontWeight:600 },
  overlay: { position:'fixed', inset:0, background:'rgba(0,0,0,0.7)', display:'flex',
             alignItems:'center', justifyContent:'center', zIndex:1000 },
  modal:   { background:'#1e293b', borderRadius:'16px', padding:'28px', width:'420px',
             maxWidth:'90vw', color:'#e2e8f0' },
  input:   { width:'100%', padding:'8px 12px', borderRadius:'8px', border:'1px solid #334155',
             background:'#0f172a', color:'#e2e8f0', fontSize:'0.85rem', marginTop:'6px',
             boxSizing:'border-box' },
  label:   { fontSize:'0.8rem', color:'#94a3b8', display:'block', marginTop:'12px' },
  select:  { width:'100%', padding:'8px 12px', borderRadius:'8px', border:'1px solid #334155',
             background:'#0f172a', color:'#e2e8f0', fontSize:'0.85rem', marginTop:'6px' },
};

export default function Students() {
  const [students, setStudents]   = useState([]);
  const [loading, setLoading]     = useState(true);
  const [showAdd, setShowAdd]     = useState(false);
  const [showSeat, setShowSeat]   = useState(null);  // student object
  const [showFace, setShowFace]   = useState(null);  // student object
  const [form, setForm]           = useState({ name:'', rollNo:'', section:'', email:'' });
  const [selectedCell, setCell]   = useState('R1C1');
  const [msg, setMsg]             = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const load = () => {
    setLoading(true);
    api.getStudents().then(d => { setStudents(d); setLoading(false); }).catch(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const flash = (m) => { setMsg(m); setTimeout(() => setMsg(''), 3000); };

  const handleAdd = async () => {
    const fd = new FormData();
    Object.entries(form).forEach(([k,v]) => fd.append(k === 'rollNo' ? 'rollNo' : k, v));
    const r = await api.createStudent(fd);
    if (r.id) { flash('Student registered!'); setShowAdd(false); setForm({name:'',rollNo:'',section:'',email:''}); load(); }
    else flash(r.detail || 'Error');
  };

  const handleDelete = async (id) => {
    if (!confirm('Delete this student?')) return;
    await api.deleteStudent(id);
    load();
  };

  const handleAssignSeat = async () => {
    const r = await api.assignSeat(showSeat.id, selectedCell);
    flash(r.message || 'Seat assigned');
    setShowSeat(null);
    load();
  };

  const startFaceCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    } catch { flash('Camera access denied'); }
  };

  const captureFace = async () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      const r = await api.registerFace(showFace.id, blob);
      if (r.thumbnail) { flash('Face registered!'); setShowFace(null); load(); }
      else flash(r.detail || 'No face detected');
      video.srcObject?.getTracks().forEach(t => t.stop());
    }, 'image/jpeg');
  };

  return (
    <div style={S.page}>
      {msg && <div style={{background:'#22c55e22',border:'1px solid #22c55e',borderRadius:'8px',padding:'10px 16px',marginBottom:'16px',color:'#22c55e'}}>{msg}</div>}

      <div style={S.header}>
        <h1 style={S.title}>Students</h1>
        <button style={S.btn} onClick={() => setShowAdd(true)}>+ Register Student</button>
      </div>

      <div style={S.card}>
        {loading ? (
          <div style={{padding:'40px',textAlign:'center',color:'#64748b'}}>Loading...</div>
        ) : (
          <table style={S.table}>
            <thead>
              <tr>
                {['Name','Roll No','Section','Grid Cell','Face','Actions'].map(h => (
                  <th key={h} style={S.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {students.length === 0 && (
                <tr><td colSpan={6} style={{...S.td, textAlign:'center', color:'#64748b', padding:'40px'}}>No students registered yet</td></tr>
              )}
              {students.map(s => (
                <tr key={s.id} style={{transition:'background 0.15s'}}
                    onMouseEnter={e => e.currentTarget.style.background='#0f172a'}
                    onMouseLeave={e => e.currentTarget.style.background='transparent'}>
                  <td style={S.td}>
                    <div style={{display:'flex',alignItems:'center',gap:'10px'}}>
                      {s.faceImage
                        ? <img src={`data:image/jpeg;base64,${s.faceImage}`} style={{width:32,height:32,borderRadius:'50%',objectFit:'cover'}} alt="" />
                        : <div style={{width:32,height:32,borderRadius:'50%',background:'#334155',display:'flex',alignItems:'center',justifyContent:'center',fontSize:'0.75rem',color:'#64748b'}}>?</div>
                      }
                      <span style={{fontWeight:600,color:'#f1f5f9'}}>{s.name}</span>
                    </div>
                  </td>
                  <td style={{...S.td,color:'#94a3b8'}}>{s.rollNo}</td>
                  <td style={{...S.td,color:'#94a3b8'}}>{s.section}</td>
                  <td style={S.td}>
                    {s.gridCell
                      ? <span style={{...S.badge,background:'#1d4ed822',color:'#60a5fa',border:'1px solid #1d4ed8'}}>{s.gridCell}</span>
                      : <span style={{...S.badge,background:'#33415522',color:'#64748b'}}>Unassigned</span>
                    }
                  </td>
                  <td style={S.td}>
                    {s.hasFace
                      ? <span style={{...S.badge,background:'#16a34a22',color:'#4ade80',border:'1px solid #16a34a'}}>✓ Registered</span>
                      : <span style={{...S.badge,background:'#dc262622',color:'#f87171'}}>No face</span>
                    }
                  </td>
                  <td style={{...S.td,display:'flex',gap:'6px',flexWrap:'wrap'}}>
                    <button style={S.btnGray} onClick={() => { setShowSeat(s); setCell(s.gridCell || 'R1C1'); }}>Assign Seat</button>
                    <button style={S.btnGray} onClick={() => { setShowFace(s); setTimeout(startFaceCapture, 100); }}>Register Face</button>
                    <button style={S.btnRed}  onClick={() => handleDelete(s.id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Add Student Modal */}
      {showAdd && (
        <div style={S.overlay} onClick={() => setShowAdd(false)}>
          <div style={S.modal} onClick={e => e.stopPropagation()}>
            <h2 style={{marginBottom:'16px',fontSize:'1.1rem',fontWeight:700}}>Register Student</h2>
            {[['name','Full Name'],['rollNo','Roll Number'],['section','Section'],['email','Email (optional)']].map(([k,l]) => (
              <div key={k}>
                <label style={S.label}>{l}</label>
                <input style={S.input} value={form[k]} onChange={e => setForm({...form,[k]:e.target.value})} placeholder={l} />
              </div>
            ))}
            <div style={{display:'flex',gap:'10px',marginTop:'20px'}}>
              <button style={S.btn} onClick={handleAdd}>Register</button>
              <button style={S.btnGray} onClick={() => setShowAdd(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* Assign Seat Modal */}
      {showSeat && (
        <div style={S.overlay} onClick={() => setShowSeat(null)}>
          <div style={S.modal} onClick={e => e.stopPropagation()}>
            <h2 style={{marginBottom:'16px',fontSize:'1.1rem',fontWeight:700}}>Assign Seat — {showSeat.name}</h2>
            <label style={S.label}>Grid Cell</label>
            <select style={S.select} value={selectedCell} onChange={e => setCell(e.target.value)}>
              {GRID_CELLS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <div style={{display:'flex',gap:'10px',marginTop:'20px'}}>
              <button style={S.btn} onClick={handleAssignSeat}>Assign</button>
              <button style={S.btnGray} onClick={() => setShowSeat(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* Face Registration Modal */}
      {showFace && (
        <div style={S.overlay} onClick={() => { showFace && videoRef.current?.srcObject?.getTracks().forEach(t=>t.stop()); setShowFace(null); }}>
          <div style={S.modal} onClick={e => e.stopPropagation()}>
            <h2 style={{marginBottom:'16px',fontSize:'1.1rem',fontWeight:700}}>Register Face — {showFace.name}</h2>
            <video ref={videoRef} style={{width:'100%',borderRadius:'8px',background:'#0f172a'}} muted />
            <canvas ref={canvasRef} style={{display:'none'}} />
            <div style={{display:'flex',gap:'10px',marginTop:'16px'}}>
              <button style={S.btn} onClick={captureFace}>Capture & Register</button>
              <button style={S.btnGray} onClick={() => { videoRef.current?.srcObject?.getTracks().forEach(t=>t.stop()); setShowFace(null); }}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
