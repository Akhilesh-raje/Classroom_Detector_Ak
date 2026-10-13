import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { 
  Users, 
  UserPlus, 
  Search, 
  Trash2, 
  Camera, 
  ExternalLink, 
  Grid, 
  CheckCircle2, 
  X, 
  Sparkles,
  Upload,
  RefreshCw,
  SlidersHorizontal
} from 'lucide-react';

export default function Students() {
  const navigate = useNavigate();
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [enrollStudent, setEnrollStudent] = useState(null); // student being multi-shot enrolled

  // New Student Form State
  const [name, setName] = useState('');
  const [rollNo, setRollNo] = useState('');
  const [section, setSection] = useState('A');
  const [gridCell, setGridCell] = useState('R1C1');
  const [faceFile, setFaceFile] = useState(null);

  const loadStudents = () => {
    setLoading(true);
    api.getStudents()
      .then(d => setStudents(Array.isArray(d) ? d : []))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadStudents();
  }, []);

  const handleCreateStudent = async (e) => {
    e.preventDefault();
    if (!name || !rollNo) return;

    const fd = new FormData();
    fd.append('name', name);
    fd.append('roll_no', rollNo);
    fd.append('section', section);
    fd.append('grid_cell', gridCell);
    if (faceFile) fd.append('face_image', faceFile);

    try {
      await api.createStudent(fd);
      setShowModal(false);
      setName('');
      setRollNo('');
      setFaceFile(null);
      loadStudents();
    } catch (err) {
      alert('Failed to register student: ' + err.message);
    }
  };

  const handleDelete = async (id, sName) => {
    if (!window.confirm(`Are you sure you want to remove ${sName}?`)) return;
    try {
      await api.deleteStudent(id);
      loadStudents();
    } catch (err) {
      alert('Failed to delete student: ' + err.message);
    }
  };

  const filtered = students.filter(s => 
    s.name.toLowerCase().includes(search.toLowerCase()) ||
    (s.rollNo && s.rollNo.toLowerCase().includes(search.toLowerCase())) ||
    (s.gridCell && s.gridCell.toLowerCase().includes(search.toLowerCase()))
  );

  const SEATS = [
    'R1C1', 'R1C2', 'R1C3', 'R1C4', 'R1C5', 'R1C6',
    'R2C1', 'R2C2', 'R2C3', 'R2C4', 'R2C5', 'R2C6'
  ];

  return (
    <div className="apple-command-root" style={{ gap: '1.75rem' }}>
      {/* Top Banner Header */}
      <div className="apple-header-card">
        <div className="apple-header-left">
          <div className="header-title-block">
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span className="shadcn-badge shadcn-badge-cyan">
                <Users size={12} />
                Student Enrollment & Roster
              </span>
              <span className="shadcn-badge shadcn-badge-emerald">
                <CheckCircle2 size={12} />
                {students.length} Registered Students
              </span>
            </div>
            <h1>Classroom Student Directory</h1>
            <p>Permanent seat mapping, multi-shot biometric face enrollment, and cognitive metrics history.</p>
          </div>
        </div>

        <div className="apple-header-actions">
          <div className="cmd-search-input-wrap">
            <Search size={14} color="var(--muted-foreground)" />
            <input 
              type="text" 
              placeholder="Filter by name, roll no, seat..." 
              value={search} 
              onChange={e => setSearch(e.target.value)} 
            />
          </div>

          <button 
            className="shadcn-btn shadcn-btn-cyan"
            onClick={() => setShowModal(true)}
          >
            <UserPlus size={14} />
            Register Student
          </button>
        </div>
      </div>

      {/* Students Data Table */}
      <div className="shadcn-table-wrap">
        <table className="shadcn-table">
          <thead>
            <tr>
              <th>Student Details</th>
              <th>Roll Number</th>
              <th>Section</th>
              <th>Assigned Seat</th>
              <th>Biometric Face</th>
              <th>Attentiveness</th>
              <th>Attendance</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="8" style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted-foreground)' }}>
                  Loading students roster...
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan="8" style={{ textAlign: 'center', padding: '2rem', color: 'var(--muted-foreground)' }}>
                  No students found matching "{search}"
                </td>
              </tr>
            ) : (
              filtered.map(student => {
                const isFront = (student.gridCell || '').startsWith('R1');
                const attn = student.avgAttentiveness ?? 85;
                const col = attn >= 75 ? '#10b981' : attn >= 55 ? '#f59e0b' : '#ef4444';

                return (
                  <tr key={student.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {student.faceImage ? (
                          <img 
                            src={`http://localhost:8000/faces/${student.faceImage}`} 
                            alt={student.name}
                            style={{ width: 34, height: 34, borderRadius: 'var(--radius-sm)', objectFit: 'cover', border: '1px solid var(--border)' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        ) : (
                          <div style={{ 
                            width: 34, height: 34, borderRadius: 'var(--radius-sm)', 
                            background: 'rgba(30, 41, 59, 0.6)', 
                            border: '1px solid var(--border)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '0.8rem', fontWeight: 700, color: 'var(--muted-foreground)'
                          }}>
                            {student.name.charAt(0)}
                          </div>
                        )}
                        <div>
                          <strong style={{ color: '#f8fafc', fontSize: '0.88rem' }}>{student.name}</strong>
                          <div style={{ fontSize: '0.72rem', color: 'var(--muted-foreground)' }}>{student.email || 'No email registered'}</div>
                        </div>
                      </div>
                    </td>
                    <td className="mono" style={{ color: 'var(--muted-foreground)' }}>{student.rollNo}</td>
                    <td><span className="shadcn-badge shadcn-badge-outline">{student.section || 'A'}</span></td>
                    <td>
                      <span 
                        className="mono" 
                        style={{ 
                          fontSize: '0.75rem', 
                          fontWeight: 700, 
                          color: isFront ? 'var(--accent-cyan)' : 'var(--accent-violet)',
                          background: isFront ? 'rgba(0, 240, 255, 0.1)' : 'rgba(168, 85, 247, 0.1)',
                          padding: '2px 8px',
                          borderRadius: '4px'
                        }}
                      >
                        {student.gridCell || 'Unassigned'}
                      </span>
                    </td>
                    <td>
                      {student.hasFace ? (
                        <span className="shadcn-badge shadcn-badge-emerald">
                          <CheckCircle2 size={11} />
                          Enrolled
                        </span>
                      ) : (
                        <span className="shadcn-badge shadcn-badge-outline" style={{ color: 'var(--muted-foreground)' }}>
                          No Face Enc
                        </span>
                      )}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{ width: 60, height: 4, background: 'rgba(255,255,255,0.08)', borderRadius: 2, overflow: 'hidden' }}>
                          <div style={{ width: `${attn}%`, height: '100%', background: col }} />
                        </div>
                        <span className="mono" style={{ fontSize: '0.78rem', fontWeight: 700, color: col }}>
                          {attn.toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="mono" style={{ color: '#f8fafc' }}>
                      {student.totalAttendance ?? 1} sessions
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '6px' }}>
                        <button 
                          className="shadcn-btn shadcn-btn-secondary" 
                          style={{ padding: '0.3rem 0.6rem', fontSize: '0.72rem' }}
                          onClick={() => navigate(`/students/${student.id}`)}
                          title="Open Student Profile"
                        >
                          <ExternalLink size={12} />
                          Profile
                        </button>
                        <button 
                          className="shadcn-btn shadcn-btn-destructive" 
                          style={{ padding: '0.3rem 0.6rem', fontSize: '0.72rem' }}
                          onClick={() => handleDelete(student.id, student.name)}
                          title="Delete Student"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Register Student Modal */}
      {showModal && (
        <div className="apple-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="apple-dossier-card" style={{ maxWidth: '500px' }} onClick={e => e.stopPropagation()}>
            <div className="dossier-header-row">
              <div className="dossier-user-block">
                <div style={{ width: 44, height: 44, borderRadius: 'var(--radius-sm)', background: 'rgba(14, 165, 233, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <UserPlus size={22} color="#00f0ff" />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#f8fafc' }}>Register New Student</h3>
                  <p style={{ fontSize: '0.75rem', color: 'var(--muted-foreground)' }}>Add student record and link permanent physical desk</p>
                </div>
              </div>
              <button className="topbar-btn" onClick={() => setShowModal(false)}><X size={15} /></button>
            </div>

            <form onSubmit={handleCreateStudent} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)', display: 'block', marginBottom: '4px' }}>Student Full Name</label>
                <input 
                  type="text" 
                  className="shadcn-input" 
                  placeholder="e.g. Alex Mercer"
                  value={name} 
                  onChange={e => setName(e.target.value)} 
                  required 
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)', display: 'block', marginBottom: '4px' }}>Roll / Registration No</label>
                  <input 
                    type="text" 
                    className="shadcn-input mono" 
                    placeholder="e.g. CS-2026-01"
                    value={rollNo} 
                    onChange={e => setRollNo(e.target.value)} 
                    required 
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)', display: 'block', marginBottom: '4px' }}>Section</label>
                  <input 
                    type="text" 
                    className="shadcn-input" 
                    placeholder="e.g. A"
                    value={section} 
                    onChange={e => setSection(e.target.value)} 
                  />
                </div>
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)', display: 'block', marginBottom: '4px' }}>Calibrated Desk Assignment</label>
                <select 
                  className="shadcn-input mono" 
                  value={gridCell} 
                  onChange={e => setGridCell(e.target.value)}
                >
                  {SEATS.map(s => (
                    <option key={s} value={s}>{s} {s.startsWith('R1') ? '(Row 1 · Front)' : '(Row 2 · Back)'}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '0.78rem', color: 'var(--muted-foreground)', display: 'block', marginBottom: '4px' }}>Enroll Reference Photo (Optional)</label>
                <input 
                  type="file" 
                  accept="image/*" 
                  className="shadcn-input" 
                  onChange={e => setFaceFile(e.target.files[0])} 
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '1rem' }}>
                <button type="button" className="shadcn-btn shadcn-btn-secondary" onClick={() => setShowModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="shadcn-btn shadcn-btn-cyan">
                  Save Student
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
