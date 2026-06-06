import React from 'react';
import { Routes, Route, Navigate, NavLink } from 'react-router-dom';
import { Video, Bell, Expand, Brain, Users, Grid, PlayCircle, BarChart2 } from 'lucide-react';
import TestVideo  from './pages/TestVideo';
import Students   from './pages/Students';
import Classroom  from './pages/Classroom';
import Sessions   from './pages/Sessions';
import Reports    from './pages/Reports';

// ── Sidebar ───────────────────────────────────────────────────────────────────
const NAV = [
  { to:'/classroom',    icon:<Grid size={17}/>,       label:'Classroom',      badge:null },
  { to:'/students',     icon:<Users size={17}/>,      label:'Students',       badge:null },
  { to:'/sessions',     icon:<PlayCircle size={17}/>, label:'Sessions',       badge:'LIVE' },
  { to:'/reports',      icon:<BarChart2 size={17}/>,  label:'Reports',        badge:null },
  { to:'/video-engine', icon:<Video size={17}/>,      label:'Video Analysis', badge:'ACTIVE' },
];

const Sidebar = () => (
  <aside className="sidebar">
    <div className="sidebar-header">
      <div className="logo">
        <div className="logo-icon"><Brain size={18} color="white" /></div>
        <div className="logo-text">
          <h2>SmartClass</h2>
          <span>AI Engine v2</span>
        </div>
      </div>
    </div>

    <div className="sidebar-nav">
      <div className="nav-section">
        <span className="nav-label">Dashboard</span>
        <ul className="nav-links">
          {NAV.map(({ to, icon, label, badge }) => (
            <li key={to}>
              <NavLink to={to} className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                {icon}
                <span>{label}</span>
                {badge && <span className="nav-badge-live">{badge}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </div>

    <div className="sidebar-footer">
      <div style={{ fontSize:'0.7rem', color:'#475569', textAlign:'center' }}>
        SmartClass AI v2.0 · All local
      </div>
    </div>
  </aside>
);

// ── Topbar ────────────────────────────────────────────────────────────────────
const PAGE_TITLES = {
  '/classroom':    'Classroom Grid',
  '/students':     'Student Management',
  '/sessions':     'Session Management',
  '/reports':      'Reports & Analytics',
  '/video-engine': 'Video Analysis Engine',
};

const Topbar = () => {
  const [time, setTime] = React.useState(new Date());
  const path = window.location.pathname;
  const pageTitle = PAGE_TITLES[path] || 'SmartClass AI';

  React.useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header className="topbar">
      <div className="flex items-center gap-4">
        <div className="topbar-brand">
          <span className="topbar-page">{pageTitle}</span>
        </div>
      </div>
      <div className="topbar-right">
        <div className="topbar-clock">
          <span className="clock-time mono">
            {time.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' })}
          </span>
          <span className="clock-date">
            {time.toLocaleDateString([], { weekday:'short', month:'short', day:'numeric' })}
          </span>
        </div>
        <button className="topbar-btn" title="Notifications"><Bell size={16} /></button>
        <button className="topbar-btn" title="Fullscreen"
          onClick={() => document.documentElement.requestFullscreen().catch(()=>{})}>
          <Expand size={16} />
        </button>
      </div>
    </header>
  );
};

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-wrapper">
        <Topbar />
        <main className="page-container">
          <Routes>
            <Route path="/"             element={<Navigate to="/classroom" replace />} />
            <Route path="/classroom"    element={<Classroom />} />
            <Route path="/students"     element={<Students />} />
            <Route path="/sessions"     element={<Sessions />} />
            <Route path="/reports"      element={<Reports />} />
            <Route path="/video-engine" element={<TestVideo />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
