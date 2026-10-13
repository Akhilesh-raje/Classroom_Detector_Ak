import React from 'react';
import { Routes, Route, Navigate, NavLink, useLocation } from 'react-router-dom';
import { Video, Bell, Expand, Brain, Users, Grid, PlayCircle, BarChart2, Layers, LayoutDashboard } from 'lucide-react';
import ClassroomApp   from './pages/ClassroomApp';
import StructureLab   from './pages/StructureLab';
import TestVideo       from './pages/TestVideo';
import Students        from './pages/Students';
import StudentProfile  from './pages/StudentProfile';
import Classroom       from './pages/Classroom';
import Attendance      from './pages/Attendance';
import Sessions        from './pages/Sessions';
import Reports         from './pages/Reports';

// ── Sidebar ───────────────────────────────────────────────────────────────────
const NAV = [
  { to: '/dashboard',     icon: <LayoutDashboard size={18} />, label: 'All-In-One Hub',     badge: 'PRO' },
  { to: '/attendance',    icon: <Users size={18} />,           label: 'Live Attendance',    badge: 'v1.0' },
  { to: '/classroom',     icon: <Grid size={18} />,            label: 'Classroom Live',     badge: 'LIVE' },
  { to: '/structure-lab', icon: <Layers size={18} />,          label: 'Structure AI Lab',   badge: 'v2.2' },
  { to: '/video-engine',  icon: <Video size={18} />,           label: 'Video Engine',       badge: 'ACTIVE' },
  { to: '/students',      icon: <Users size={18} />,           label: 'Students Roster',    badge: null },
  { to: '/sessions',      icon: <PlayCircle size={18} />,      label: 'Sessions',           badge: null },
  { to: '/reports',       icon: <BarChart2 size={18} />,       label: 'Analytics Reports',  badge: null },
];

const Sidebar = () => (
  <aside className="sidebar">
    <div className="sidebar-header">
      <div className="logo">
        <div className="logo-icon"><Brain size={20} color="#04101e" /></div>
        <div className="logo-text">
          <h2>SmartClass</h2>
          <span>AI Engine v2.2</span>
        </div>
      </div>
    </div>

    <div className="sidebar-nav">
      <div className="nav-section">
        <span className="nav-label">Command Center</span>
        <ul className="nav-links">
          {NAV.map(({ to, icon, label, badge }) => (
            <li key={to}>
              <NavLink to={to} className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
                {icon}
                <span>{label}</span>
                {badge && (
                  <span className={badge === 'PRO' ? 'nav-badge-live' : badge === 'LIVE' ? 'nav-badge-live' : 'nav-badge-active'}>
                    {badge}
                  </span>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </div>

    <div className="sidebar-footer">
      <div>SmartClass AI · All-In-One Classroom Command Center</div>
    </div>
  </aside>
);

// ── Topbar ────────────────────────────────────────────────────────────────────
const PAGE_TITLES = {
  '/dashboard':    'All-In-One Classroom Command Center',
  '/attendance':   'Live Attendance Command Center',
  '/structure-lab': 'Structure AI Laboratory',
  '/classroom':    'Classroom Seating Grid',
  '/students':     'Student Management & Roster',
  '/sessions':     'Session Monitoring & History',
  '/reports':      'Reports & Longitudinal Analytics',
  '/video-engine': 'Video Analysis & Expression Engine',
};

const Topbar = () => {
  const [time, setTime] = React.useState(new Date());
  const location = useLocation();
  const pageTitle = PAGE_TITLES[location.pathname] || 'SmartClass AI';

  React.useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header className="topbar">
      <div className="flex items-center gap-4">
        <span className="topbar-page">{pageTitle}</span>
      </div>
      <div className="topbar-right">
        <div className="topbar-clock">
          <span className="clock-time mono">
            {time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
          <span className="clock-date">
            {time.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}
          </span>
        </div>
        <button className="topbar-btn" title="System Notifications"><Bell size={16} /></button>
        <button 
          className="topbar-btn" 
          title="Toggle Fullscreen"
          onClick={() => document.documentElement.requestFullscreen().catch(() => {})}
        >
          <Expand size={16} />
        </button>
      </div>
    </header>
  );
};

// ── App Shell ─────────────────────────────────────────────────────────────────
export default function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-wrapper">
        <Topbar />
        <main className="page-container">
          <Routes>
            <Route path="/"              element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard"     element={<ClassroomApp />} />
            <Route path="/attendance"    element={<Attendance />} />
            <Route path="/structure-lab" element={<StructureLab />} />
            <Route path="/classroom"     element={<Classroom />} />
            <Route path="/students"      element={<Students />} />
            <Route path="/students/:id"  element={<StudentProfile />} />
            <Route path="/sessions"      element={<Sessions />} />
            <Route path="/reports"       element={<Reports />} />
            <Route path="/video-engine"  element={<TestVideo />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
