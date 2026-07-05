import { Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import Compare from './pages/Compare';
import Metrics from './pages/Metrics';
import History from './pages/History';

const NAV = [
  { to: '/', icon: 'TQ', label: 'Tổng quan' },
  { to: '/analyze', icon: 'PT', label: 'Phân tích' },
  { to: '/compare', icon: 'SS', label: 'So sánh' },
  { to: '/metrics', icon: 'CS', label: 'Chỉ số' },
  { to: '/history', icon: 'LS', label: 'Lịch sử' },
];

export default function App() {
  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span>MS</span> MedSeg
        </div>
        {NAV.map(n => (
          <NavLink key={n.to} to={n.to} end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <span className="nav-icon">{n.icon}</span>
            <span>{n.label}</span>
          </NavLink>
        ))}
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
