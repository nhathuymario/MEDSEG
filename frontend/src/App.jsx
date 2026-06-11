import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import Compare from './pages/Compare';
import History from './pages/History';

const NAV = [
  { to: '/', icon: '📊', label: 'Dashboard' },
  { to: '/analyze', icon: '🔬', label: 'Analyze' },
  { to: '/compare', icon: '⚖️', label: 'Compare' },
  { to: '/history', icon: '📋', label: 'History' },
];

export default function App() {
  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span>🏥</span> MedSeg
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
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
