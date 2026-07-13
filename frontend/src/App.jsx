import { Navigate, NavLink, Route, Routes } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Results from './pages/Results';
import Compare from './pages/Compare';
import Metrics from './pages/Metrics';
import History from './pages/History';

const ANALYSIS_NAV = [
  { to: '/analyze/detect', icon: 'DA', label: 'Phát hiện tổn thương da' },
  { to: '/analyze/segment', icon: 'XR', label: 'Phân đoạn X-quang' },
  { to: '/analyze/pipeline', icon: 'PL', label: 'Full pipeline da' },
];

const REPORT_NAV = [
  { to: '/compare', icon: 'SS', label: 'So sánh' },
  { to: '/metrics', icon: 'CS', label: 'Chỉ số' },
  { to: '/history', icon: 'LS', label: 'Lịch sử' },
];

function MenuLink({ item, end = false, sub = false }) {
  return (
    <NavLink
      to={item.to}
      end={end}
      className={({ isActive }) => `nav-link ${sub ? 'nav-link-sub' : ''} ${isActive ? 'active' : ''}`}
    >
      <span className="nav-icon">{item.icon}</span>
      <span>{item.label}</span>
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span>MS</span> MedSeg
        </div>

        <MenuLink item={{ to: '/', icon: 'TQ', label: 'Tổng quan' }} end />

        <div className="nav-section-label">Phân tích</div>
        {ANALYSIS_NAV.map((item) => <MenuLink key={item.to} item={item} sub />)}

        <div className="nav-section-label">Báo cáo</div>
        {REPORT_NAV.map((item) => <MenuLink key={item.to} item={item} />)}
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Navigate to="/analyze/detect" replace />} />
          <Route path="/analyze/:mode" element={<Analyze />} />
          <Route path="/results/:id" element={<Results />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}
