/**
 * MedSeg API client.
 */
const BASE = 'http://localhost:8000/api';

async function post(endpoint, file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}${endpoint}`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Lỗi API: ${res.status}`);
  }
  return res.json();
}

async function get(endpoint) {
  const res = await fetch(`${BASE}${endpoint}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Lỗi API: ${res.status}`);
  }
  return res.json();
}

async function postJson(endpoint) {
  const res = await fetch(`${BASE}${endpoint}`, { method: 'POST' });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Lỗi API: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => get('/health'),
  detect: (file) => post('/detect', file),
  segment: (file) => post('/segment', file),
  pipeline: (file) => post('/pipeline', file),
  metrics: () => get('/metrics'),
  evaluationStatus: () => get('/metrics/evaluation-status'),
  runEvaluation: (kind, limit) => postJson(`/metrics/evaluate/${kind}?limit=${limit}`),
};
