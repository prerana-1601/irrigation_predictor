import { Navigate, Outlet } from 'react-router-dom';

export function getUser() {
  try { return JSON.parse(localStorage.getItem('user') || 'null'); } catch { return null; }
}
export function isAuthed() {
  return !!localStorage.getItem('token') && !!getUser();
}

export function RequireAuth() {
  if (!isAuthed()) return <Navigate to="/" replace />;
  return <Outlet />;
}

export function RequireAdmin() {
  const user = getUser();
  if (!isAuthed() || user?.role !== 'admin') return <Navigate to="/dashboard" replace />;
  return <Outlet />;
}
