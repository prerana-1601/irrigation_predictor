import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import logo from './assets/logo.svg';

export default function App() {
  const loc = useLocation();
  const navigate = useNavigate();
  const hideNav = loc.pathname === '/' || loc.pathname === '/signup';

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/', { replace: true });
  };

  return (
    <div className="min-h-screen">
      {!hideNav && (
        <nav className="bg-white shadow sticky top-0 z-10">
          <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src={logo} alt="Hansal Logo" className="h-8" />
            </div>
            <div className="flex items-center gap-4 text-sm">
              <Link to="/dashboard" className="hover:text-hansal-teal">Dashboard</Link>
              {/**/}
              <button onClick={logout} className="text-gray-600 hover:text-hansal-teal">Log out</button>
            </div>
          </div>
        </nav>
      )}

      <Outlet />

      {!hideNav && (
        <footer className="bg-gray-100 mt-12">
          <div className="max-w-6xl mx-auto px-4 py-6 text-sm text-gray-600">
            © {new Date().getFullYear()} Hansal Smart Irrigation
          </div>
        </footer>
      )}
    </div>
  );
}
