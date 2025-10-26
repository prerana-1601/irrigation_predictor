import { createBrowserRouter } from 'react-router-dom';
import App from './App';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import FieldDetails from './pages/FieldDetails';
import AdminFields from './pages/AdminFields';
import AdminUserFields from './pages/AdminUserFields';
import { RequireAuth, RequireAdmin } from './auth';

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { path: '/', element: <Login /> },
      { path: '/signup', element: <Signup /> },

      {
        element: <RequireAuth />,
        children: [
          { path: '/dashboard', element: <Dashboard /> },
          { path: '/fields/:id', element: <FieldDetails /> },

          { element: <RequireAdmin />, children: [
            { path: '/admin/fields', element: <AdminFields /> },
            { path: '/admin/users/:userId/fields', element: <AdminUserFields /> },  // NEW
          ]},
        ],
      },
    ],
  },
]);

export default router;
