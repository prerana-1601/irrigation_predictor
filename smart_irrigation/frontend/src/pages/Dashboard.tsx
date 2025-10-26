import { useEffect, useState } from 'react';
import axios from 'axios';
import Card from '../components/Card';
import Button from '../components/Button';
import Chart from '../components/Chart';
import { Link, useNavigate } from 'react-router-dom';

type Reading = { timestamp: string; temperature: number; humidity: number; soil_moisture: number; rainfall: number };
type Field = { id: number; name: string; location: string; sensorData: Reading[]; irrigation_needed: boolean };
type User = { id:number; name:string; email:string; role:string; fields:number[] };

function getUser(): User | null {
  try { return JSON.parse(localStorage.getItem('user') || 'null'); } catch { return null; }
}

export default function Dashboard() {
  const me = getUser();
  const navigate = useNavigate();
  const [fields, setFields] = useState<Field[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  const isAdmin = me?.role === 'admin';

  useEffect(() => {
    (async () => {
      try {
        if (isAdmin) {
          const resUsers = await axios.get<User[]>('/api/admin/users');
          setUsers(resUsers.data.filter(u => u.role !== 'admin'));
        } else {
          const resFields = await axios.get<Field[]>('/api/fields/');
          const allowed = (me?.fields ?? []);
          setFields(resFields.data.filter(f => allowed.includes(f.id)));
        }
      } finally {
        setLoading(false);
      }
    })();
  }, [isAdmin]);

  if (!me) {
    navigate('/', { replace: true });
    return null;
  }
  if (loading) return <div className="max-w-6xl mx-auto px-4 py-10">Loading...</div>;

  // ---------- ADMIN DASHBOARD ----------
  if (isAdmin) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-hansal-charcoal">Dashboard (Admin)</h1>
            <p className="text-sm text-gray-500">Manage users and view system overview</p>
          </div>
          <div className="flex items-center gap-2">
            <Link to="/admin/fields">
              <Button>View All Fields</Button>
            </Link>
          </div>
        </div>

        <Card>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-2 pr-4">Name</th>
                  <th className="py-2 pr-4">Email</th>
                  <th className="py-2 pr-4">Role</th>
                  <th className="py-2">Fields</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr
                    key={u.id}
                    className="border-b last:border-b-0 hover:bg-gray-50 cursor-pointer"
                    onClick={() => navigate(`/admin/users/${u.id}/fields`)}
                    title="View user's fields"
                  >
                    <td className="py-2 pr-4 text-hansal-teal underline">{u.name}</td>
                    <td className="py-2 pr-4">{u.email}</td>
                    <td className="py-2 pr-4">{u.role}</td>
                    <td className="py-2">{u.fields?.length ? u.fields.join(', ') : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    );
  }

  // ---------- NORMAL USER DASHBOARD ----------
  const alertCount = fields.filter(f => f.irrigation_needed).length;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-hansal-charcoal">Dashboard</h1>
          <p className="text-sm text-gray-500">Overview of your fields and irrigation predictions</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => window.location.reload()}>Refresh</Button>
        </div>
      </div>

      {alertCount > 0 && (
        <Card>
          <div className="text-orange-700">⚠️ {alertCount} field(s) may need irrigation soon.</div>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {fields.map((f) => (
          <Card key={f.id}>
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold">{f.name}</div>
                <div className="text-sm text-gray-500">{f.location}</div>
              </div>
              <Link to={`/fields/${f.id}`} className="text-hansal-teal text-sm hover:underline">Details</Link>
            </div>
            <div className="mt-3 text-sm">
              Status:&nbsp;
              {f.irrigation_needed
                ? <span className="text-orange-700 font-medium">Irrigation Needed</span>
                : <span className="text-green-700 font-medium">OK</span>}
            </div>
            <div className="mt-3">
              <Chart
                data={f.sensorData.map(d => ({ ...d, time: new Date(d.timestamp).toLocaleString() }))}
                xKey="time"
                yKeys={[{ key: 'soil_moisture', label: 'Soil Moisture' }]}
              />
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
