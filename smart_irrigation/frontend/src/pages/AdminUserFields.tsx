import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';
import Card from '../components/Card';
import Chart from '../components/Chart';
import Button from '../components/Button';

type Reading = {
  timestamp: string;
  temperature: number;
  humidity: number;
  soil_moisture: number;
  rainfall: number;
};

type Field = {
  id: number;
  name: string;
  location: string;
  sensorData: Reading[];
  irrigation_needed: boolean;
  owner_user_id?: number;
  owner_name?: string;
};

type User = { id: number; name: string; email: string; role: string; fields: number[] };

export default function AdminUserFields() {
  const { userId } = useParams<{ userId: string }>();
  const [fields, setFields] = useState<Field[]>([]);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const res = await axios.get<{ user: User; fields: Field[] }>(
        `/api/admin/users/${userId}/fields`
      );
      setUser(res.data.user);
      setFields(res.data.fields);
      setLoading(false);
    })();
  }, [userId]);

  if (loading) {
    return <div className="max-w-6xl mx-auto px-4 py-10">Loading...</div>;
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-hansal-charcoal">
            Fields for {user?.name ?? `User #${userId}`}
          </h1>
          <p className="text-sm text-gray-500">{user?.email}</p>
        </div>
        <Link to="/dashboard">
          <Button>← Back to Admin</Button>
        </Link>
      </div>

      {fields.length === 0 && (
        <Card>
          <div className="text-sm text-gray-600">No fields found for this user.</div>
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
              {/* NEW: Details link just like user dashboard */}
              <Link
                to={`/fields/${f.id}`}
                className="text-hansal-teal text-sm hover:underline"
                title="Open field details"
              >
                Details
              </Link>
            </div>

            <div className="mt-3 text-sm">
              Status:{' '}
              {f.irrigation_needed ? (
                <span className="text-orange-700 font-medium">Irrigation Needed</span>
              ) : (
                <span className="text-green-700 font-medium">OK</span>
              )}
            </div>

            <div className="mt-3">
              <Chart
                data={f.sensorData.map((d) => ({
                  ...d,
                  time: new Date(d.timestamp).toLocaleString(),
                }))}
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
