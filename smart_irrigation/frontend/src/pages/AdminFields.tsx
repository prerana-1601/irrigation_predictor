import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import Card from '../components/Card';
import Button from '../components/Button';
import Chart from '../components/Chart';

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
  irrigation_needed: boolean;
  sensorData: Reading[];
  owner_user_id: number;
  owner_name?: string;
};

type User = { id: number; name: string; email: string; role: string; fields: number[] };

export default function AdminFields() {
  const [fields, setFields] = useState<Field[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);

  // create form state
  const [newName, setNewName] = useState('');
  const [newLocation, setNewLocation] = useState('');
  const [newOwner, setNewOwner] = useState<number | ''>('');

  // per-card edit state
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState('');
  const [editLocation, setEditLocation] = useState('');
  const [editOwner, setEditOwner] = useState<number | ''>('');

  useEffect(() => {
    (async () => {
      try {
        const [fRes, uRes] = await Promise.all([
          axios.get<Field[]>('/api/admin/fields'),
          axios.get<User[]>('/api/admin/users'),
        ]);
        setFields(fRes.data);
        setUsers(uRes.data);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const userOptions = useMemo(
    () => users.map((u) => ({ value: u.id, label: `${u.name} (${u.email})` })),
    [users]
  );

  async function refresh() {
    const fRes = await axios.get<Field[]>('/api/admin/fields');
    setFields(fRes.data);
  }

  async function handleCreate() {
    if (!newName || !newOwner) return;
    await axios.post('/api/admin/fields', {
      name: newName,
      location: newLocation,
      owner_user_id: Number(newOwner),
      irrigation_needed: false,
      sensorData: [],
    });
    setNewName('');
    setNewLocation('');
    setNewOwner('');
    await refresh();
  }

  function startEdit(f: Field) {
    setEditingId(f.id);
    setEditName(f.name);
    setEditLocation(f.location);
    setEditOwner(f.owner_user_id);
  }

  function cancelEdit() {
    setEditingId(null);
    setEditName('');
    setEditLocation('');
    setEditOwner('');
  }

  async function saveEdit(fieldId: number) {
    await axios.put(`/api/admin/fields/${fieldId}`, {
      name: editName,
      location: editLocation,
      owner_user_id: Number(editOwner),
    });
    cancelEdit();
    await refresh();
  }

  async function remove(fieldId: number) {
    if (!confirm('Delete this field?')) return;
    await axios.delete(`/api/admin/fields/${fieldId}`);
    await refresh();
  }

  if (loading) {
    return <div className="max-w-6xl mx-auto px-4 py-8">Loading…</div>;
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-hansal-charcoal">All Fields</h1>
          <p className="text-sm text-gray-500">Create, assign, edit, or delete fields</p>
        </div>
        <Button onClick={refresh}>Refresh</Button>
      </div>

      {/* Create new field */}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
          <div className="md:col-span-2">
            <label className="text-sm text-gray-600">Name</label>
            <input
              className="w-full border rounded-lg px-3 py-2"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Field name"
            />
          </div>
          <div className="md:col-span-2">
            <label className="text-sm text-gray-600">Location</label>
            <input
              className="w-full border rounded-lg px-3 py-2"
              value={newLocation}
              onChange={(e) => setNewLocation(e.target.value)}
              placeholder="Location"
            />
          </div>
          <div>
            <label className="text-sm text-gray-600">Owner</label>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={newOwner}
              onChange={(e) => setNewOwner(e.target.value ? Number(e.target.value) : '')}
            >
              <option value="">Select owner…</option>
              {userOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
          <div className="md:col-span-5">
            <Button onClick={handleCreate} disabled={!newName || !newOwner}>
              + Create Field
            </Button>
          </div>
        </div>
      </Card>

      {/* Grid of field cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {fields.map((f) => {
          const isEditing = editingId === f.id;
          return (
            <Card key={f.id}>
              {!isEditing ? (
                <>
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-semibold">{f.name}</div>
                      <div className="text-sm text-gray-500">{f.location}</div>
                      <div className="text-sm text-gray-600 mt-1">
                        Owner:{' '}
                        <span className="font-medium">
                          {f.owner_name || `User #${f.owner_user_id}`}
                        </span>
                      </div>
                    </div>
                    {/* NEW: Details link added here */}
                    <div className="flex items-center gap-3">
                      <Link
                        to={`/fields/${f.id}`}
                        className="text-hansal-teal text-sm hover:underline"
                        title="Open field details"
                      >
                        Details
                      </Link>
                      <Button onClick={() => startEdit(f)}>Edit</Button>
                      <Button onClick={() => remove(f.id)}>Delete</Button>
                    </div>
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
                </>
              ) : (
                <>
                  <div className="grid grid-cols-1 gap-3">
                    <div>
                      <label className="text-sm text-gray-600">Name</label>
                      <input
                        className="w-full border rounded-lg px-3 py-2"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="text-sm text-gray-600">Location</label>
                      <input
                        className="w-full border rounded-lg px-3 py-2"
                        value={editLocation}
                        onChange={(e) => setEditLocation(e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="text-sm text-gray-600">Owner</label>
                      <select
                        className="w-full border rounded-lg px-3 py-2"
                        value={editOwner}
                        onChange={(e) => setEditOwner(e.target.value ? Number(e.target.value) : '')}
                      >
                        {userOptions.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={() => saveEdit(f.id)} disabled={!editName || !editOwner}>
                        Save
                      </Button>
                      <Button onClick={cancelEdit}>Cancel</Button>
                    </div>
                  </div>
                </>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
