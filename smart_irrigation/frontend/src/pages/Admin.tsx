
import { useEffect, useState } from 'react'
import axios from 'axios'
import Card from '../components/Card'

type User = { id:number; name:string; email:string; role:string; fields:number[] }

export default function Admin() {
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      const res = await axios.get('/api/admin/users')
      setUsers(res.data)
      setLoading(false)
    })()
  }, [])

  if (loading) return <div className="max-w-6xl mx-auto px-4 py-10">Loading...</div>

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-hansal-charcoal">Admin Panel</h1>
        <p className="text-sm text-gray-500">Manage users and review system overview</p>
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
                <tr key={u.id} className="border-b last:border-b-0">
                  <td className="py-2 pr-4">{u.name}</td>
                  <td className="py-2 pr-4">{u.email}</td>
                  <td className="py-2 pr-4">{u.role}</td>
                  <td className="py-2">{u.fields.join(', ') || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
