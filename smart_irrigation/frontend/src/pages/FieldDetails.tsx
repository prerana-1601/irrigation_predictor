
import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import axios from 'axios'
import Card from '../components/Card'
import Chart from '../components/Chart'

type Field = {
  id: number
  name: string
  location: string
  sensorData: { timestamp: string; temperature: number; humidity: number; soil_moisture: number; rainfall: number }[]
  irrigation_needed: boolean
}

export default function FieldDetails() {
  const { id } = useParams()
  const [field, setField] = useState<Field | null>(null)

  useEffect(() => {
    (async () => {
      const res = await axios.get(`/api/fields/${id}`)
      setField(res.data)
    })()
  }, [id])

  if (!field) return <div className="max-w-6xl mx-auto px-4 py-10">Loading...</div>

  const data = field.sensorData.map(d => ({ ...d, time: new Date(d.timestamp).toLocaleString() }))

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-hansal-charcoal">{field.name}</h1>
          <p className="text-sm text-gray-500">{field.location}</p>
        </div>
        <Link to="/dashboard" className="text-hansal-teal hover:underline text-sm">← Back to Dashboard</Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <div className="font-medium mb-2">Soil Moisture (Last Readings)</div>
          <Chart data={data} xKey="time" yKeys={[{ key:'soil_moisture', label:'Soil Moisture' }]} />
        </Card>
        <Card>
          <div className="font-medium mb-2">Temperature & Humidity</div>
          <Chart data={data} xKey="time" yKeys={[{ key:'temperature', label:'Temperature' }, { key:'humidity', label:'Humidity' }]} />
        </Card>
        <Card>
          <div className="font-medium mb-2">Rainfall</div>
          <Chart data={data} xKey="time" yKeys={[{ key:'rainfall', label:'Rainfall' }]} />
        </Card>
        <Card>
          <div className="font-medium mb-2">Irrigation Status</div>
          <div className="text-sm">
            {field.irrigation_needed
              ? <span className="text-orange-700">Irrigation likely needed soon</span>
              : <span className="text-green-700">No irrigation needed</span>}
          </div>
        </Card>
      </div>
    </div>
  )
}
