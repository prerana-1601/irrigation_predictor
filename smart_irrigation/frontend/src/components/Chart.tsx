
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from 'recharts'

type Props = {
  data: any[]
  xKey: string
  yKeys: { key: string; label: string }[]
}

export default function Chart({ data, xKey, yKeys }: Props) {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          {yKeys.map((s) => (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
