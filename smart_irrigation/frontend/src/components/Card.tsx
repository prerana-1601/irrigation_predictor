
import { PropsWithChildren } from 'react'

export default function Card({ children }: PropsWithChildren) {
  return (
    <div className="bg-white rounded-2xl shadow-card p-4">{children}</div>
  )
}
