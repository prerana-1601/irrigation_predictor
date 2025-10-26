
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Button from '../components/Button'
import axios from 'axios'

export default function Login() {
  const [email, setEmail] = useState('demo@hansal.ai')
  const [password, setPassword] = useState('hansal123')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      const res = await axios.post('/api/auth/login', { email, password })
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('user', JSON.stringify(res.data.user))
      navigate('/dashboard')
    } catch (err: any) {
      setError(err?.response?.data?.error || 'Login failed')
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-hansal-teal/10 to-hansal-teal/30 flex items-center justify-center p-4">
      <form onSubmit={submit} className="bg-white rounded-2xl shadow-card p-6 w-full max-w-md">
        <div className="flex justify-center mb-4">
          <img src="/logo.svg" alt="Hansal Logo" className="h-8" />
        </div>
        <h1 className="text-xl font-semibold text-hansal-charcoal mb-2">Welcome back</h1>
        <p className="text-sm text-gray-500 mb-4">Log in to Hansal Smart Irrigation</p>
        {error && <div className="text-red-600 text-sm mb-2">{error}</div>}
        <div className="space-y-3">
          <input value={email} onChange={e=>setEmail(e.target.value)} type="email" placeholder="Email"
            className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-hansal-teal" />
          <input value={password} onChange={e=>setPassword(e.target.value)} type="password" placeholder="Password"
            className="w-full border rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-hansal-teal" />
          <Button type="submit" className="w-full">Log In</Button>
          <div className="text-sm text-center">
            Don&apos;t have an account? <Link to="/signup" className="text-hansal-teal hover:underline">Sign up</Link>
          </div>
        </div>
      </form>
    </div>
  )
}
