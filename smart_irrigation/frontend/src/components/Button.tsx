
import { ButtonHTMLAttributes } from 'react'

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'outline'
}

export default function Button({ variant='primary', className='', ...props }: Props) {
  const base = 'px-4 py-2 rounded-2xl transition transform active:scale-[0.98]'
  const styles = variant === 'primary'
    ? 'bg-hansal-teal text-white hover:brightness-110 shadow'
    : 'border border-hansal-teal text-hansal-teal bg-white hover:bg-hansal-teal hover:text-white'
  return <button className={`${base} ${styles} ${className}`} {...props} />
}
