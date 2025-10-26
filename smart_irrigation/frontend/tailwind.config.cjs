
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        hansal: {
          teal: '#008080',
          tealLight: '#70B8B8',
          charcoal: '#263238'
        }
      },
      boxShadow: {
        card: '0 10px 20px rgba(0,0,0,0.06)'
      },
      borderRadius: {
        '2xl': '1rem'
      }
    }
  },
  plugins: []
}
