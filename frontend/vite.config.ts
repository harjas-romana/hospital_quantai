import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  logLevel: 'error', // This will suppress warnings

  server:{
    host: true,
  },
  plugins: [
    tailwindcss(),
    react()],
})
