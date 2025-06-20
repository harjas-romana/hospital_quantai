/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Enable class-based dark mode
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', 'Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-in': 'slideIn 0.5s ease-in-out',
        'pulse-slow': 'pulse 2s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(26, 59, 92, 0.3)' },
          '100%': { boxShadow: '0 0 30px rgba(26, 59, 92, 0.6)' },
        }
      },
      colors: {
        'police-blue': '#1a3b5c',
        'police-gold': '#d4af37',
        // Dark theme colors
        'dark-bg': '#000000',
        'dark-surface': '#111111',
        'dark-card': '#1a1a1a',
        'dark-border': '#333333',
        'dark-text': '#ffffff',
        'dark-text-secondary': '#a0a0a0',
        // Light theme colors
        'light-bg': '#f8f9fa',
        'light-surface': '#ffffff',
        'light-card': '#ffffff',
        'light-border': '#e5e7eb',
        'light-text': '#1f2937',
        'light-text-secondary': '#6b7280',
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
} 