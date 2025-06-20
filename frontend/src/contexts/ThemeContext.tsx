import React, { createContext, useContext, useEffect } from 'react';

interface ThemeContextType {
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  // No-op function since we always use dark mode
  const toggleTheme = () => {
    // No operation - always dark mode
    console.log('Theme toggle requested, but application is dark-mode only');
  };

  useEffect(() => {
    // Always apply dark theme
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <ThemeContext.Provider value={{ toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}; 