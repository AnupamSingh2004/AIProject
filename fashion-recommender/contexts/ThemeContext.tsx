'use client'

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

interface ThemeProviderProps {
  children: ReactNode
  defaultTheme?: Theme
  storageKey?: string
}

export function ThemeProvider({
  children,
  defaultTheme = 'light',
  storageKey = 'fashion-theme',
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(defaultTheme)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    // Get theme from localStorage or system preference
    const stored = localStorage.getItem(storageKey) as Theme
    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
    
    const initialTheme = stored || systemTheme
    setThemeState(initialTheme)
    setMounted(true)
  }, [storageKey])

  useEffect(() => {
    if (!mounted) return

    const root = window.document.documentElement
    
    // Remove previous theme classes
    root.classList.remove('light', 'dark')
    
    // Add new theme class
    root.classList.add(theme)
    
    // Store theme preference
    localStorage.setItem(storageKey, theme)
  }, [theme, mounted, storageKey])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
  }

  const toggleTheme = () => {
    setThemeState((prevTheme: Theme) => prevTheme === 'light' ? 'dark' : 'light')
  }

  const contextValue = {
    theme,
    setTheme,
    toggleTheme,
  }

  return (
    <ThemeContext.Provider value={contextValue}>
      <div suppressHydrationWarning>
        {children}
      </div>
    </ThemeContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeContext)
  
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  
  return context
}

// Safe hook that doesn't throw during SSR
export const useThemeSafe = () => {
  const context = useContext(ThemeContext)
  return context || { theme: 'light' as Theme, setTheme: () => {}, toggleTheme: () => {} }
}