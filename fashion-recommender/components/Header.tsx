'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';
import { Menu, X, Sparkles, User, Moon, Sun, Shirt, Camera, Heart, Palette } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  
  useEffect(() => {
    setMounted(true);
  }, []);

  const { theme, toggleTheme } = useTheme();

  const navItems = [
    { name: 'Home', href: '/', icon: Sparkles },
    { name: 'Analyze', href: '/analyze', icon: Camera },
    { name: 'Wardrobe', href: '/wardrobe', icon: Shirt },
    { name: 'Recommendations', href: '/recommendations', icon: Heart },
    { name: 'Mix & Match', href: '/mix-match', icon: Palette },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-md transition-theme">
      <div className="container-responsive">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2 group cursor-pointer">
            <div className="h-8 w-8 rounded-md fashion-gradient flex items-center justify-center group-hover:scale-110 group-hover:shadow-theme-md transition-all duration-300 group-hover:rotate-3">
              <Sparkles className="h-5 w-5 text-white group-hover:animate-pulse" />
            </div>
            <span className="hidden sm:inline-block font-bold text-xl bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent group-hover:from-primary-500 group-hover:to-secondary-500 transition-all duration-300">
              StyleAI
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const IconComponent = item.icon
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`group relative cursor-pointer flex items-center space-x-2 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 hover:scale-105 hover:shadow-theme-sm ${
                    isActive 
                      ? 'text-primary-600 bg-primary-50 dark:bg-primary-950/50 shadow-theme-sm scale-105' 
                      : 'text-foreground hover:text-primary-600 hover:bg-surface-hover'
                  }`}
                >
                  <IconComponent className={`h-4 w-4 group-hover:scale-110 transition-transform duration-300 ${
                    isActive ? 'scale-110 text-primary-600' : ''
                  }`} />
                  <span className={`group-hover:translate-x-0.5 transition-transform duration-300 ${
                    isActive ? 'translate-x-0.5 font-semibold' : ''
                  }`}>{item.name}</span>
                  <div className={`absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-primary-600 to-secondary-600 transition-transform duration-300 origin-left rounded-full ${
                    isActive ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100'
                  }`}></div>
                </Link>
              )
            })}
          </nav>

          {/* Theme Toggle & User Profile */}
          <div className="flex items-center space-x-3">
            <button
              onClick={mounted ? toggleTheme : undefined}
              className="group cursor-pointer relative inline-flex items-center justify-center rounded-xl p-3 transition-all duration-300 hover:bg-surface-hover hover:scale-110 hover:shadow-theme-md focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              <Sun
                className={`h-5 w-5 transition-all duration-300 group-hover:rotate-12 ${
                  mounted && theme === 'dark'
                    ? 'rotate-90 scale-0 opacity-0'
                    : 'rotate-0 scale-100 opacity-100'
                }`}
              />
              <Moon
                className={`absolute h-5 w-5 transition-all duration-300 group-hover:-rotate-12 ${
                  mounted && theme === 'dark'
                    ? 'rotate-0 scale-100 opacity-100'
                    : '-rotate-90 scale-0 opacity-0'
                }`}
              />
            </button>
            
            <Link
              href="/profile"
              className="group cursor-pointer btn-primary btn-sm flex items-center space-x-2 hover:scale-105 hover:shadow-theme-lg transition-all duration-300"
            >
              <User className="h-4 w-4 group-hover:scale-110 transition-transform duration-300" />
              <span className="hidden sm:inline group-hover:translate-x-0.5 transition-transform duration-300">Profile</span>
            </Link>

            
            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden group cursor-pointer btn-ghost btn-sm p-3 hover:scale-110 hover:bg-surface-hover hover:shadow-theme-md transition-all duration-300"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="h-5 w-5 group-hover:rotate-90 transition-transform duration-300" />
              ) : (
                <Menu className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-background animate-fade-in">
            <nav className="px-4 pt-4 pb-6 space-y-2">
              {navItems.map((item, index) => {
                const IconComponent = item.icon
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`group cursor-pointer flex items-center space-x-4 px-4 py-4 rounded-xl text-lg font-medium transition-all duration-300 hover:shadow-theme-sm ${
                      isActive 
                        ? 'text-primary-600 bg-primary-50 dark:bg-primary-950/50 shadow-theme-sm scale-105' 
                        : 'text-foreground hover:text-primary-600 hover:bg-surface-hover hover:scale-105'
                    }`}
                    onClick={() => setMobileMenuOpen(false)}
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <IconComponent className={`h-6 w-6 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300 ${
                      isActive ? 'scale-110 text-primary-600' : ''
                    }`} />
                    <span className={`group-hover:translate-x-1 transition-transform duration-300 ${
                      isActive ? 'translate-x-1 font-semibold' : ''
                    }`}>{item.name}</span>
                    <div className={`ml-auto transition-opacity duration-300 ${
                      isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    }`}>
                      <div className="w-2 h-2 rounded-full fashion-gradient"></div>
                    </div>
                  </Link>
                )
              })}
              <div className="pt-4">
                <Link
                  href="/profile"
                  className="group cursor-pointer btn-primary btn-lg flex items-center space-x-3 w-full justify-center hover:scale-105 hover:shadow-theme-lg transition-all duration-300"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <User className="h-5 w-5 group-hover:scale-110 transition-transform duration-300" />
                  <span className="group-hover:translate-x-0.5 transition-transform duration-300">Profile</span>
                </Link>
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}

