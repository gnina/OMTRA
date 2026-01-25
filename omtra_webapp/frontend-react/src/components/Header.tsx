'use client';

import { useState } from 'react';
import { Menu, X } from 'lucide-react';

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
      <nav className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8" aria-label="Top">
        <div className="flex w-full items-center justify-between py-4">
          <div className="flex items-center">
            <a href="/" className="flex items-center gap-2">
              <span className="text-2xl">🧪</span>
              <span className="text-xl font-bold text-slate-900">OMTRA</span>
            </a>
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:gap-8">
            <a 
              href="#workspace" 
              onClick={(e) => {
                e.preventDefault();
                document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="text-sm font-medium text-slate-700 hover:text-primary-600 transition-colors"
            >
              Workspace
            </a>
            <a 
              href="#help" 
              onClick={(e) => {
                e.preventDefault();
                const helpSection = document.getElementById('help') || document.querySelector('[id="help"]');
                helpSection?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="text-sm font-medium text-slate-700 hover:text-primary-600 transition-colors"
            >
              Documentation
            </a>
            <a 
              href="#about" 
              onClick={(e) => {
                e.preventDefault();
                document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="text-sm font-medium text-slate-700 hover:text-primary-600 transition-colors"
            >
              About
            </a>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md p-2 text-slate-700 hover:bg-slate-100 hover:text-slate-900"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <span className="sr-only">Open main menu</span>
              {mobileMenuOpen ? (
                <X className="h-6 w-6" aria-hidden="true" />
              ) : (
                <Menu className="h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-slate-200 py-4">
            <div className="space-y-2 px-2">
              <a
                href="#workspace"
                className="block rounded-md px-3 py-2 text-base font-medium text-slate-700 hover:bg-slate-50 hover:text-primary-600"
                onClick={(e) => {
                  e.preventDefault();
                  setMobileMenuOpen(false);
                  setTimeout(() => {
                    document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                }}
              >
                Workspace
              </a>
              <a
                href="#help"
                className="block rounded-md px-3 py-2 text-base font-medium text-slate-700 hover:bg-slate-50 hover:text-primary-600"
                onClick={(e) => {
                  e.preventDefault();
                  setMobileMenuOpen(false);
                  setTimeout(() => {
                    const helpSection = document.getElementById('help') || document.querySelector('[id="help"]');
                    helpSection?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                }}
              >
                Documentation
              </a>
              <a
                href="#about"
                className="block rounded-md px-3 py-2 text-base font-medium text-slate-700 hover:bg-slate-50 hover:text-primary-600"
                onClick={(e) => {
                  e.preventDefault();
                  setMobileMenuOpen(false);
                  setTimeout(() => {
                    document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
                  }, 100);
                }}
              >
                About
              </a>
            </div>
          </div>
        )}
      </nav>
    </header>
  );
}

