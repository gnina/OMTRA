'use client';

export function Footer() {
  return (
    <footer className="bg-slate-900 text-white">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">🧪</span>
              <span className="text-xl font-bold">OMTRA</span>
            </div>
            <p className="text-sm text-slate-400 max-w-md">
              Precision-grade molecular sampling platform for generating novel molecular structures 
              using deep learning. Engineered for applied research teams.
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider mb-4">Platform</h3>
            <ul className="space-y-2">
              <li>
                <a href="#workspace" className="text-sm text-slate-400 hover:text-white transition-colors">
                  Workspace
                </a>
              </li>
              <li>
                <a href="#help" className="text-sm text-slate-400 hover:text-white transition-colors">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#about" className="text-sm text-slate-400 hover:text-white transition-colors">
                  About
                </a>
              </li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider mb-4">Resources</h3>
            <ul className="space-y-2">
              <li>
                <a 
                  href="https://github.com/gnina/OMTRA" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-white transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a 
                  href="https://github.com/gnina/OMTRA/issues" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-white transition-colors"
                >
                  Report Issues
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-slate-400 hover:text-white transition-colors">
                  Examples
                </a>
              </li>
            </ul>
          </div>
        </div>
        
        <div className="mt-8 border-t border-slate-800 pt-8">
          <p className="text-sm text-slate-400 text-center">
            © {new Date().getFullYear()} OMTRA Platform. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}








