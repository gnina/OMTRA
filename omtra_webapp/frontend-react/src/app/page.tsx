'use client';

import { useState, useEffect } from 'react';
import { JobSubmissionForm } from '@/components/JobSubmissionForm';
import { JobList } from '@/components/JobList';
import { JobViewer } from '@/components/JobViewer';
import { HelpTab } from '@/components/HelpTab';

export default function HomePage() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'jobs' | 'help'>('jobs');

  // Scroll to top when a job is selected
  useEffect(() => {
    if (selectedJobId) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [selectedJobId]);

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200/60 bg-white/80 backdrop-blur-md shadow-sm">
        <nav className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🧪</span>
              <span className="text-xl font-bold text-slate-900">OMTRA</span>
            </div>
            <div className="flex items-center gap-6">
              <button
                onClick={() => {
                  setActiveTab('jobs');
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`text-sm font-medium transition-colors ${
                  activeTab === 'jobs'
                    ? 'text-primary-600 font-semibold'
                    : 'text-slate-700 hover:text-primary-600'
                }`}
              >
                Workspace
              </button>
              <button
                onClick={() => {
                  setActiveTab('help');
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`text-sm font-medium transition-colors ${
                  activeTab === 'help'
                    ? 'text-primary-600 font-semibold'
                    : 'text-slate-700 hover:text-primary-600'
                }`}
              >
                Help
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Disclaimer Banner */}
      <div className="bg-amber-50 border-b border-amber-200/60">
        <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <span className="text-amber-600 text-lg">⚠️</span>
            </div>
            <div className="flex-1 text-sm text-amber-900">
              <p className="font-medium mb-1">Disclaimer</p>
              <p className="text-amber-800/90"> 
                This is a proof of concept server for demonstration purposes. 
                All results are public, and jobs are automatically removed after 48 hours.
                For issues, feature requests, or to contribute, please visit our{' '}
                <a 
                  href="https://github.com/gnina/OMTRA" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="font-semibold underline hover:text-amber-900 transition-colors"
                >
                  GitHub repository
                </a>
                {' '}and open an issue.
              </p>
            </div>
          </div>
        </div>
      </div>

      <main className="flex-1">
        {/* Workspace Section */}
        <section id="workspace" className="py-8">
          <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8">
            {activeTab === 'jobs' && (
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
                <div className="lg:col-span-1">
                  <div className="bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6 lg:sticky lg:top-24 lg:max-h-[calc(100vh-8rem)] lg:overflow-y-auto">
                    <JobSubmissionForm onJobSubmitted={setSelectedJobId} />
                  </div>
                </div>

                <div className="lg:col-span-2">
                  <div className="bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6">
                    {selectedJobId ? (
                      <JobViewer jobId={selectedJobId} onBack={() => setSelectedJobId(null)} />
                    ) : (
                      <JobList onJobSelect={setSelectedJobId} />
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'help' && (
              <div id="help" className="bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6">
                <HelpTab />
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 text-white mt-12">
        <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
          <div className="border-t border-slate-800 pt-6">
            <p className="text-sm text-slate-400 text-center">
              © {new Date().getFullYear()} OMTRA.{' '}
              <a 
                href="https://github.com/gnina/OMTRA" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-slate-300 hover:text-white underline transition-colors"
              >
                View on GitHub
              </a>
              {' '}•{' '}
              <a 
                href="https://github.com/gnina/OMTRA/issues" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-slate-300 hover:text-white underline transition-colors"
              >
                Report Issues
              </a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

