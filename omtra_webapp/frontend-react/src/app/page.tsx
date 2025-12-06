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
    </div>
  );
}

