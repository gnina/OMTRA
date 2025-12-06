'use client';

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { JobStatus } from '@/types';
import { Loader2, Download, ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react';
import { MolecularViewer } from './MolecularViewer';
import { MetricsTable } from './MetricsTable';
import { InteractionDiagram2D, prefetchInteractionDiagram } from './InteractionDiagram2D';

interface JobViewerProps {
  jobId: string;
  onBack: () => void;
}

export function JobViewer({ jobId, onBack }: JobViewerProps) {
  const [moleculeIndex, setMoleculeIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'3d' | '2d'>('3d');
  

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['job-status', jobId],
    queryFn: () => apiClient.getJobStatus(jobId),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state === 'QUEUED' || state === 'RUNNING' ? 2000 : false;
    },
  });

  const { data: result, isLoading: resultLoading } = useQuery({
    queryKey: ['job-result', jobId],
    queryFn: () => apiClient.getJobResult(jobId),
    enabled: status?.state === 'SUCCEEDED',
  });

  const isLoading = statusLoading || (status?.state === 'SUCCEEDED' && resultLoading);

  // Background: once results exist, start PoseView (2D) generation for all samples,
  // but do NOT block or sequence this before any 3D loading.
  useEffect(() => {
    if (!result) return;
    const mode = result.params.sampling_mode;
    if (mode !== 'Protein-conditioned' && mode !== 'Protein+Pharmacophore-conditioned') {
      return;
    }

    const sdfFilesForPrefetch = result.artifacts
      .filter((a) => a.filename.startsWith('sample_') && a.filename.endsWith('.sdf'))
      .sort((a, b) => {
        const numA = parseInt(a.filename.match(/\d+/)?.[0] || '0');
        const numB = parseInt(b.filename.match(/\d+/)?.[0] || '0');
        return numA - numB;
      });

    if (!sdfFilesForPrefetch.length) return;

    sdfFilesForPrefetch.forEach((artifact) => {
      prefetchInteractionDiagram(jobId, artifact.filename).catch(() => {
        // Errors are cached and surfaced in the 2D component; ignore here
      });
    });
  }, [jobId, result]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
      </div>
    );
  }

  if (!status) {
    return (
      <div className="p-8 text-center text-slate-500">
        <p>Job not found</p>
        <button
          onClick={onBack}
          className="mt-4 px-5 py-2.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors"
        >
          Back to Jobs
        </button>
      </div>
    );
  }

  const jobState = status.state;

  if (jobState === 'QUEUED' || jobState === 'RUNNING') {
    return (
      <div className="space-y-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="font-medium">Back to Jobs</span>
        </button>
        <div className="p-8 text-center bg-slate-50/50 rounded-2xl shadow-sm">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-lg font-semibold text-slate-900">
            Job is {jobState.toLowerCase()}
          </p>
          <p className="text-sm text-slate-600 mt-2">
            Results will appear when complete
          </p>
          {status.progress > 0 && (
            <div className="mt-6 max-w-md mx-auto">
              <div className="w-full bg-slate-200 rounded-full h-2.5">
                <div
                  className="bg-primary-600 h-2.5 rounded-full transition-all"
                  style={{ width: `${status.progress}%` }}
                />
              </div>
              <p className="text-xs text-slate-500 mt-2">{status.progress}% complete</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (jobState === 'FAILED') {
    return (
      <div className="space-y-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="font-medium">Back to Jobs</span>
        </button>
        <div className="p-6 bg-red-50/70 rounded-2xl shadow-sm">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Job Failed</h3>
          <p className="text-sm text-red-700">{status.message || 'Unknown error'}</p>
        </div>
      </div>
    );
  }

  if (jobState !== 'SUCCEEDED' || !result) {
    return (
      <div className="p-8 text-center text-slate-500">
        <p>No results available</p>
        <button
          onClick={onBack}
          className="mt-4 px-5 py-2.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors"
        >
          Back to Jobs
        </button>
      </div>
    );
  }

  const sdfFiles = result.artifacts
    .filter((a) => a.filename.startsWith('sample_') && a.filename.endsWith('.sdf'))
    .sort((a, b) => {
      const numA = parseInt(a.filename.match(/\d+/)?.[0] || '0');
      const numB = parseInt(b.filename.match(/\d+/)?.[0] || '0');
      return numA - numB;
    });

  if (sdfFiles.length === 0) {
    return (
      <div className="space-y-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="font-medium">Back to Jobs</span>
        </button>
        <div className="p-8 text-center text-slate-500 bg-slate-50/50 rounded-2xl shadow-sm">
          <p>No molecule files generated</p>
        </div>
      </div>
    );
  }

  const currentFile = sdfFiles[Math.min(moleculeIndex, sdfFiles.length - 1)];
  
  console.log(`[JobViewer] Render: moleculeIndex=${moleculeIndex}, filename=${currentFile?.filename}, activeTab=${activeTab}`);
  console.warn(`[JobViewer] WARN RENDER: moleculeIndex=${moleculeIndex}, filename=${currentFile?.filename}, activeTab=${activeTab}`);
  
  // Log when we're about to render InteractionDiagram2D
  if (activeTab === '2d' && (result.params.sampling_mode === 'Protein-conditioned' || result.params.sampling_mode === 'Protein+Pharmacophore-conditioned')) {
    console.warn(`[JobViewer] WARN: About to render InteractionDiagram2D with filename=${currentFile?.filename}, key=diagram-${jobId}-${currentFile?.filename}-${moleculeIndex}`);
  } // Use warn so it's more visible

  const handleDownloadAll = async () => {
    try {
      const blob = await apiClient.downloadAllOutputs(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${jobId}_outputs.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Download failed');
    }
  };

  return (
    <div className="space-y-6">
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="font-medium">Back to Jobs</span>
      </button>

      <div className="border-b border-slate-200/60 pb-4">
        <h2 className="text-2xl font-semibold text-slate-900 mb-2">
          Job Details: {jobId.substring(0, 12)}...
        </h2>
        <div className="text-sm text-slate-600">
          {result.params.sampling_mode} • {result.params.n_samples} samples •{' '}
          {result.params.steps} steps
        </div>
      </div>

      {/* Navigation controls above viewer */}
      <div className="flex items-center justify-center gap-3 bg-slate-50/50 rounded-2xl p-4 relative shadow-sm">
        {/* Left Arrow */}
        <button
          onClick={() => {
            const newIndex = Math.max(0, moleculeIndex - 1);
            const newFile = sdfFiles[Math.min(newIndex, sdfFiles.length - 1)];
            console.log(`[JobViewer] Back button clicked: ${moleculeIndex} -> ${newIndex}`);
            console.log(`[JobViewer] Current file: ${currentFile?.filename}, New file: ${newFile?.filename}`);
            console.warn(`[JobViewer] WARN: Setting moleculeIndex to ${newIndex}, filename will be ${newFile?.filename}`);
            setMoleculeIndex(newIndex);
          }}
          disabled={moleculeIndex === 0}
          className="absolute left-4 p-3 bg-white rounded-full shadow-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 transition-all hover:scale-110"
        >
          <ChevronLeft className="w-6 h-6 text-slate-700" />
        </button>
        
        {/* Right Arrow */}
        <button
          onClick={() => {
            const newIndex = Math.min(sdfFiles.length - 1, moleculeIndex + 1);
            console.log(`[JobViewer] Forward button clicked: ${moleculeIndex} -> ${newIndex}`);
            setMoleculeIndex(newIndex);
          }}
          disabled={moleculeIndex >= sdfFiles.length - 1}
          className="absolute right-4 p-3 bg-white rounded-full shadow-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-50 transition-all hover:scale-110"
        >
          <ChevronRight className="w-6 h-6 text-slate-700" />
        </button>
        
        <div className="text-center">
          <span className="font-semibold text-slate-900">
            Sample {moleculeIndex} of {sdfFiles.length - 1}
          </span>
        </div>
        <input
          type="number"
          min={0}
          max={sdfFiles.length - 1}
          value={moleculeIndex}
          onChange={(e) => {
            const val = parseInt(e.target.value);
            if (!isNaN(val) && val >= 0 && val < sdfFiles.length) {
              setMoleculeIndex(val);
            }
          }}
          className="w-20 px-3 py-2 border border-slate-200 rounded-xl text-center bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 shadow-sm"
        />
        <button
          onClick={handleDownloadAll}
          className="flex items-center gap-2 px-5 py-2.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-sm hover:shadow-md"
        >
          <Download className="w-4 h-4" />
          Download All
        </button>
      </div>

      {/* Viewer */}
      {result.params.sampling_mode === 'Protein-conditioned' ||
      result.params.sampling_mode === 'Protein+Pharmacophore-conditioned' ? (
        <div className="rounded-2xl bg-white shadow-sm">
          {/* Tabs */}
          <div className="flex border-b border-slate-200/60">
            <button
              onClick={() => setActiveTab('3d')}
              className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === '3d'
                  ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50/50'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              3D Viewer
            </button>
            <button
              onClick={() => setActiveTab('2d')}
              className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${
                activeTab === '2d'
                  ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50/50'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
              }`}
            >
              2D Interaction Diagram
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="p-4">
            {activeTab === '3d' ? (
              <MolecularViewer
                jobId={jobId}
                filename={currentFile.filename}
                samplingMode={result.params.sampling_mode}
              />
            ) : (
              <InteractionDiagram2D 
                jobId={jobId} 
                filename={currentFile.filename} 
              />
            )}
          </div>
        </div>
      ) : (
        <div className="rounded-2xl p-4 bg-white shadow-sm">
          <MolecularViewer
            jobId={jobId}
            filename={currentFile.filename}
            samplingMode={result.params.sampling_mode}
          />
        </div>
      )}

      {/* Metrics Table */}
      <MetricsTable
        jobId={jobId}
        samplingMode={result.params.sampling_mode}
        onRowSelect={setMoleculeIndex}
        selectedIndex={moleculeIndex}
      />
    </div>
  );
}


