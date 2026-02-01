'use client';

import { useState, useEffect, useMemo } from 'react';
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
  const [inputValue, setInputValue] = useState('0');
  const [activeTab, setActiveTab] = useState<'3d' | '2d'>('3d');
  const [prefetchedMolecules, setPrefetchedMolecules] = useState<Record<string, string>>({});

  // Structural data state
  const [proteinData, setProteinData] = useState<{ text: string, format: string } | null>(null);
  const [pharmData, setPharmData] = useState<{ text: string, extension: string } | null>(null);
  const [hasLoadedStructural, setHasLoadedStructural] = useState(false);

  // Sync inputValue when moleculeIndex changes from other sources (arrows, table selection)
  useEffect(() => {
    setInputValue(String(moleculeIndex));
  }, [moleculeIndex]);

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

  const { data: inputFiles } = useQuery({
    queryKey: ['job-inputs', jobId],
    queryFn: () => apiClient.listInputFiles(jobId),
  });

  const sdfFiles = useMemo(() => {
    if (!result?.artifacts) return [];
    return result.artifacts
      .filter((a: any) => a.filename.startsWith('sample_') && a.filename.endsWith('.sdf'))
      .sort((a: any, b: any) => {
        const numA = parseInt(a.filename.match(/\d+/)?.[0] || '0');
        const numB = parseInt(b.filename.match(/\d+/)?.[0] || '0');
        return numA - numB;
      });
  }, [result]);

  // Bulk Pre-fetch result molecules, structural data, and interaction diagrams
  useEffect(() => {
    if (!result?.artifacts || sdfFiles.length === 0 || !status) return;

    const prefetchEverything = async () => {
      // 1. Fetch Structural Data (Protein & Pharma) if not already loaded
      if (!hasLoadedStructural) {
        try {
          // Use the inputFiles from useQuery if available
          const inputs = inputFiles || await apiClient.listInputFiles(jobId);
          const mode = (result.params as any).docking_mode || result.params.sampling_mode || 'Unconditional';

          const needsProtein = ['Protein-conditioned', 'Protein+Pharmacophore-conditioned', 'Rigid Docking', 'Rigid Docking + Pharmacophore'].includes(mode) || mode.toLowerCase().includes('protein') || mode.toLowerCase().includes('docking');
          const needsPharmacophore = ['Pharmacophore-conditioned', 'Protein+Pharmacophore-conditioned', 'Rigid Docking + Pharmacophore'].includes(mode) || mode.toLowerCase().includes('pharmacophore');

          // Parallel fetch for speed
          const structuralPromises = [];

          if (needsProtein) {
            const protFile = inputs.files.find(f => f.extension === '.pdb' || f.extension === '.cif');
            if (protFile) {
              structuralPromises.push(apiClient.downloadInputFile(jobId, protFile.filename).then(async b => ({
                type: 'protein' as const,
                text: await b.text(),
                format: protFile.extension === '.pdb' ? 'pdb' : 'cif'
              })));
            }
          }

          if (needsPharmacophore) {
            const pharmFile = inputs.files.find(f => ['.xyz', '.json'].includes(f.extension.toLowerCase())) ||
              inputs.files.find(f => f.extension.toLowerCase() === '.sdf');
            if (pharmFile) {
              structuralPromises.push(apiClient.downloadInputFile(jobId, pharmFile.filename).then(async b => ({
                type: 'pharm' as const,
                text: await b.text(),
                extension: pharmFile.extension.toLowerCase()
              })));
            }
          }

          const results = await Promise.all(structuralPromises);
          results.forEach(res => {
            if (res.type === 'protein') setProteinData({ text: res.text, format: res.format });
            if (res.type === 'pharm') setPharmData({ text: res.text, extension: res.extension });
          });
          setHasLoadedStructural(true);
        } catch (err) {
          console.error('Failed to prefetch structural data:', err);
        }
      }

      // 2. Prefetch SDF contents for 3D Viewer
      for (const file of sdfFiles) {
        if (!prefetchedMolecules[file.filename]) {
          try {
            const blob = await apiClient.downloadFile(jobId, file.filename);
            const text = await blob.text();
            setPrefetchedMolecules(prev => ({ ...prev, [file.filename]: text }));
          } catch (err) {
            console.error(`Failed to prefetch ${file.filename}:`, err);
          }
        }
      }

      // 3. Prefetch 2D diagrams
      for (const file of sdfFiles) {
        prefetchInteractionDiagram(jobId, file.filename).catch(() => { });
      }
    };

    if (status?.state === 'SUCCEEDED') {
      prefetchEverything();
    }
  }, [result, sdfFiles, jobId, status?.state, inputFiles, hasLoadedStructural]);

  const isLoading = statusLoading || (status?.state === 'SUCCEEDED' && resultLoading);

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
    <div className="space-y-6" style={{ width: '100%', minWidth: 0 }}>
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="font-medium">Back to Jobs</span>
      </button>

      <div className="border-b border-slate-200/60 pb-4" style={{ width: '100%', minWidth: 0, overflow: 'visible' }}>
        <div className="mb-2 flex items-baseline gap-2 flex-wrap">
          <h2 className="text-2xl font-semibold text-slate-900">
            Job Details:
          </h2>
          <span
            className="text-2xl font-semibold text-slate-900"
            style={{
              wordBreak: 'break-all',
              overflowWrap: 'anywhere',
              whiteSpace: 'normal'
            }}
          >
            {jobId}
          </span>
        </div>
        <div className="text-sm text-slate-600">
          {(result.params as any).docking_mode || result.params.sampling_mode || 'Unconditional'} • {(result.params as any).n_samples || (result.params as any).num_samples || 'N/A'} samples •{' '}
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
            setInputValue(String(newIndex));
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
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
          }}
          onBlur={(e) => {
            const val = parseInt(e.target.value, 10);
            if (isNaN(val) || val < 0) {
              setMoleculeIndex(0);
              setInputValue('0');
            } else if (val >= sdfFiles.length) {
              setMoleculeIndex(sdfFiles.length - 1);
              setInputValue(String(sdfFiles.length - 1));
            } else {
              setMoleculeIndex(val);
              setInputValue(String(val));
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.currentTarget.blur();
            }
          }}
          className="w-24 px-3 py-2.5 border border-slate-200 rounded-xl text-center bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
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
        result.params.sampling_mode === 'Protein+Pharmacophore-conditioned' ||
        (result.params as any).docking_mode ? (
        <div className="rounded-2xl bg-white shadow-sm">
          {/* Tabs */}
          <div className="flex border-b border-slate-200/60">
            <button
              onClick={() => setActiveTab('3d')}
              className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${activeTab === '3d'
                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50/50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
            >
              3D Viewer
            </button>
            <button
              onClick={() => setActiveTab('2d')}
              className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${activeTab === '2d'
                ? 'text-primary-600 border-b-2 border-primary-600 bg-primary-50/50'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
            >
              2D Interaction Diagram
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-4">
            <div className={activeTab === '3d' ? 'block' : 'hidden'}>
              <MolecularViewer
                jobId={jobId}
                filename={currentFile.filename}
                samplingMode={(result.params as any).docking_mode || result.params.sampling_mode}
                inputFilesList={inputFiles}
                prefetchedContent={prefetchedMolecules[currentFile.filename]}
              />
            </div>
            <div className={activeTab === '2d' ? 'block' : 'hidden'}>
              <InteractionDiagram2D
                jobId={jobId}
                filename={currentFile.filename}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl p-4 bg-white shadow-sm">
          <MolecularViewer
            jobId={jobId}
            filename={currentFile.filename}
            samplingMode={(result.params as any).docking_mode || result.params.sampling_mode || 'Unconditional'}
            pocketSelection={result.params.pocket_selection}
            inputFilesList={inputFiles}
            prefetchedContent={prefetchedMolecules[currentFile.filename]}
          />
        </div>
      )}

      <MetricsTable
        jobId={jobId}
        samplingMode={(result.params as any).docking_mode || result.params.sampling_mode || 'Unconditional'}
        onRowSelect={(index) => {
          setMoleculeIndex(index);
          setInputValue(String(index));
        }}
        selectedIndex={moleculeIndex}
      />
    </div>
  );
}


