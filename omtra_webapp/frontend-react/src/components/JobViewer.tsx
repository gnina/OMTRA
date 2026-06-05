'use client';

import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { JobStatus } from '@/types';
import { Loader2, Download, ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react';
import { MolecularViewer } from './MolecularViewer';
import { MetricsTable } from './MetricsTable';
import { InteractionDiagram2D } from './InteractionDiagram2D';

interface JobViewerProps {
  jobId: string;
  onBack: () => void;
}

export function JobViewer({ jobId, onBack }: JobViewerProps) {
  const [moleculeIndex, setMoleculeIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'3d' | '2d'>('3d');
  const [prefetchedMolecules, setPrefetchedMolecules] = useState<Record<string, string>>({});

  // Structural data state
  const [proteinData, setProteinData] = useState<{ text: string, format: string } | null>(null);
  const [pharmData, setPharmData] = useState<{ text: string, extension: string } | null>(null);
  const [hasLoadedStructural, setHasLoadedStructural] = useState(false);

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
    const refs = result.artifacts.filter((a: any) => a.filename === 'reference_ligand.sdf');
    const samples = result.artifacts
      .filter((a: any) => a.filename.startsWith('sample_') && a.filename.endsWith('.sdf'))
      .sort((a: any, b: any) => {
        const numA = parseInt(a.filename.match(/\d+/)?.[0] || '0');
        const numB = parseInt(b.filename.match(/\d+/)?.[0] || '0');
        return numA - numB;
      });
    return [...refs, ...samples];
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

          const files = Array.isArray(inputs) ? inputs : (inputs as any).files ?? [];

          const structuralPromises = [];

          if (needsProtein) {
            const protFile = files.find((f: any) => f.extension === '.pdb' || f.extension === '.cif');
            if (protFile) {
              structuralPromises.push(apiClient.downloadInputFile(jobId, protFile.filename).then(async b => ({
                type: 'protein' as const,
                text: await b.text(),
                format: protFile.extension === '.pdb' ? 'pdb' : 'cif'
              })));
            }
          }

          if (needsPharmacophore) {
            const pharmFile = files.find((f: any) => ['.xyz', '.json'].includes(f.extension.toLowerCase())) ||
              files.find((f: any) => f.extension.toLowerCase() === '.sdf');
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
      const sdfNames = new Set(sdfFiles.map((f) => f.filename));
      sdfNames.add('fixed_structure_reference.sdf');
      for (const name of sdfNames) {
        if (!prefetchedMolecules[name]) {
          try {
            const blob = await apiClient.downloadFile(jobId, name);
            const text = await blob.text();
            setPrefetchedMolecules((prev) => ({ ...prev, [name]: text }));
          } catch (err) {
            if (name !== 'fixed_structure_reference.sdf') {
              console.error(`Failed to prefetch ${name}:`, err);
            }
          }
        }
      }

      // 2D diagrams load on demand when the user opens the 2D tab (avoids hammering PoseView).
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

  const showDiagramTab = (result.params as any).metrics_options?.poseview !== false;
  const effectiveTab = showDiagramTab ? activeTab : '3d';

  const sampleLabel = (() => {
    const hasRef = sdfFiles[0]?.filename === "reference_ligand.sdf";
    if (sdfFiles[moleculeIndex]?.filename === "reference_ligand.sdf") {
      return "Reference Ligand";
    }
    const totalSamples = sdfFiles.length - (hasRef ? 1 : 0);
    const sampleNumber = hasRef ? moleculeIndex : moleculeIndex + 1;
    return `Sample ${sampleNumber} of ${totalSamples}`;
  })();

  return (
    <div className="space-y-6" style={{ width: '100%', minWidth: 0 }}>
      <button
        onClick={onBack}
        className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span className="font-medium">Back to Jobs</span>
      </button>

      <div className="border-b border-slate-200/60 pb-4 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div style={{ width: "100%", minWidth: 0, overflow: "visible" }}>
          <div className="mb-2 flex items-baseline gap-2 flex-wrap">
            <h2 className="text-2xl font-semibold text-slate-900">
              Job Details:
            </h2>
            <span
              className="text-2xl font-semibold text-slate-900"
              style={{
                wordBreak: "break-all",
                overflowWrap: "anywhere",
                whiteSpace: "normal"
              }}
            >
              {jobId}
            </span>
          </div>
          <div className="text-sm text-slate-600 flex items-center gap-2">
            <span>
              {!!(result.params as any).fixed_atom_indices?.length ? "Partial " : ""}
              {(result.params as any).docking_mode || result.params.sampling_mode || "Unconditional"} • {(result.params as any).n_samples || (result.params as any).num_samples || "N/A"} samples • {result.params.steps} steps
            </span>
          </div>
        </div>

        <div className="relative group flex-shrink-0 h-fit">
          <button
            onClick={handleDownloadAll}
            className="flex items-center gap-2 px-5 py-2.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 transition-colors shadow-sm hover:shadow-md w-full"
          >
            <Download className="w-4 h-4" />
            Download All
          </button>
          <div className="absolute top-full right-0 mt-2 w-56 p-2 bg-slate-800 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 text-center pointer-events-none">
            Download a ZIP containing all generated structures and metrics
          </div>
        </div>
      </div>

      {/* Viewer Area */}
      <div className="w-full relative group mb-8">
        {/* Overlay Navigation Arrows */}
        <button
          onClick={() => {
            const newIndex = Math.max(0, moleculeIndex - 1);
            setMoleculeIndex(newIndex);
          }}
          disabled={moleculeIndex === 0}
          className="absolute left-6 top-1/2 -translate-y-1/2 z-40 p-3 bg-white/90 hover:bg-white text-slate-800 rounded-full shadow-lg backdrop-blur-sm transition-all disabled:hidden border border-slate-200"
        >
          <ChevronLeft className="w-8 h-8" />
        </button>

        <button
          onClick={() => {
            const newIndex = Math.min(sdfFiles.length - 1, moleculeIndex + 1);
            setMoleculeIndex(newIndex);
          }}
          disabled={moleculeIndex >= sdfFiles.length - 1}
          className="absolute right-6 top-1/2 -translate-y-1/2 z-40 p-3 bg-white/90 hover:bg-white text-slate-800 rounded-full shadow-lg backdrop-blur-sm transition-all disabled:hidden border border-slate-200"
        >
          <ChevronRight className="w-8 h-8" />
        </button>

        {result.params.sampling_mode === "Protein-conditioned" ||
          result.params.sampling_mode === "Protein+Pharmacophore-conditioned" ||
          (result.params as any).docking_mode ? (
          <div className="rounded-2xl bg-white shadow-sm border border-slate-200">
            {/* Tabs */}
            <div className="flex border-b border-slate-200/60">
              <button
                onClick={() => setActiveTab("3d")}
                className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${effectiveTab === "3d"
                  ? "text-primary-600 border-b-2 border-primary-600 bg-primary-50/50"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
              >
                3D Viewer
              </button>
              {showDiagramTab && (
              <button
                onClick={() => setActiveTab("2d")}
                className={`flex-1 px-6 py-3 text-sm font-medium transition-colors ${effectiveTab === "2d"
                  ? "text-primary-600 border-b-2 border-primary-600 bg-primary-50/50"
                  : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
              >
                2D Interaction Diagram
              </button>
              )}
            </div>

            {/* Tab Content */}
            <div className="p-4">
              <div className={effectiveTab === "3d" ? "block" : "hidden"}>
                <MolecularViewer
                  jobId={jobId}
                  filename={currentFile.filename}
                  samplingMode={(result.params as any).docking_mode || result.params.sampling_mode}
                  pocketSelection={(result.params as any).pocket_selection}
                  inputFilesList={inputFiles}
                  prefetchedContent={prefetchedMolecules[currentFile.filename]}
                  referenceLigandContent={
                    prefetchedMolecules["fixed_structure_reference.sdf"]
                    ?? prefetchedMolecules["reference_ligand.sdf"]
                  }
                  fixedAtomIndices={(result.params as any).fixed_atom_indices}
                  fixedBricsFragments={(result.params as any).fixed_brics_fragments}
                  sampleLabel={sampleLabel}
                />
              </div>
              {showDiagramTab && (
              <div className={effectiveTab === "2d" ? "block" : "hidden"}>
                <InteractionDiagram2D
                  jobId={jobId}
                  filename={currentFile.filename}
                  sampleLabel={sampleLabel}
                />
              </div>
              )}
            </div>
          </div>
        ) : (
          <div className="rounded-2xl p-4 bg-white shadow-sm border border-slate-200">
            <MolecularViewer
              jobId={jobId}
              filename={currentFile.filename}
              samplingMode={(result.params as any).docking_mode || result.params.sampling_mode || "Unconditional"}
              pocketSelection={result.params.pocket_selection}
              inputFilesList={inputFiles}
              prefetchedContent={prefetchedMolecules[currentFile.filename]}
              referenceLigandContent={
                prefetchedMolecules["fixed_structure_reference.sdf"]
                ?? prefetchedMolecules["reference_ligand.sdf"]
              }
              fixedAtomIndices={(result.params as any).fixed_atom_indices}
              fixedBricsFragments={(result.params as any).fixed_brics_fragments}
                  sampleLabel={sampleLabel}
            />
          </div>
        )}
      </div>
      <MetricsTable
        jobId={jobId}
        samplingMode={(result.params as any).docking_mode || result.params.sampling_mode || 'Unconditional'}
        metricsOptions={result.params.metrics_options}
        onRowSelect={(index) => {
          setMoleculeIndex(index);
        }}
        selectedIndex={moleculeIndex}
      />
    </div>
  );
}


