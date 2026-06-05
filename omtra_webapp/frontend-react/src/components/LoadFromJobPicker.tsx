'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { Loader2, RefreshCw } from 'lucide-react';
import type { JobMetadata } from '@/types';
import { JobStatus } from '@/types';

export type JobFileSource = 'inputs' | 'outputs';

const EXCLUDED_FILENAMES = new Set([
  'fixed_structure_reference.sdf',
  'reference_ligand.sdf',
  'per_molecule_metrics.json',
  'summary.json',
  'protein_from_cif.pdb',
]);

interface JobFileOption {
  filename: string;
  source: JobFileSource;
  label: string;
}

interface LoadFromJobPickerProps {
  acceptedExtensions: string[];
  onFileLoaded: (file: File) => void;
  disabled?: boolean;
}

export function LoadFromJobPicker({
  acceptedExtensions,
  onFileLoaded,
  disabled = false,
}: LoadFromJobPickerProps) {
  const [jobs, setJobs] = useState<JobMetadata[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [files, setFiles] = useState<JobFileOption[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [selectedFile, setSelectedFile] = useState('');
  const [loadingFile, setLoadingFile] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const extSet = new Set(acceptedExtensions.map((e) => e.toLowerCase()));

  const fetchJobs = useCallback(async () => {
    setLoadingJobs(true);
    try {
      const res = await apiClient.listJobs();
      const succeeded = (res.jobs || []).filter((j) => j.state === JobStatus.SUCCEEDED);
      setJobs(succeeded);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load jobs');
    } finally {
      setLoadingJobs(false);
    }
  }, []);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const loadFilesForJob = useCallback(
    async (jobId: string) => {
      setLoadingFiles(true);
      setFiles([]);
      setSelectedFile('');
      setError(null);
      try {
        const inputFiles = await apiClient.listInputFiles(jobId);
        const options: JobFileOption[] = inputFiles
          .filter((f) => extSet.has(f.extension) && !EXCLUDED_FILENAMES.has(f.filename))
          .map((f) => ({
            filename: f.filename,
            source: 'inputs' as const,
            label: `[input] ${f.filename}`,
          }));

        try {
          const result = await apiClient.getJobResult(jobId);
          const outputSdgs = (result.artifacts || [])
            .filter((a) => {
              const ext = a.filename.includes('.')
                ? a.filename.slice(a.filename.lastIndexOf('.')).toLowerCase()
                : '';
              
              if (EXCLUDED_FILENAMES.has(a.filename) || a.filename.endsWith('_diagram_error.json')) return false;

              return extSet.has(ext);
            })
            .map((a) => ({
              filename: a.filename,
              source: 'outputs' as const,
              label: `[output] ${a.filename}`,
            }));
          options.push(...outputSdgs);
        } catch {
          /* outputs optional */
        }

        setFiles(options);
        if (options.length === 0) {
          setError('No matching files in this job');
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to list job files');
      } finally {
        setLoadingFiles(false);
      }
    },
    [acceptedExtensions],
  );

  const handleJobChange = (jobId: string) => {
    setSelectedJobId(jobId);
    if (jobId) loadFilesForJob(jobId);
    else {
      setFiles([]);
      setSelectedFile('');
    }
  };

  const handleLoad = async () => {
    if (!selectedJobId || !selectedFile) return;
    const opt = files.find((f) => `${f.source}:${f.filename}` === selectedFile);
    if (!opt) return;

    setLoadingFile(true);
    setError(null);
    try {
      const blob =
        opt.source === 'inputs'
          ? await apiClient.downloadInputFile(selectedJobId, opt.filename)
          : await apiClient.downloadFile(selectedJobId, opt.filename);
      const file = new File([blob], opt.filename, { type: blob.type || 'application/octet-stream' });
      onFileLoaded(file);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load file');
    } finally {
      setLoadingFile(false);
    }
  };

  return (
    <div className="mt-4 space-y-3">
      <div className="flex items-center gap-3">
        <div className="flex-1 h-px bg-slate-200"></div>
        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">or load from past job</p>
        <div className="flex-1 h-px bg-slate-200"></div>
      </div>
      {loadingJobs ? (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="w-3 h-3 animate-spin" /> Loading jobs...
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <select
            value={selectedJobId}
            onChange={(e) => handleJobChange(e.target.value)}
            disabled={disabled || jobs.length === 0}
            className="flex-1 text-xs px-2 py-2 border border-slate-200 rounded-lg bg-white truncate"
          >
            <option value="">Select a job...</option>
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                {j.job_id}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={fetchJobs}
            disabled={disabled || loadingJobs}
            className="flex-shrink-0 p-2 text-slate-400 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors border border-transparent hover:border-primary-200"
            title="Refresh job list"
          >
            <RefreshCw className={`w-4 h-4 ${loadingJobs ? 'animate-spin' : ''}`} />
          </button>
        </div>
      )}
      {selectedJobId && (
        <>
          {loadingFiles ? (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Loader2 className="w-3 h-3 animate-spin" /> Loading files...
            </div>
          ) : (
            <select
              value={selectedFile}
              onChange={(e) => setSelectedFile(e.target.value)}
              disabled={disabled || files.length === 0}
              className="w-full text-xs px-2 py-2 border border-slate-200 rounded-lg bg-white"
            >
              <option value="">Select a file...</option>
              {files.map((f) => (
                <option key={`${f.source}:${f.filename}`} value={`${f.source}:${f.filename}`}>
                  {f.label}
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            onClick={handleLoad}
            disabled={disabled || !selectedFile || loadingFile}
            className="w-full text-xs font-semibold py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50"
          >
            {loadingFile ? 'Loading...' : 'Use this file'}
          </button>
        </>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
