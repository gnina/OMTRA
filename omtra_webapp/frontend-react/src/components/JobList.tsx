'use client';

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import type { JobMetadata, JobStatus } from '@/types';
import { Loader2, CheckCircle2, XCircle, Clock, PlayCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface JobListProps {
  onJobSelect: (jobId: string) => void;
}

const statusIcons: Record<JobStatus, React.ReactNode> = {
  QUEUED: <Clock className="w-4 h-4" />,
  RUNNING: <PlayCircle className="w-4 h-4 animate-spin" />,
  SUCCEEDED: <CheckCircle2 className="w-4 h-4" />,
  FAILED: <XCircle className="w-4 h-4" />,
  CANCELED: <XCircle className="w-4 h-4" />,
};

const statusColors: Record<JobStatus, string> = {
  QUEUED: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  RUNNING: 'bg-blue-100 text-blue-800 border-blue-200',
  SUCCEEDED: 'bg-green-100 text-green-800 border-green-200',
  FAILED: 'bg-red-100 text-red-800 border-red-200',
  CANCELED: 'bg-gray-100 text-gray-800 border-gray-200',
};

export function JobList({ onJobSelect }: JobListProps) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => apiClient.listJobs(),
    retry: 1,
    retryDelay: 1000,
    staleTime: 5000,
    refetchInterval: (query) => {
      // Poll every 2 seconds if there are running jobs
      const jobs = query.state.data?.jobs || [];
      const hasRunning = jobs.some((j: JobMetadata) => j.state === 'QUEUED' || j.state === 'RUNNING');
      return hasRunning ? 2000 : false;
    },
  });

  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const handleJobClick = (jobId: string) => {
    setSelectedJobId(jobId);
    onJobSelect(jobId);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
        <span className="ml-2 text-gray-600">Loading jobs...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <p className="text-red-600 mb-2">Failed to load jobs</p>
        <p className="text-sm text-gray-500 mb-4">
          {error instanceof Error ? error.message : 'Unknown error'}
        </p>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const jobs = data?.jobs || [];

  if (jobs.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>No jobs submitted yet. Use the sidebar to start sampling!</p>
      </div>
    );
  }

  // Sort by creation time (newest first)
  const sortedJobs = [...jobs].sort((a, b) => {
    const timeA = new Date(a.created_at).getTime();
    const timeB = new Date(b.created_at).getTime();
    return timeB - timeA;
  });

  const runningCount = sortedJobs.filter((j) => j.state === 'QUEUED' || j.state === 'RUNNING').length;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">Job Queue</h2>
          <p className="mt-1 text-sm text-slate-500">View and manage your jobs</p>
        </div>
        {runningCount > 0 && (
          <div className="text-sm text-slate-600 bg-blue-50/70 rounded-xl px-4 py-2 shadow-sm">
            💡 {runningCount} job(s) running. Checking for updates...
          </div>
        )}
      </div>

      <div className="space-y-3">
        {sortedJobs.map((job) => {
          const status = job.state || 'QUEUED';
          const elapsed = job.elapsed_seconds
            ? formatElapsedTime(job.elapsed_seconds)
            : null;
          const nSamples = job.params?.n_samples || 'N/A';

          return (
            <div
              key={job.job_id}
              onClick={() => handleJobClick(job.job_id)}
              className={`p-5 rounded-2xl cursor-pointer transition-all ${
                selectedJobId === job.job_id
                  ? 'bg-primary-50/70 ring-2 ring-primary-200 shadow-md'
                  : 'bg-white shadow-sm hover:shadow-md'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-mono text-sm font-semibold text-slate-900 break-all">
                      {job.job_id}
                    </span>
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border ${statusColors[status]}`}
                    >
                      {statusIcons[status]}
                      {status}
                    </span>
                  </div>
                  <div className="text-sm text-slate-600">
                    {nSamples} samples • {job.params?.sampling_mode}
                    {elapsed && ` • ${elapsed}`}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {(() => {
                      // Parse timestamp - API returns ISO format without timezone
                      // Treat it as UTC if no timezone is specified
                      let createdDate: Date;
                      const timestamp = job.created_at;
                      if (timestamp.includes('Z') || timestamp.includes('+') || timestamp.includes('-', 10)) {
                        // Has timezone info
                        createdDate = new Date(timestamp);
                      } else {
                        // No timezone - assume UTC and convert to local
                        createdDate = new Date(timestamp + 'Z');
                      }
                      
                      const now = new Date();
                      const diffMs = now.getTime() - createdDate.getTime();
                      
                      // If date appears to be in the future (more than 1 hour ahead), 
                      // it's likely a timezone parsing issue - show absolute time
                      if (diffMs < -3600000) {
                        return createdDate.toLocaleString();
                      }
                      
                      // Otherwise show relative time
                      return formatDistanceToNow(createdDate, { addSuffix: true });
                    })()}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatElapsedTime(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }
}

