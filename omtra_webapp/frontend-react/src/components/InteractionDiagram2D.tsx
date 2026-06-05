'use client';

import { useState, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import { apiClient, fastApiErrorDetail } from '@/lib/api-client';
import { Loader2, AlertCircle, Download } from 'lucide-react';

const PROTEINS_PLUS_POSEVIEW_URL = 'https://proteins.plus/';
const POSEVIEW_ERROR_MESSAGE = 'PoseView failed to generate diagram';
const DIAGRAM_POLL_MS = 2000;
const DIAGRAM_POLL_MAX_ATTEMPTS = 45;

interface InteractionDiagram2DProps {
  jobId: string;
  filename: string;
  sampleLabel?: string;
}

export interface DiagramError {
  message: string;
  reason?: string;
  statusCode?: number;
}

export const errorCache = new Map<string, DiagramError>();
export const svgCache = new Map<string, string>();

function isBlankSvg(svg: string | null | undefined): boolean {
  if (!svg) return true;
  const trimmed = svg.trim().replace(/\s+/g, ' ');
  if (!trimmed) return true;
  if (trimmed === '<svg></svg>' || trimmed === '<svg/>') return true;

  const borderPatterns = [
    /<path[^>]*d="[^"]*M\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+Z/i,
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+600\s+0\s+L\s+600\s+600\s+L\s+0\s+600\s+Z/i,
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+\d+\s+0\s+L\s+\d+\s+\d+\s+L\s+0\s+\d+\s+Z/i,
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+\d+\s+0\s+L\s+\d+\s+\d+\s+L\s+0\s+\d+\s+Z\s+M\s+0\s+0/i,
  ];

  const pathCount = trimmed.match(/<path[^>]*>/gi)?.length ?? 0;
  if (borderPatterns.some((pattern) => pattern.test(trimmed)) && (trimmed.length < 500 || pathCount === 1)) {
    return true;
  }

  const hasMeaningfulContent =
    /<text[^>]*>/i.test(trimmed) ||
    /<circle[^>]*r="[^"]*"[^>]*>/i.test(trimmed) ||
    (/<path[^>]*d="[^"]*[ML][^"]*[ML]"/i.test(trimmed) && trimmed.length > 500) ||
    /<path[^>]*d="[^"]*[CcQqSsTtAaZz]/.test(trimmed);

  return !hasMeaningfulContent && trimmed.length < 500;
}

export const extractInteractionDiagramErrorDetails = (err: unknown): DiagramError => {
  const base = { message: POSEVIEW_ERROR_MESSAGE };
  if (axios.isAxiosError(err)) {
    const detail = fastApiErrorDetail(err.response?.data);
    if (detail) return { ...base, reason: detail };
  }

  if (err instanceof Error) {
    const m = err.message.trim();
    if (!m || m === base.message) return base;
    if (/^Request failed with status code \d+$/.test(m)) {
      return {
        ...base,
        reason:
          'The server returned an error without a JSON description. Often this is a ProteinsPlus or network timeout.',
      };
    }
    return { ...base, reason: m };
  }
  return base;
};

function cacheDiagramSuccess(cacheKey: string, svg: string): DiagramError | null {
  if (isBlankSvg(svg)) {
    const error: DiagramError = {
      message: POSEVIEW_ERROR_MESSAGE,
      reason: 'Generated diagram is empty or blank',
    };
    cacheDiagramError(cacheKey, error);
    return error;
  }
  svgCache.set(cacheKey, svg);
  errorCache.delete(cacheKey);
  return null;
}

function cacheDiagramError(cacheKey: string, error: DiagramError): void {
  errorCache.set(cacheKey, error);
  svgCache.delete(cacheKey);
}

async function pollInteractionDiagram(
  jobId: string,
  filename: string,
  cacheKey: string,
  onPending?: (attempt: number) => void,
): Promise<{ svg?: string; error?: DiagramError }> {
  for (let attempt = 0; attempt < DIAGRAM_POLL_MAX_ATTEMPTS; attempt++) {
    if (attempt > 0) {
      onPending?.(attempt);
      await new Promise((r) => setTimeout(r, DIAGRAM_POLL_MS));
    }
    try {
      const result = await apiClient.getInteractionDiagram(jobId, filename);
      if (result.status === 'ready') {
        const errorDetails = cacheDiagramSuccess(cacheKey, result.svg);
        if (errorDetails) return { error: errorDetails };
        return { svg: result.svg };
      }
      if (result.status === 'failed') {
        const error: DiagramError = {
          message: POSEVIEW_ERROR_MESSAGE,
          reason: result.detail,
        };
        cacheDiagramError(cacheKey, error);
        return { error };
      }
    } catch (err) {
      const errorDetails = extractInteractionDiagramErrorDetails(err);
      cacheDiagramError(cacheKey, errorDetails);
      return { error: errorDetails };
    }
  }
  const timeoutError: DiagramError = {
    message: POSEVIEW_ERROR_MESSAGE,
    reason: 'Diagram generation timed out. Try again or check worker logs.',
  };
  cacheDiagramError(cacheKey, timeoutError);
  return { error: timeoutError };
}

export async function prefetchInteractionDiagram(jobId: string, filename: string): Promise<void> {
  const cacheKey = `${jobId}/${filename}`;
  if (svgCache.has(cacheKey) || errorCache.has(cacheKey)) return;
  await pollInteractionDiagram(jobId, filename, cacheKey);
}

function ProteinsPlusHint({ compact = false }: { compact?: boolean }) {
  return (
    <p
      className={
        compact
          ? 'text-xs text-center text-red-800/90 max-w-xs'
          : 'text-xs text-slate-500 text-center max-w-md'
      }
    >
      <a
        href={PROTEINS_PLUS_POSEVIEW_URL}
        target="_blank"
        rel="noopener noreferrer"
        className={
          compact
            ? 'text-primary-700 hover:text-primary-800 underline font-medium'
            : 'text-primary-600 hover:text-primary-700 underline font-medium'
        }
      >
        ProteinsPlus
      </a>{' '}
      {compact
        ? 'may work in your browser when the API cannot.'
        : 'may work in your browser when the API server cannot reach it.'}
    </p>
  );
}

function ErrorDetails({ error, compact = false }: { error: DiagramError; compact?: boolean }) {
  return (
    <>
      <AlertCircle className={compact ? 'w-6 h-6 text-red-500' : 'w-8 h-8 text-red-500'} />
      <p
        className={
          compact
            ? 'text-sm text-red-700 font-medium text-center'
            : 'text-slate-900 font-semibold text-center'
        }
      >
        {error.message}
      </p>
      {error.reason ? (
        <p
          className={
            compact
              ? 'text-xs text-red-600/90 text-center max-w-sm whitespace-pre-wrap break-words font-mono bg-white/80 rounded p-2 border border-red-100'
              : 'text-xs text-slate-600 text-center max-w-lg whitespace-pre-wrap break-words font-mono bg-slate-50 border border-slate-200 rounded-lg p-3'
          }
        >
          {error.reason}
        </p>
      ) : null}
      <ProteinsPlusHint compact={compact} />
    </>
  );
}

export function InteractionDiagram2D({ jobId, filename, sampleLabel }: InteractionDiagram2DProps) {
  const cacheKey = useMemo(() => `${jobId}/${filename}`, [jobId, filename]);
  const [svgContent, setSvgContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<DiagramError | null>(null);
  const [pollAttempt, setPollAttempt] = useState(0);

  const handlePending = useCallback((attempt: number) => {
    setPollAttempt(attempt);
  }, []);

  const handleDownload = () => {
    if (!svgContent) return;
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_diagram.svg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const cachedError = errorCache.get(cacheKey) || null;
    const cachedSvg = svgCache.get(cacheKey) || null;

    if (cachedSvg && !isBlankSvg(cachedSvg)) {
      setSvgContent(cachedSvg);
      setError(null);
      setIsLoading(false);
      return;
    }

    if (cachedError) {
      setError(cachedError);
      setSvgContent(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setSvgContent(null);
    setError(null);
    setIsLoading(true);
    setPollAttempt(0);

    const loadDiagram = async () => {
      const result = await pollInteractionDiagram(jobId, filename, cacheKey, handlePending);
      if (cancelled) return;
      if (result.svg) {
        setSvgContent(result.svg);
        setError(null);
      } else if (result.error) {
        setError(result.error);
        setSvgContent(null);
      }
      setIsLoading(false);
    };

    void loadDiagram();
    return () => {
      cancelled = true;
    };
  }, [jobId, filename, cacheKey, handlePending]);

  return (
    <div className="relative">
      <div
        className="relative w-full border border-slate-200 rounded-2xl overflow-hidden bg-white shadow-sm"
        style={{ height: '500px' }}
      >
          <div className="absolute top-4 left-4 z-20 flex flex-col gap-2 pointer-events-auto items-start">
            {sampleLabel && (
              <div className="px-3 py-2 bg-white/90 text-slate-800 rounded-lg shadow-sm border border-slate-200 text-sm font-semibold backdrop-blur-sm">
                {sampleLabel}
              </div>
            )}
            <button
              onClick={handleDownload}
              disabled={!svgContent || !!error || isLoading}
              className="flex items-center gap-2 px-4 py-2 bg-white/90 text-slate-700 rounded-lg hover:bg-white transition-colors shadow-sm border border-slate-200 text-sm font-medium backdrop-blur-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
          </div>
          {isLoading && !error && (
            <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
              <div className="flex flex-col items-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary-600 mb-2" />
                <p className="text-sm text-slate-600">Generating interaction diagram…</p>
                <p className="text-xs text-slate-400 mt-1">
                  {pollAttempt > 0
                    ? 'Waiting for PoseView…'
                    : 'Results are ready.'}
                </p>
              </div>
            </div>
          )}

          {error && !svgContent && !isLoading && (
            <div className="flex flex-col items-center justify-center h-96 p-6 bg-white gap-4">
              <ErrorDetails error={error} />
            </div>
          )}

          {error && svgContent && (
            <div className="absolute inset-0 bg-red-50/90 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg border-2 border-red-200">
              <div className="flex flex-col items-center p-4 max-w-md gap-3">
                <ErrorDetails error={error} compact />
              </div>
            </div>
          )}

          {svgContent && !error ? (
            <div
              key={`svg-${filename}`}
              className="w-full h-full p-4 bg-white flex items-center justify-center [&>svg]:w-full [&>svg]:h-full [&>svg]:max-h-full [&>svg]:max-w-full"
              dangerouslySetInnerHTML={{ __html: svgContent }}
            />
          ) : null}

          {!error && !svgContent && !isLoading && (
            <div className="flex items-center justify-center h-96 bg-white">
              <div className="text-center">
                <p className="text-slate-600 mb-2">2D Interaction Diagram</p>
                <p className="text-sm text-slate-500">No diagram available</p>
              </div>
            </div>
          )}
      </div>
    </div>
  );
}
