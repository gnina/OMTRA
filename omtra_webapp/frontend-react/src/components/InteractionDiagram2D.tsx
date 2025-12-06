'use client';

import { useState, useEffect, useLayoutEffect, useRef, useMemo } from 'react';
import { apiClient } from '@/lib/api-client';
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react';

interface InteractionDiagram2DProps {
  jobId: string;
  filename: string;
}

export interface DiagramError {
  message: string;
  reason?: string;
  statusCode?: number;
}

// Cache errors per filename to avoid retrying failed diagrams
export const errorCache = new Map<string, DiagramError>();
export const svgCache = new Map<string, string>();

// Helper function to extract error details from API error
export const extractInteractionDiagramErrorDetails = (err: any): DiagramError => {
  const message = 'PoseView failed to generate diagram';
  let reason: string | undefined;
  let statusCode: number | undefined;

  if (err?.response) {
    statusCode = err.response.status;
    const detail = err.response.data?.detail || err.response.data?.message || err.response.data;

    if (typeof detail === 'string') {
      reason = detail;
    } else if (detail) {
      reason = typeof detail === 'object' ? JSON.stringify(detail) : String(detail);
    }
  } else if (err?.message) {
    reason = err.message;
  }

  return { message, reason, statusCode };
};

// Prefetch diagram for given job+filename combination so that the UI can show it instantly later.
export async function prefetchInteractionDiagram(jobId: string, filename: string): Promise<void> {
  const cacheKey = `${jobId}/${filename}`;

  if (svgCache.has(cacheKey) || errorCache.has(cacheKey)) {
    return;
  }

  try {
    const svg = await apiClient.getInteractionDiagram(jobId, filename);
    svgCache.set(cacheKey, svg);
    errorCache.delete(cacheKey);
  } catch (err) {
    const errorDetails = extractInteractionDiagramErrorDetails(err);
    errorCache.set(cacheKey, errorDetails);
    svgCache.delete(cacheKey);
  }
}

export function InteractionDiagram2D({ jobId, filename }: InteractionDiagram2DProps) {
  // Check cache immediately during initialization - compute synchronously
  const cacheKey = useMemo(() => `${jobId}/${filename}`, [jobId, filename]);
  
  // Get cached values synchronously during initialization and normalize error message
  const initialCachedErrorRaw = errorCache.get(cacheKey) || null;
  const initialCachedError = initialCachedErrorRaw ? {
    message: 'PoseView failed to generate diagram', // Always use simple message
    reason: initialCachedErrorRaw.reason || initialCachedErrorRaw.message, // Use reason if available, otherwise old message
    statusCode: initialCachedErrorRaw.statusCode
  } : null;
  const initialCachedSvg = svgCache.get(cacheKey) || null;
  
  const [svgContent, setSvgContent] = useState<string | null>(initialCachedSvg);
  const [isLoading, setIsLoading] = useState(!initialCachedError && !initialCachedSvg); // Only loading if no cache
  const [error, setError] = useState<DiagramError | null>(initialCachedError);
  const previousFilenameRef = useRef<string>(filename);

  // Use useLayoutEffect to check cache synchronously before paint - runs before browser paints
  useLayoutEffect(() => {
    // Check cache synchronously
    const cachedErrorRaw = errorCache.get(cacheKey) || null;
    const cachedSvg = svgCache.get(cacheKey) || null;
    
    if (cachedErrorRaw) {
      // Normalize cached error to ensure it always has the simple message
      const cachedError: DiagramError = {
        message: 'PoseView failed to generate diagram', // Always use simple message
        reason: cachedErrorRaw.reason || cachedErrorRaw.message, // Use reason if available, otherwise old message
        statusCode: cachedErrorRaw.statusCode
      };
      // Update cache with normalized error
      errorCache.set(cacheKey, cachedError);
      // We've seen an error for this diagram before - show it immediately without retrying
      // Set all state synchronously to prevent any loading flash
      setError(cachedError);
      setSvgContent(null);
      setIsLoading(false);
      return;
    }
    
    if (cachedSvg) {
      // We have cached SVG - show it immediately
      setSvgContent(cachedSvg);
      setError(null);
      setIsLoading(false);
      return;
    }
    
    // No cache - need to load
    setError(null);
    setSvgContent(null);
    setIsLoading(true);
  }, [jobId, filename, cacheKey]);

  useEffect(() => {
    // Always clear previous content when filename changes
    const filenameChanged = previousFilenameRef.current !== filename;
    previousFilenameRef.current = filename;
    
    // Create a cache key for this job+filename combination
    const currentCacheKey = `${jobId}/${filename}`;
    
    // If we have cache, don't load (already handled in useLayoutEffect)
    if (errorCache.has(currentCacheKey) || svgCache.has(currentCacheKey)) {
      return;
    }
    
    // Load diagram content - exactly like MolecularViewer loads file content
    const loadDiagram = async () => {
      try {
        const svg = await apiClient.getInteractionDiagram(jobId, filename);
        // Only set if this is still the current filename
        if (previousFilenameRef.current === filename) {
          setSvgContent(svg);
          setError(null);
          setIsLoading(false);
          // Cache successful SVG
          svgCache.set(currentCacheKey, svg);
          // Clear any error cache for this filename
          errorCache.delete(currentCacheKey);
        }
      } catch (err) {
        console.error('Failed to load diagram:', err);
        if (previousFilenameRef.current === filename) {
          const errorDetails = extractInteractionDiagramErrorDetails(err);
          setError(errorDetails);
          setIsLoading(false);
          // Cache the error so we don't retry automatically
          errorCache.set(currentCacheKey, errorDetails);
          // Clear SVG cache for this filename
          svgCache.delete(currentCacheKey);
        }
      }
    };

    loadDiagram();
  }, [jobId, filename, cacheKey]);

  const handleRefresh = async () => {
    const cacheKey = `${jobId}/${filename}`;
    
    // Clear caches for this filename to force a fresh load
    errorCache.delete(cacheKey);
    svgCache.delete(cacheKey);
    
    setIsLoading(true);
    setError(null);
    
    try {
      const svg = await apiClient.getInteractionDiagram(jobId, filename);
      setSvgContent(svg);
      setError(null);
      setIsLoading(false);
      // Cache successful SVG
      svgCache.set(cacheKey, svg);
    } catch (err) {
      const errorDetails = extractInteractionDiagramErrorDetails(err);
      setError(errorDetails);
      setIsLoading(false);
      // Cache the error
      errorCache.set(cacheKey, errorDetails);
      svgCache.delete(cacheKey);
    }
  };

  // Always render the same structure - keep UI static
  return (
    <div className="flex flex-col items-center justify-center min-h-96 bg-white rounded-lg p-4 overflow-auto shadow-sm">
      <div className="w-full max-w-4xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-slate-900">2D Interaction Diagram</h3>
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
            title="Refresh diagram"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
        <div className="relative w-full border border-slate-200/60 rounded-lg bg-white overflow-auto shadow-sm" style={{ minHeight: '400px' }}>
          {/* Loading overlay - only shows when loading and no error, doesn't replace content */}
          {isLoading && !error && (
            <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
              <div className="flex flex-col items-center">
                <Loader2 className="w-8 h-8 animate-spin text-primary-600 mb-2" />
                <p className="text-sm text-slate-600">Loading diagram...</p>
              </div>
            </div>
          )}
          
          {/* Error state - no previous content - white background like successful diagrams */}
          {error && !svgContent && !isLoading && (
            <div className="flex flex-col items-center justify-center h-96 p-6 bg-white">
              <AlertCircle className="w-8 h-8 text-red-500 mb-4" />
              <p className="text-slate-900 font-semibold mb-4">{error.message}</p>
              <button
                onClick={handleRefresh}
                className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Retry</span>
              </button>
            </div>
          )}
          
          {/* Error overlay - shows on top of existing content if we have previous image */}
          {error && svgContent && (
            <div className="absolute inset-0 bg-red-50/90 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg border-2 border-red-200">
              <div className="flex flex-col items-center p-4 max-w-md">
                <AlertCircle className="w-6 h-6 text-red-500 mb-2" />
                <p className="text-sm text-red-700 font-medium mb-3">{error.message}</p>
                <button
                  onClick={handleRefresh}
                  className="flex items-center gap-2 px-3 py-1.5 text-xs bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>Retry</span>
                </button>
              </div>
            </div>
          )}
          
          {/* SVG content - always rendered if available, stays visible while loading new one */}
          {svgContent && !error ? (
            <div
              key={`svg-${filename}`}
              className="p-4 bg-white"
              dangerouslySetInnerHTML={{ __html: svgContent }}
            />
          ) : !error && !isLoading && (
            <div className="flex items-center justify-center h-96 bg-white">
              <div className="text-center">
                <p className="text-slate-600 mb-2">2D Interaction Diagram</p>
                <p className="text-sm text-slate-500">No diagram available</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
