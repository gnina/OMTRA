'use client';

import { useState, useEffect, useMemo } from 'react';
import { apiClient } from '@/lib/api-client';
import { Loader2, AlertCircle } from 'lucide-react';

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

// Helper function to check if SVG is blank/empty
function isBlankSvg(svg: string | null | undefined): boolean {
  if (!svg) return true;
  const trimmed = svg.trim().replace(/\s+/g, ' ');
  if (!trimmed) return true;
  if (trimmed === '<svg></svg>' || trimmed === '<svg/>') return true;
  
  // Check if it's just a border rectangle (border-only SVG)
  const borderPatterns = [
    /<path[^>]*d="[^"]*M\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+L\s+\d+\s+\d+\s+Z/i,  // Standard border
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+600\s+0\s+L\s+600\s+600\s+L\s+0\s+600\s+Z/i,  // Exact 600x600 border
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+\d+\s+0\s+L\s+\d+\s+\d+\s+L\s+0\s+\d+\s+Z/i,  // Border rectangle pattern
    /<path[^>]*d="[^"]*M\s+0\s+0\s+L\s+\d+\s+0\s+L\s+\d+\s+\d+\s+L\s+0\s+\d+\s+Z\s+M\s+0\s+0/i,  // Border with extra M
  ];
  
  // Count path elements
  const pathMatches = trimmed.match(/<path[^>]*>/gi);
  const pathCount = pathMatches ? pathMatches.length : 0;
  
  for (const pattern of borderPatterns) {
    if (pattern.test(trimmed)) {
      // If it's a short SVG with only border, it's blank
      if (trimmed.length < 500) return true;
      // Even if longer, if it only has one path element and it's a border, it's blank
      if (pathCount === 1) return true;
    }
  }
  
  // Check if it has meaningful content
  const hasMeaningfulContent = 
    /<text[^>]*>/i.test(trimmed) ||
    /<circle[^>]*r="[^"]*"[^>]*>/i.test(trimmed) ||
    (/<path[^>]*d="[^"]*[ML][^"]*[ML]"/i.test(trimmed) && trimmed.length > 500) ||
    /<path[^>]*d="[^"]*[CcQqSsTtAaZz]/.test(trimmed);  // Curved paths
  
  // If it's a very short SVG with no meaningful content, it's blank
  if (!hasMeaningfulContent && trimmed.length < 500) return true;
  
  return false;
}

// Helper function to extract error details from API error
export const extractInteractionDiagramErrorDetails = (err: any): DiagramError => {
  // Always return the same simple message - no details
  return { message: 'PoseView failed to generate diagram' };
};

// Prefetch diagram for given job+filename combination so that the UI can show it instantly later.
export async function prefetchInteractionDiagram(jobId: string, filename: string): Promise<void> {
  const cacheKey = `${jobId}/${filename}`;

  if (svgCache.has(cacheKey) || errorCache.has(cacheKey)) {
    return;
  }

  try {
    const svg = await apiClient.getInteractionDiagram(jobId, filename);
    // Check if SVG is blank - treat blank SVGs as errors
    if (isBlankSvg(svg)) {
      const errorDetails: DiagramError = {
        message: 'PoseView failed to generate diagram',
        reason: 'Generated diagram is empty or blank'
      };
      errorCache.set(cacheKey, errorDetails);
      svgCache.delete(cacheKey);
    } else {
      svgCache.set(cacheKey, svg);
      errorCache.delete(cacheKey);
    }
  } catch (err) {
    const errorDetails = extractInteractionDiagramErrorDetails(err);
    errorCache.set(cacheKey, errorDetails);
    svgCache.delete(cacheKey);
  }
}

export function InteractionDiagram2D({ jobId, filename }: InteractionDiagram2DProps) {
  const cacheKey = useMemo(() => `${jobId}/${filename}`, [jobId, filename]);
  
  const [svgContent, setSvgContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<DiagramError | null>(null);
  
  // Reset state and load from cache or API when filename changes
  useEffect(() => {
    // Get cached data for this cacheKey
    const cachedError = errorCache.get(cacheKey) || null;
    const cachedSvg = svgCache.get(cacheKey) || null;
    const cachedSvgIsBlank = cachedSvg ? isBlankSvg(cachedSvg) : false;
    
    // If we have valid cached SVG, use it immediately
    if (cachedSvg && !cachedSvgIsBlank) {
      setSvgContent(cachedSvg);
      setError(null);
      setIsLoading(false);
      return;
    }
    
    // If we have cached error, show it immediately
    if (cachedError) {
      setError(cachedError);
      setSvgContent(null);
      setIsLoading(false);
      return;
    }
    
    // Otherwise, reset state and load from API
    setSvgContent(null);
    setError(null);
    setIsLoading(true);
    
    // Load diagram from API
    const loadDiagram = async () => {
      try {
        const svg = await apiClient.getInteractionDiagram(jobId, filename);
        const isBlank = isBlankSvg(svg);
        
        if (isBlank) {
          const errorDetails: DiagramError = { message: 'PoseView failed to generate diagram' };
          setError(errorDetails);
          setSvgContent(null);
          setIsLoading(false);
          errorCache.set(cacheKey, errorDetails);
          svgCache.delete(cacheKey);
        } else {
          setSvgContent(svg);
          setError(null);
          setIsLoading(false);
          svgCache.set(cacheKey, svg);
          errorCache.delete(cacheKey);
        }
      } catch (err) {
        const errorDetails = extractInteractionDiagramErrorDetails(err);
        setError(errorDetails);
        setSvgContent(null);
        setIsLoading(false);
        errorCache.set(cacheKey, errorDetails);
        svgCache.delete(cacheKey);
      }
    };

    loadDiagram();
  }, [jobId, filename, cacheKey]);

  // Always render the same structure - keep UI static
  return (
    <div className="flex flex-col items-center justify-center min-h-96 bg-white rounded-lg p-4 overflow-auto shadow-sm">
      <div className="w-full max-w-4xl">
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-900">2D Interaction Diagram</h3>
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
              <p className="text-slate-900 font-semibold">{error.message}</p>
            </div>
          )}
          
          {/* Error overlay - shows on top of existing content if we have previous image */}
          {error && svgContent && (
            <div className="absolute inset-0 bg-red-50/90 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg border-2 border-red-200">
              <div className="flex flex-col items-center p-4 max-w-md">
                <AlertCircle className="w-6 h-6 text-red-500 mb-2" />
                <p className="text-sm text-red-700 font-medium">{error.message}</p>
              </div>
            </div>
          )}
          
          {/* SVG content - only render if we have valid content and no error */}
          {svgContent && !error ? (
            <div
              key={`svg-${filename}`}
              className="p-4 bg-white"
              dangerouslySetInnerHTML={{ __html: svgContent }}
            />
          ) : null}
          
          {/* Empty state - only show if no error, no content, and not loading */}
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
    </div>
  );
}
