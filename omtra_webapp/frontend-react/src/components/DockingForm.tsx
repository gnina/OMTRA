'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '@/lib/api-client';
import type { DockingMode, DockingParams, PocketInfo } from '@/types';
import { FileUpload } from './FileUpload';
import { AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';

interface DockingFormProps {
  onJobSubmitted: (jobId: string) => void;
  onProteinContentChange?: (content: string | null) => void;
  onProteinFormatChange?: (format: 'pdb' | 'cif' | undefined) => void;
  onLigandContentChange?: (content: string | null) => void;
  onPocketsDetected?: (pockets: PocketInfo[]) => void;
  detectedPockets?: PocketInfo[];
  selectedPocketId?: string | null;
  onPocketSelect?: (pocketId: string | null) => void;
  onPharmacophoresChange?: (pharmacophores: Array<{ type: string; position: [number, number, number] }>) => void;
  pharmacophores?: Array<{ type: string; position: [number, number, number] }>;
  selectedPharmacophoreIndices?: number[];
  onPharmacophoreSelectionChange?: (indices: number[]) => void;
  pocketSelectionMethod: 'detected' | 'ligand' | 'manual';
  setPocketSelectionMethod: (method: 'detected' | 'ligand' | 'manual') => void;
  manualCenter: { x: string; y: string; z: string };
  setManualCenter: (center: { x: string; y: string; z: string }) => void;
  bboxLength: string;
  setBboxLength: (length: string) => void;
  ligandCenter: [number, number, number] | null;
  setLigandCenter: (center: [number, number, number] | null) => void;
  refLigandContent?: string | null;
  setRefLigandContent?: (content: string | null) => void;
  refLigandToken?: string | null;
  setRefLigandToken?: (token: string | null) => void;
  refLigandFileName?: string | null;
  setRefLigandFileName?: (name: string | null) => void;
}

export function DockingForm({
  onJobSubmitted,
  onProteinContentChange,
  onProteinFormatChange,
  onLigandContentChange,
  onPocketsDetected,
  detectedPockets = [],
  selectedPocketId,
  onPocketSelect,
  onPharmacophoresChange,
  pharmacophores = [],
  selectedPharmacophoreIndices = [],
  onPharmacophoreSelectionChange,
  pocketSelectionMethod,
  setPocketSelectionMethod,
  manualCenter,
  setManualCenter,
  bboxLength,
  setBboxLength,
  ligandCenter,
  setLigandCenter,
  refLigandContent,
  setRefLigandContent,
  refLigandToken,
  setRefLigandToken,
  refLigandFileName,
  setRefLigandFileName,
}: DockingFormProps) {
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [dockingMode, setDockingMode] = useState<DockingMode>('Rigid Docking');
  const [seedInput, setSeedInput] = useState('42');
  const [nSamplesInput, setNSamplesInput] = useState('10');
  const [stepsInput, setStepsInput] = useState('100');
  const [useCustomJobId, setUseCustomJobId] = useState(false);
  const [customJobId, setCustomJobId] = useState('');

  // Specific file states
  const [proteinFile, setProteinFile] = useState<File | null>(null);
  const [ligandFile, setLigandFile] = useState<File | null>(null);
  const [pharmacophoreFile, setPharmacophoreFile] = useState<File | null>(null);

  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]); // Keep for now if needed, or remove later
  // File tokens
  const [proteinToken, setProteinToken] = useState<string | null>(null);
  const [ligandToken, setLigandToken] = useState<string | null>(null);
  const [pharmacophoreToken, setPharmacophoreToken] = useState<string | null>(null);

  const [uploadTokens, setUploadTokens] = useState<string[]>([]); // Deprecated, but keeping for now
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDetectingPockets, setIsDetectingPockets] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resetForm = useCallback(() => {
    setSeedInput('42');
    setNSamplesInput('10');
    setStepsInput('100');
    setProteinFile(null);
    setLigandFile(null);
    setPharmacophoreFile(null);
    setProteinToken(null);
    setLigandToken(null);
    setPharmacophoreToken(null);
    setError(null);

    // Notify parent to clear visual state
    onProteinContentChange?.(null);
    onProteinFormatChange?.(undefined);
    onLigandContentChange?.(null);
    onPharmacophoresChange?.([]);
    onPharmacophoreSelectionChange?.([]);
    onPocketsDetected?.([]);
    onPocketSelect?.(null);

    setPocketSelectionMethod('detected');
    setManualCenter({ x: '0', y: '0', z: '0' });
    setBboxLength('15.0');
    setLigandCenter(null);
    setRefLigandFileName?.(null);
    setRefLigandContent?.(null);
    setRefLigandToken?.(null);
  }, [
    onProteinContentChange,
    onProteinFormatChange,
    onLigandContentChange,
    onPharmacophoresChange,
    onPharmacophoreSelectionChange,
    onPocketsDetected,
    onPocketSelect,
    setRefLigandContent,
    setRefLigandToken,
    setRefLigandFileName,
    setPocketSelectionMethod,
    setManualCenter,
    setBboxLength,
    setLigandCenter,
  ]);

  useEffect(() => {
    // Reset form when docking mode changes
    resetForm();
    // But preserve the new docking mode (resetForm doesn't touch it currently, 
    // but if it did we'd need to restore it)
  }, [dockingMode, resetForm]);

  // Helper to parse pharmacophores locally (XYZ/JSON) or via API (SDF)
  // Handle Ref Ligand Upload for Step 2
  const handleRefLigandUpload = async (file: File) => {
    setError(null);
    try {
      // 1. Upload to API
      const token = await uploadFileToApi(file);
      setRefLigandToken?.(token);

      // 2. Read content for Viewer and calculate center
      const content = await file.text();
      const base64Content = btoa(content);
      setRefLigandContent?.(base64Content);
      setRefLigandFileName?.(file.name);

      // Extract coordinates for center (simplified for common formats like SDF/MOL2)
      // This is a basic attempt and might need more robust parsing for different formats.
      const lines = content.split('\n');
      let xSum = 0, ySum = 0, zSum = 0, count = 0;
      let atomBlock = false;
      for (const line of lines) {
        // For SDF/MOL2, atom block usually starts after header and before M  END
        // This is a very basic heuristic.
        if (line.includes('V2000') || line.includes('V3000') || line.includes('MOL')) { atomBlock = true; continue; }
        if (atomBlock && line.trim().split(/\s+/).length >= 4) {
          // Assuming XYZ coordinates are the first three numbers
          const parts = line.trim().split(/\s+/);
          const x = parseFloat(parts[0]);
          const y = parseFloat(parts[1]);
          const z = parseFloat(parts[2]);
          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            xSum += x; ySum += y; zSum += z; count++;
          }
        }
        if (line.includes('M  END') || line.includes('$$$$')) break; // End of molecule block
      }
      if (count > 0) {
        setLigandCenter([xSum / count, ySum / count, zSum / count]);
      } else {
        // Try fallback for SDF files - sometimes the atom block is just lines with 10+ columns
        let xFallback = 0, yFallback = 0, zFallback = 0, countFallback = 0;
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.length > 30) {
            const parts = trimmed.split(/\s+/);
            if (parts.length >= 10) {
              const x = parseFloat(parts[0]);
              const y = parseFloat(parts[1]);
              const z = parseFloat(parts[2]);
              if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
                xFallback += x; yFallback += y; zFallback += z; countFallback++;
              }
            }
          }
        }
        if (countFallback > 0) {
          setLigandCenter([xFallback / countFallback, yFallback / countFallback, zFallback / countFallback]);
        } else {
          console.warn("Could not determine center from reference ligand file.");
          setLigandCenter(null);
        }
      }
    } catch (err: any) {
      console.error('Ref ligand upload failed:', err);
      setError(`Failed to upload reference ligand: ${err.message}`);
      setRefLigandToken?.(null);
      setRefLigandContent?.(null);
      setLigandCenter(null);
    }
  };

  const handleLigandUpload = async (file: File) => {
    setLigandFile(file);
    setLigandToken(null); // Reset token until uploaded
    setError(null);

    try {
      // 1. Read content for Viewer (if needed, though docking ligand isn't usually shown)
      const content = await file.text();
      const ligandB64 = btoa(content);
      onLigandContentChange?.(ligandB64); // Pass to parent for viewer if it handles docking ligand

      // 2. Upload to API
      const token = await uploadFileToApi(file);
      setLigandToken(token);
    } catch (err: any) {
      console.error('Docking ligand upload failed:', err);
      setError(`Failed to upload docking ligand file: ${err.message}`);
      setLigandToken(null);
    }
  };

  const handlePharmUpload = async (file: File) => {
    setPharmacophoreFile(file);
    setPharmacophoreToken(null);
    if (!onPharmacophoresChange) return;

    try {
      const filename = file.name.toLowerCase();

      // Upload file to API first (needed for job submission)
      const token = await uploadFileToApi(file);
      setPharmacophoreToken(token);

      // Extract pharmacophores for display using backend API for all formats
      if (filename.endsWith('.sdf') || filename.endsWith('.json') || filename.endsWith('.xyz')) {
        const result = await apiClient.extractPharmacophore(file);
        onPharmacophoresChange(result.pharmacophores);

        // For SDF, also set content for viewer to center
        if (filename.endsWith('.sdf')) {
          try {
            const content = await file.text();
            onLigandContentChange?.(btoa(content));
          } catch (e) {
            console.error("Failed to read pharmacophore SDF for viewer:", e);
          }
        }
      } else {
        console.error("Unsupported pharmacophore file format:", filename);
        setError(`Unsupported file format. Please upload SDF, XYZ, or JSON file.`);
      }
    } catch (e) {
      console.error("Failed to process pharmacophore file:", e);
      setError(`Failed to process pharmacophore file: ${e instanceof Error ? e.message : String(e)}`);
      setPharmacophoreToken(null);
    }
  };

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const isHealthy = await apiClient.healthCheck();
        setApiConnected(isHealthy);
      } catch (error) {
        console.error('API health check error:', error);
        setApiConnected(false);
      }
    };
    checkHealth();
  }, []);

  const uploadFileToApi = async (file: File): Promise<string> => {
    const initResponse = await apiClient.initUpload();
    await apiClient.uploadFile(initResponse.upload_token, file);
    return initResponse.upload_token;
  };

  const handleProteinUpload = async (file: File) => {
    setProteinFile(file);
    setProteinToken(null); // Reset token until uploaded
    setError(null);

    const filename = file.name.toLowerCase();
    try {
      // 1. Read content for Viewer
      const content = await file.text();
      const proteinB64 = btoa(content);
      onProteinContentChange?.(proteinB64);
      const format = filename.endsWith('.pdb') ? 'pdb' : 'cif';
      onProteinFormatChange?.(format);

      // 2. Upload to API
      try {
        const token = await uploadFileToApi(file);
        setProteinToken(token);
      } catch (err: any) {
        console.error('Protein upload failed:', err);
        setError('Failed to upload protein file to server');
        return;
      }

      // 3. Detect pockets
      setIsDetectingPockets(true);
      try {
        const result = await apiClient.detectPockets(file);
        onPocketsDetected?.(result.pockets);
      } catch (err: any) {
        console.error('Pocket detection failed:', err);
      } finally {
        setIsDetectingPockets(false);
      }
    } catch (err: any) {
      setError(`Failed to process protein file: ${err.message}`);
    }
  };

  const handleSubmit = async () => {
    setError(null);
    setIsSubmitting(true);

    try {
      // Validate required files (check FILE objects, not tokens)
      if (!proteinFile) {
        setError('Docking requires a protein file');
        setIsSubmitting(false);
        return;
      }
      if (!ligandFile) {
        setError('Docking requires a ligand file');
        setIsSubmitting(false);
        return;
      }

      // Ensure protein is uploaded (re-upload if needed)
      let finalProteinToken = proteinToken;
      if (!finalProteinToken && proteinFile) {
        try {
          finalProteinToken = await uploadFileToApi(proteinFile);
          setProteinToken(finalProteinToken);
        } catch (err) {
          throw new Error("Failed to upload protein file");
        }
      }

      // Ensure ligand is uploaded (re-upload if needed)
      let finalLigandToken = ligandToken;
      if (!finalLigandToken && ligandFile) {
        try {
          finalLigandToken = await uploadFileToApi(ligandFile);
          setLigandToken(finalLigandToken);
        } catch (err) {
          throw new Error("Failed to upload ligand file");
        }
      }

      // Ensure pharmacophore is uploaded or generated
      let finalPharmToken = pharmacophoreToken;

      if (dockingMode === 'Rigid Docking + Pharmacophore') {
        if (!pharmacophoreFile) {
          setError('Pharmacophore file is required for this mode');
          setIsSubmitting(false);
          return;
        }

        // For ALL file types (SDF, JSON, XYZ), convert selected pharmacophores to XYZ
        // This ensures only selected spheres are used
        if (!pharmacophores || pharmacophores.length === 0) {
          throw new Error("No pharmacophores found. Please ensure the file contains valid pharmacophores.");
        }

        // If no selection, default to ALL
        const indicesToUse = selectedPharmacophoreIndices.length > 0
          ? selectedPharmacophoreIndices
          : pharmacophores.map((_, i) => i);

        try {
          // Convert selected pharmacophores to XYZ format
          // Pass false for center to keep coordinates relative to protein pocket
          const xyzResult = await apiClient.pharmacophoreToXyz(pharmacophores, indicesToUse, false);

          // Upload the generated XYZ content with only selected features
          if (xyzResult && xyzResult.xyz_content) {
            const blob = new Blob([xyzResult.xyz_content], { type: 'text/plain' });
            const file = new File([blob], "selected_pharmacophores.xyz");
            finalPharmToken = await uploadFileToApi(file);
          } else {
            throw new Error("Backend returned empty XYZ content");
          }
        } catch (err: any) {
          throw new Error(`Failed to process selected pharmacophores: ${err.message}`);
        }
      }

      const parsedSamples = parseInt(nSamplesInput, 10);
      if (Number.isNaN(parsedSamples) || parsedSamples < 1 || parsedSamples > 20) {
        setError('Number of samples must be between 1 and 20');
        setIsSubmitting(false);
        return;
      }

      const parsedSteps = parseInt(stepsInput, 10);
      if (Number.isNaN(parsedSteps) || parsedSteps < 10 || parsedSteps > 300) {
        setError('Sampling steps must be between 10 and 300');
        setIsSubmitting(false);
        return;
      }

      let parsedSeed: number | null = null;
      if (seedInput.trim() !== '') {
        const numericSeed = Number(seedInput);
        if (!Number.isInteger(numericSeed) || numericSeed < 0) {
          setError('Random seed must be a non-negative integer');
          setIsSubmitting(false);
          return;
        }
        parsedSeed = numericSeed;
      }

      // Build pocket selection based on selected method
      let pocketSelection = undefined;

      if (pocketSelectionMethod === 'detected' && selectedPocketId) {
        const selectedPocket = detectedPockets.find(p => p.id === selectedPocketId);
        if (selectedPocket) {
          pocketSelection = {
            type: 'center' as const,
            value: selectedPocket.center,
            bbox_length: parseFloat(bboxLength),
          };
        }
      } else if (pocketSelectionMethod === 'manual') {
        const center: [number, number, number] = [
          parseFloat(manualCenter.x),
          parseFloat(manualCenter.y),
          parseFloat(manualCenter.z)
        ];
        if (center.some(isNaN)) throw new Error("Manual coordinates must be valid numbers");

        pocketSelection = {
          type: 'center' as const,
          value: center,
          bbox_length: parseFloat(bboxLength),
        };
      } else if (pocketSelectionMethod === 'ligand') {
        if (!refLigandToken) throw new Error("Reference ligand file is required for this pocket definition. Please upload one in Step 2.");

        pocketSelection = {
          type: 'file' as const,
          value: refLigandToken,
        };
      }

      const params: DockingParams = {
        docking_mode: dockingMode,
        seed: parsedSeed ?? null,
        n_samples: parsedSamples,
        steps: parsedSteps,
        pocket_selection: pocketSelection,
      };

      const uploads = [finalProteinToken, finalLigandToken];
      if (finalPharmToken) uploads.push(finalPharmToken);

      const jobData = {
        params,
        uploads: uploads as string[],
        job_id: useCustomJobId && customJobId.trim() ? customJobId.trim() : undefined,
      };

      const response = await apiClient.submitDockingJob(jobData);

      // Show success notification
      alert(`✅ Job submitted successfully!\n\nJob ID: ${response.job_id}\n\nYour job is now running. Switch to the "Jobs" tab to view progress.`);

      onJobSubmitted(response.job_id);

      // Clear tokens so they are re-uploaded on next submit (since backend consumes them)
      setProteinToken(null);
      setLigandToken(null);
      setPharmacophoreToken(null);

    } catch (err: any) {
      setError(`Job submission failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (apiConnected === null) {
    return (
      <div className="flex items-center justify-center p-4">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
      </div>
    );
  }

  if (apiConnected === false) {
    return (
      <div className="p-4 bg-red-50/70 rounded-xl shadow-sm">
        <div className="flex items-center gap-2 text-red-700">
          <AlertCircle className="w-5 h-5" />
          <span className="font-medium">API Unavailable</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 text-emerald-600 mb-4">
        <CheckCircle2 className="w-5 h-5" />
        <span className="text-sm font-medium">API Connected</span>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Docking Mode
          </label>
          <select
            value={dockingMode}
            onChange={(e) => setDockingMode(e.target.value as DockingMode)}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          >
            <option value="Rigid Docking">Rigid Docking</option>
            <option value="Rigid Docking + Pharmacophore">Rigid Docking + Pharmacophore</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Random Seed
          </label>
          <input
            type="number"
            value={seedInput}
            onChange={(e) => setSeedInput(e.target.value)}
            min={0}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Number of Samples (1-20)
          </label>
          <input
            type="number"
            value={nSamplesInput}
            onChange={(e) => {
              const val = e.target.value;
              if (val === '') {
                setNSamplesInput(val);
                return;
              }
              const num = parseInt(val, 10);
              // Prevent typing > 20
              if (!isNaN(num) && num <= 20) {
                setNSamplesInput(val);
              }
            }}
            min={1}
            max={20}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Sampling Steps (50-300)
          </label>
          <input
            type="number"
            value={stepsInput}
            onChange={(e) => {
              const val = e.target.value;
              if (val === '') {
                setStepsInput(val);
                return;
              }
              const num = parseInt(val, 50);
              // Prevent typing > 300
              if (!isNaN(num) && num <= 300) {
                setStepsInput(val);
              }
            }}
            min={10}
            max={300}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>
      </div>

      <div className="border-t border-slate-200/60 pt-6 space-y-6">
        {/* Step 1: Protein Upload */}
        <div>
          <h3 className="text-base font-semibold text-slate-900 mb-2">1. Upload Protein</h3>
          <p className="text-xs text-slate-500 mb-3">Upload a PDB or CIF file to define the receptor.</p>
          <FileUpload
            key={`protein-${dockingMode}`}
            onFilesUploaded={async (files) => {
              if (files.length > 0) handleProteinUpload(files[0]);
              else {
                onProteinContentChange?.(null);
                onProteinFormatChange?.(undefined);
                setProteinFile(null);
                setProteinToken(null);
                // Clear pockets and bounding boxes when protein is removed
                onPocketsDetected?.([]);
                onPocketSelect?.(null);
                setPocketSelectionMethod('detected');
                setManualCenter({ x: '0', y: '0', z: '0' });
                setBboxLength('15.0');
                setLigandCenter(null);
              }
            }}
            acceptedTypes={['.pdb', '.cif']}
            maxFiles={1}
            maxSize={25 * 1024 * 1024}
          />
          {isDetectingPockets && (
            <div className="mt-3 p-3 bg-blue-50/70 rounded-xl text-sm text-blue-700 shadow-sm">
              <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
              Detecting pockets...
            </div>
          )}
        </div>

        {/* Step 2: Pocket Selection */}
        {proteinFile && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 space-y-4">
            <h3 className="text-base font-semibold text-slate-900 mb-2">2. Pocket Selection</h3>
            <div className="flex flex-col gap-3 p-1 bg-slate-100 rounded-xl mb-4">
              <div className="grid grid-cols-3 gap-1">
                <button
                  onClick={() => setPocketSelectionMethod('detected')}
                  className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${pocketSelectionMethod === 'detected' ? 'bg-white text-primary-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                >
                  Detected
                </button>
                <button
                  onClick={() => setPocketSelectionMethod('ligand')}
                  className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${pocketSelectionMethod === 'ligand' ? 'bg-white text-primary-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                >
                  Ref Ligand
                </button>
                <button
                  onClick={() => setPocketSelectionMethod('manual')}
                  className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${pocketSelectionMethod === 'manual' ? 'bg-white text-primary-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                >
                  Manual
                </button>
              </div>
            </div>

            {pocketSelectionMethod === 'detected' && (
              <div className="space-y-3">
                {detectedPockets.length > 0 ? (
                  <div className="p-4 border border-emerald-200 bg-emerald-50/50 rounded-xl">
                    <p className="text-sm text-emerald-800 font-medium mb-1">
                      ✓ {detectedPockets.length} pocket(s) detected.
                    </p>
                    {selectedPocketId ? (
                      <p className="text-sm text-slate-600">
                        Pocket <strong>{detectedPockets.find(p => p.id === selectedPocketId)?.id}</strong> selected.<br />
                        Center: {detectedPockets.find(p => p.id === selectedPocketId)?.center.map((c: number) => c.toFixed(2)).join(', ')}
                      </p>
                    ) : (
                      <p className="text-sm text-amber-600 animate-pulse font-medium">
                        👆 Select a pocket in the 3D viewer.
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="p-3 bg-slate-100 rounded-xl text-sm text-slate-500">
                    {isDetectingPockets ? "Detecting pockets..." : "No pockets detected automatically."}
                  </div>
                )}
              </div>
            )}

            {pocketSelectionMethod === 'ligand' && (
              <div className="space-y-3">
                <p className="text-xs text-slate-500 italic mb-2">
                  Note: Defines the pocket as residues 8Å from reference ligand atoms.
                </p>
                {!refLigandToken ? (
                  <FileUpload
                    key="ref-ligand-upload"
                    onFilesUploaded={(files) => {
                      if (files.length > 0) handleRefLigandUpload(files[0]);
                      else {
                        setRefLigandContent?.(null);
                        setRefLigandToken?.(null);
                        setRefLigandFileName?.(null);
                        setLigandCenter(null);
                      }
                    }}
                    acceptedTypes={['.sdf']}
                    maxFiles={1}
                    maxSize={5 * 1024 * 1024}
                  />
                ) : (
                  <div className="p-4 border border-blue-200 bg-blue-50/50 rounded-xl relative group">
                    <button
                      onClick={() => {
                        setRefLigandContent?.(null);
                        setRefLigandToken?.(null);
                        setRefLigandFileName?.(null);
                        setLigandCenter(null);
                      }}
                      className="absolute top-2 right-2 p-1.5 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Remove reference ligand"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 rounded-lg">
                        <span className="text-xl">🧬</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-blue-900 truncate">
                          {refLigandFileName || 'Reference Ligand'}
                        </p>
                        <p className="text-xs text-blue-600 font-medium">✓ Uploaded & Processed</p>
                      </div>
                    </div>
                    {ligandCenter && (
                      <div className="mt-3 pt-3 border-t border-blue-100/50">
                        <p className="text-[10px] font-bold text-blue-400 uppercase tracking-tighter mb-1">Center</p>
                        <p className="text-xs font-mono text-blue-700 bg-blue-100/30 px-2 py-1 rounded inline-block">
                          [{ligandCenter[0].toFixed(2)}, {ligandCenter[1].toFixed(2)}, {ligandCenter[2].toFixed(2)}]
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {pocketSelectionMethod === 'manual' && (
              <div className="space-y-4 p-4 border border-slate-200 rounded-xl bg-white shadow-sm">
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wider">Manual Center Coordinates</p>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[10px] font-bold text-slate-400 mb-1">X</label>
                    <input
                      type="number"
                      value={manualCenter.x}
                      onChange={(e) => setManualCenter({ ...manualCenter, x: e.target.value })}
                      className="w-full px-2 py-1.5 border border-slate-200 rounded-lg text-sm"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold text-slate-400 mb-1">Y</label>
                    <input
                      type="number"
                      value={manualCenter.y}
                      onChange={(e) => setManualCenter({ ...manualCenter, y: e.target.value })}
                      className="w-full px-2 py-1.5 border border-slate-200 rounded-lg text-sm"
                      step="0.1"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-bold text-slate-400 mb-1">Z</label>
                    <input
                      type="number"
                      value={manualCenter.z}
                      onChange={(e) => setManualCenter({ ...manualCenter, z: e.target.value })}
                      className="w-full px-2 py-1.5 border border-slate-200 rounded-lg text-sm"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>
            )}

            <div className={`pt-2 ${pocketSelectionMethod === 'ligand' ? 'opacity-40 grayscale pointer-events-none' : ''}`}>
              <label className="block text-sm font-semibold text-slate-700 mb-2">
                Adjust Bounding Box Length
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="5.0"
                  max="35.0"
                  step="0.5"
                  value={bboxLength}
                  onChange={(e) => setBboxLength(e.target.value)}
                  disabled={pocketSelectionMethod === 'ligand'}
                  className="flex-1 accent-primary-600"
                />
                <div className="w-16 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-center text-sm font-mono font-bold text-slate-700">
                  {parseFloat(bboxLength).toFixed(1)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Pharmacophore (Conditional) */}
        {proteinFile && dockingMode === 'Rigid Docking + Pharmacophore' && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 delay-150">
            <h3 className="text-base font-semibold text-slate-900 mb-2">3. Upload Pharmacophore</h3>
            <p className="text-xs text-slate-500 mb-3">Upload a pharmacophore file (.xyz, .json, .sdf) to guide docking.</p>
            <FileUpload
              key={`pharm-${dockingMode}`}
              onFilesUploaded={(files) => {
                if (files.length > 0) handlePharmUpload(files[0]);
                else {
                  setPharmacophoreFile(null);
                  onPharmacophoresChange?.([]);
                }
              }}
              acceptedTypes={['.xyz', '.json', '.sdf']}
              maxFiles={1}
              maxSize={5 * 1024 * 1024}
            />
            {/* Pharmacophore Selection Indicator */}
            {pharmacophores.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="text-sm font-semibold text-slate-800">
                  Pharmacophore Selection
                </h4>
                <div className="mb-3 text-sm text-slate-600 italic">
                  💡 Click spheres in the 3D viewer to select/deselect features
                </div>
                {selectedPharmacophoreIndices.length > 0 ? (
                  <div className="mt-3 p-3 bg-emerald-50/70 rounded-xl text-sm text-emerald-700 shadow-sm border border-emerald-100 flex items-center justify-between">
                    <span>✓ {selectedPharmacophoreIndices.length} of {pharmacophores.length} features selected</span>
                    <button
                      onClick={() => onPharmacophoreSelectionChange?.([])}
                      className="text-emerald-600 hover:text-emerald-700 text-xs font-semibold"
                    >
                      Clear All
                    </button>
                  </div>
                ) : (
                  <div className="p-3 bg-amber-50 rounded-xl text-sm text-amber-700 animate-pulse font-medium border border-amber-100">
                    👆 Select pharmacophore features in the 3D viewer
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Step 4: Ligand Upload */}
        {proteinFile && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 delay-300">
            <h3 className="text-base font-semibold text-slate-900 mb-2">
              {dockingMode === 'Rigid Docking + Pharmacophore' ? '4.' : '3.'} Upload Ligand
            </h3>
            <p className="text-xs text-slate-500 mb-3">Upload the ligand SDF file you want to dock.</p>
            <FileUpload
              key={`ligand-${dockingMode}`}
              onFilesUploaded={async (files) => {
                if (files.length > 0) handleLigandUpload(files[0]);
                else {
                  setLigandFile(null);
                  setLigandToken(null);
                  onLigandContentChange?.(null);
                }
              }}
              acceptedTypes={['.sdf']}
              maxFiles={1}
              maxSize={10 * 1024 * 1024}
            />
          </div>
        )}
      </div>

      <div className="border-t border-slate-200/60 pt-6">
        <h3 className="text-base font-semibold text-slate-900 mb-3">Job Settings</h3>
        <div className="space-y-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useCustomJobId}
              onChange={(e) => setUseCustomJobId(e.target.checked)}
              className="w-4 h-4 text-primary-600 border-slate-300 rounded focus:ring-primary-500 focus:ring-2"
            />
            <span className="text-sm text-slate-700">Use custom job ID</span>
          </label>
          {useCustomJobId && (
            <input
              type="text"
              value={customJobId}
              onChange={(e) => setCustomJobId(e.target.value)}
              placeholder="my-custom-job-123"
              className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
            />
          )}
        </div>
      </div>

      {
        error && (
          <div className="p-4 bg-red-50/70 rounded-xl text-sm text-red-700 shadow-sm">
            {error}
          </div>
        )
      }

      <button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className="w-full px-5 py-3.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 disabled:bg-slate-400 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md flex items-center justify-center gap-2"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Submitting...
          </>
        ) : (
          <>
            🚀 Run Docking
          </>
        )}
      </button>
    </div >
  );
}

function calculateCenterFromSdf(content: string): [number, number, number] | null {
  const lines = content.split('\n');
  const atomLines: string[] = [];
  let foundCountsLine = false;
  let numAtoms = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!foundCountsLine && i >= 3) {
      const parts = line.split(/\s+/).filter(Boolean);
      if (parts.length >= 2) {
        numAtoms = parseInt(parts[0], 10);
        if (!isNaN(numAtoms)) {
          foundCountsLine = true;
          continue;
        }
      }
    }
    if (foundCountsLine && atomLines.length < numAtoms) {
      atomLines.push(line);
    }
  }

  if (atomLines.length === 0) return null;

  let sumX = 0, sumY = 0, sumZ = 0;
  let validAtoms = 0;

  for (const line of atomLines) {
    const parts = line.split(/\s+/).filter(Boolean);
    if (parts.length >= 3) {
      const x = parseFloat(parts[0]);
      const y = parseFloat(parts[1]);
      const z = parseFloat(parts[2]);
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        sumX += x;
        sumY += y;
        sumZ += z;
        validAtoms++;
      }
    }
  }

  if (validAtoms === 0) return null;
  return [sumX / validAtoms, sumY / validAtoms, sumZ / validAtoms];
}

