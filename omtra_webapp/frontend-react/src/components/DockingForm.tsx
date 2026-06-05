'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  apiClient,
  encodeBase64Unicode
} from '@/lib/api-client';
import type { DockingMode, DockingParams, MetricsOptions, PocketInfo, BricsFragment } from '@/types';
import { DEFAULT_METRICS_OPTIONS } from '@/types';
import { FileUpload } from './FileUpload';
import { FixedStructurePanel } from './FixedStructurePanel';
import { MetricsOptionsForm } from './MetricsOptionsForm';
import { LoadFromJobPicker } from './LoadFromJobPicker';
import { PharmacophoreSelector } from './PharmacophoreSelector';
import { useFixedAtomSelection } from '@/hooks/useFixedAtomSelection';
import { AlertCircle, CheckCircle2, Loader2, Play, X, ChevronDown, ChevronRight } from 'lucide-react';

interface DockingFormProps {
  initialDockingMode?: DockingMode;
  onDockingModeChange?: (mode: DockingMode) => void;
  onJobSubmitted: (jobId: string) => void;
  onProteinContentChange?: (content: string | null) => void;
  onProteinFormatChange?: (format: 'pdb' | 'cif' | undefined) => void;
  onLigandContentChange?: (content: string | null) => void;
  onPocketsDetected?: (pockets: PocketInfo[]) => void;
  detectedPockets?: PocketInfo[];
  selectedPocketId?: string | null;
  onPocketSelect?: (pocketId: string | null) => void;
  hiddenPocketIds?: string[];
  onHiddenPocketsChange?: (ids: string[]) => void;
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
  bricsFragments: BricsFragment[];
  setBricsFragments: (frags: BricsFragment[]) => void;
  bricsRawSdf: string | null;
  setBricsRawSdf: (sdf: string | null) => void;
  fixStructureExpanded: boolean;
  setFixStructureExpanded: (v: boolean) => void;
  fixedSelection: ReturnType<typeof useFixedAtomSelection>;
  totalAtomCount: number;
}

export function DockingForm({
  initialDockingMode = 'Rigid Docking',
  onDockingModeChange,
  onJobSubmitted,
  onProteinContentChange,
  onProteinFormatChange,
  onLigandContentChange,
  onPocketsDetected,
  detectedPockets = [],
  selectedPocketId,
  onPocketSelect,
  hiddenPocketIds = [],
  onHiddenPocketsChange,
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
  bricsFragments,
  setBricsFragments,
  bricsRawSdf,
  setBricsRawSdf,
  fixStructureExpanded,
  setFixStructureExpanded,
  fixedSelection,
  totalAtomCount,
}: DockingFormProps) {
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [dockingMode, setDockingModeState] = useState<DockingMode>(initialDockingMode);

  const setDockingMode = useCallback((mode: DockingMode) => {
      setDockingModeState(mode);
      onDockingModeChange?.(mode);
  }, [onDockingModeChange]);

  const prevDockingModeRef = useRef<DockingMode>(dockingMode);
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

  // BRICS fragment selection state (fragments/selectedIds/rawSdf lifted to parent)
  const [bricsLoading, setBricsLoading] = useState(false);

  const togglePocketHidden = useCallback((pocketId: string) => {
    const isHidden = hiddenPocketIds.includes(pocketId);
    const nextHidden = isHidden
      ? hiddenPocketIds.filter((id) => id !== pocketId)
      : [...hiddenPocketIds, pocketId];
    onHiddenPocketsChange?.(nextHidden);
    if (!isHidden && selectedPocketId === pocketId) {
      onPocketSelect?.(null);
    }
  }, [hiddenPocketIds, onHiddenPocketsChange, onPocketSelect, selectedPocketId]);

  const [uploadTokens, setUploadTokens] = useState<string[]>([]); // Deprecated, but keeping for now
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isDetectingPockets, setIsDetectingPockets] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formResetKey, setFormResetKey] = useState(0);
  const [metricsOptions, setMetricsOptions] = useState<MetricsOptions>({ ...DEFAULT_METRICS_OPTIONS });
  const [refLigandFile, setRefLigandFile] = useState<File | null>(null);

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
    setUploadedFiles([]);
    setUploadTokens([]);
    setUseCustomJobId(false);
    setCustomJobId('');
    setError(null);
    setFormResetKey((key) => key + 1);

    // Clear BRICS fragment state
    setBricsFragments([]);
    fixedSelection.resetSelection();
    setBricsRawSdf(null);
    setFixStructureExpanded(false);

    // Notify parent to clear visual state
    onProteinContentChange?.(null);
    onProteinFormatChange?.(undefined);
    onLigandContentChange?.(null);
    onPharmacophoresChange?.([]);
    onPharmacophoreSelectionChange?.([]);
    onPocketsDetected?.([]);
    onPocketSelect?.(null);

    setPocketSelectionMethod('ligand');
    setManualCenter({ x: '0', y: '0', z: '0' });
    setBboxLength('15.0');
    setLigandCenter(null);
    setRefLigandFileName?.(null);
    setRefLigandContent?.(null);
    setRefLigandToken?.(null);
    setRefLigandFile(null);
    setMetricsOptions({ ...DEFAULT_METRICS_OPTIONS });
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
    setBricsFragments,
    setBricsRawSdf,
    fixedSelection,
  ]);

  useEffect(() => {
    // Reset form only when the docking mode actually changes.
    if (prevDockingModeRef.current !== dockingMode) {
      resetForm();
      prevDockingModeRef.current = dockingMode;
    }
  }, [dockingMode, resetForm]);

  // Helper to parse pharmacophores locally (XYZ/JSON) or via API (SDF)
  // Handle Ref Ligand Upload for Step 2
  const handleRefLigandUpload = async (file: File) => {
    setError(null);
    setRefLigandFile(file);
    try {
      // 1. Upload to API
      const token = await uploadFileToApi(file);
      setRefLigandToken?.(token);

      // 2. Read content for Viewer and calculate center
      const content = await file.text();
      const base64Content = encodeBase64Unicode(content);
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
    setLigandToken(null);
    setError(null);
    setBricsFragments([]);
    fixedSelection.resetSelection();
    setBricsRawSdf(null);

    try {
      const content = await file.text();
      const ligandB64 = encodeBase64Unicode(content);
      onLigandContentChange?.(ligandB64);
      setBricsRawSdf(content);

      const token = await uploadFileToApi(file);
      setLigandToken(token);

      // Extract BRICS fragments in background
      setBricsLoading(true);
      try {
        const result = await apiClient.extractBricsFragments(file);
        setBricsFragments(result.fragments);
        if (result.num_fragments > 1) setFixStructureExpanded(true);
      } catch (e) {
        console.error('BRICS extraction failed:', e);
      } finally {
        setBricsLoading(false);
      }
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
        onPharmacophoresChange?.(result.pharmacophores);

        // For SDF, also set content for viewer to center
        if (filename.endsWith('.sdf')) {
          try {
            const content = await file.text();
            onLigandContentChange?.(encodeBase64Unicode(content));
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
      const proteinB64 = encodeBase64Unicode(content);
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

    const capturedFixedIndices =
      dockingMode === 'Rigid Docking' && fixedSelection.fixedCount > 0
        ? fixedSelection.getFixedAtomIndicesForSubmit()
        : undefined;

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

      const finalProteinToken = await uploadFileToApi(proteinFile);
      setProteinToken(finalProteinToken);

      const finalLigandToken = await uploadFileToApi(ligandFile);
      setLigandToken(finalLigandToken);

      let finalRefLigandToken: string | null = null;
      if (refLigandFile) {
        finalRefLigandToken = await uploadFileToApi(refLigandFile);
        setRefLigandToken?.(finalRefLigandToken);
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
        const alphaSphereCenters = selectedPocket?.alpha_sphere_centers ?? [];
        if (selectedPocket && alphaSphereCenters.length > 0) {
          pocketSelection = {
            type: 'coords' as const,
            value: alphaSphereCenters,
          };
        } else if (selectedPocket) {
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
        if (!refLigandFile || !finalRefLigandToken) {
          throw new Error("Reference ligand file is required for this pocket definition. Please upload one in Step 2.");
        }

        pocketSelection = {
          type: 'file' as const,
          value: finalRefLigandToken,
        };
      }

      const fixedAtomIndicesForJob = capturedFixedIndices?.length ? capturedFixedIndices : undefined;

      const params: DockingParams = {
        docking_mode: dockingMode,
        seed: parsedSeed ?? null,
        n_samples: parsedSamples,
        steps: parsedSteps,
        pocket_selection: pocketSelection,
        ...(fixedAtomIndicesForJob ? { fixed_atom_indices: fixedAtomIndicesForJob } : {}),
        metrics_options: metricsOptions,
      };

      const uploads = [finalProteinToken, finalLigandToken];
      if (finalPharmToken) uploads.push(finalPharmToken);
      if (finalRefLigandToken && pocketSelectionMethod === 'ligand') uploads.push(finalRefLigandToken);

      const jobData = {
        params,
        uploads: uploads as string[],
        job_id: useCustomJobId && customJobId.trim() ? customJobId.trim() : undefined,
      };

      const response = await apiClient.submitDockingJob(jobData);

      // Show success notification
      alert(`Job submitted successfully.\n\nJob ID: ${response.job_id}\n\nYour job is now running. Switch to the "Jobs" tab to view progress.`);

      onJobSubmitted(response.job_id);

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
              const num = parseInt(val, 10);
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
          {!proteinFile ? (
            <>
              <FileUpload
                key={`protein-${dockingMode}-${formResetKey}`}
                onFilesUploaded={async (files) => {
                  if (files.length > 0) handleProteinUpload(files[0]);
                  else {
                    onProteinContentChange?.(null);
                    onProteinFormatChange?.(undefined);
                    setProteinFile(null);
                    setProteinToken(null);
                    onPocketsDetected?.([]);
                    onPocketSelect?.(null);
                    setPocketSelectionMethod('ligand');
                    setManualCenter({ x: '0', y: '0', z: '0' });
                    setBboxLength('15.0');
                    setLigandCenter(null);
                  }
                }}
                acceptedTypes={['.pdb', '.cif']}
                maxFiles={1}
                maxSize={25 * 1024 * 1024}
              />
              <LoadFromJobPicker
                key={`load-prot-${dockingMode}-${formResetKey}`}
                acceptedExtensions={['.pdb', '.cif']}
                onFileLoaded={handleProteinUpload}
                disabled={isSubmitting}
              />
            </>
          ) : (
          <div className="p-3 border border-blue-200 bg-blue-50/50 rounded-xl relative group">
  <button
    onClick={() => {
      setProteinFile(null);
      setProteinToken(null);
      onProteinContentChange?.(null);
    }}
    className="absolute top-2 right-2 p-1 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
    title="Remove protein file"
  >
    <X className="w-3.5 h-3.5" />
  </button>
  <div className="flex items-center gap-2.5">
    <div className="flex-1 min-w-0">
      <p className="text-sm font-semibold text-blue-900 truncate">
        {proteinFile.name || 'Protein File'}
      </p>
      <p className="text-[10px] text-blue-600 font-bold uppercase tracking-tight flex items-center gap-1">
        <CheckCircle2 className="w-3 h-3" aria-hidden />
        Processed
      </p>
    </div>
  </div>
</div>
          )}
          {isDetectingPockets && (
            <div className="mt-3 p-3 bg-blue-50/70 rounded-xl text-sm text-blue-700 shadow-sm">
              <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
              Detecting pockets...
            </div>
          )}
        </div>

        {/* Step 2: Pocket Selection */}
        {(proteinFile || refLigandFile) && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 space-y-4">
            <h3 className="text-base font-semibold text-slate-900 mb-2">2. Pocket Selection</h3>
            <div className="flex flex-col gap-3 p-1 bg-slate-100 rounded-xl mb-4">
              <div className="grid grid-cols-3 gap-1">
                <button
                  onClick={() => setPocketSelectionMethod('ligand')}
                  className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${pocketSelectionMethod === 'ligand' ? 'bg-white text-primary-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                >
                  Ref Ligand
                </button>
                <button
                  onClick={() => setPocketSelectionMethod('detected')}
                  className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${pocketSelectionMethod === 'detected' ? 'bg-white text-primary-600 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                    }`}
                >
                  Detected
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
                <p className="text-xs text-slate-500 italic">
                  Note: Defines the pocket as residues 8Å from alpha sphere centers.
                </p>
                {detectedPockets.length > 0 ? (
                  <div className="p-4 border border-emerald-200 bg-emerald-50/50 rounded-xl">
                    <p className="text-sm text-emerald-800 font-medium mb-1">
                      <span className="flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden />
                        {detectedPockets.length} pocket(s) detected.
                      </span>
                    </p>
                    <div className="mt-3 space-y-2">
                      {detectedPockets.map((pocket, idx) => {
                        const isHidden = hiddenPocketIds.includes(pocket.id);
                        const isSelected = selectedPocketId === pocket.id;
                        return (
                          <div
                            key={pocket.id}
                            className={`flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-sm ${
                              isSelected
                                ? 'border-primary-300 bg-white text-primary-700'
                                : 'border-emerald-100 bg-white/70 text-slate-600'
                            } ${isHidden ? 'opacity-60' : ''}`}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                if (!isHidden) onPocketSelect?.(isSelected ? null : pocket.id);
                              }}
                              className="min-w-0 flex-1 text-left"
                            >
                              <span className="font-semibold">Pocket {idx + 1}</span>
                              <span className="ml-2 font-mono text-xs text-slate-500">
                                score {typeof pocket.score === 'number' ? pocket.score.toFixed(3) : 'n/a'} · volume {typeof pocket.volume === 'number' ? pocket.volume.toFixed(1) : 'n/a'}
                              </span>
                            </button>
                            <button
                              type="button"
                              onClick={() => togglePocketHidden(pocket.id)}
                              className={`rounded-md px-2.5 py-1 text-xs font-semibold ${
                                isHidden
                                  ? 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                  : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'
                              }`}
                            >
                              {isHidden ? 'Show' : 'Hide'}
                            </button>
                          </div>
                        );
                      })}
                    </div>
                    {selectedPocketId ? (
                      <p className="text-sm text-slate-600">
                        Pocket <strong>{detectedPockets.find(p => p.id === selectedPocketId)?.id}</strong> selected.<br />
                        Score: {typeof detectedPockets.find(p => p.id === selectedPocketId)?.score === 'number'
                          ? detectedPockets.find(p => p.id === selectedPocketId)?.score?.toFixed(3)
                          : 'n/a'} · Volume: {typeof detectedPockets.find(p => p.id === selectedPocketId)?.volume === 'number'
                          ? detectedPockets.find(p => p.id === selectedPocketId)?.volume?.toFixed(1)
                          : 'n/a'}
                      </p>
                    ) : (
                      <p className="text-sm text-amber-600 animate-pulse font-medium">
                        Select a pocket in the 3D viewer.
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
                {!refLigandFile ? (
                  <>
                  <FileUpload
                    key={`ref-ligand-${dockingMode}-${formResetKey}`}
                    onFilesUploaded={(files) => {
                      if (files.length > 0) handleRefLigandUpload(files[0]);
                      else {
                        setRefLigandFile(null);
                        setRefLigandContent?.(null);
                        setRefLigandToken?.(null);
                        setRefLigandFileName?.(null);
                        setLigandCenter(null);
                        onPocketSelect?.(null);
                        onPocketsDetected?.([]);
                      }
                    }}
                    acceptedTypes={['.sdf']}
                    maxFiles={1}
                    maxSize={5 * 1024 * 1024}
                  />
                  <LoadFromJobPicker
                    acceptedExtensions={['.sdf']}
                    onFileLoaded={handleRefLigandUpload}
                    disabled={isSubmitting}
                  />
                  </>
                ) : (
                  <div className="p-3 border border-blue-200 bg-blue-50/50 rounded-xl relative group">
                    <button
                      onClick={() => {
                        setRefLigandFile(null);
                        setRefLigandContent?.(null);
                        setRefLigandToken?.(null);
                        setRefLigandFileName?.(null);
                        setLigandCenter(null);
                        onPocketSelect?.(null);
                        onPocketsDetected?.([]);
                      }}
                      className="absolute top-2 right-2 p-1.5 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Remove reference ligand"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    <div className="flex items-center gap-3">
  <div className="flex-1 min-w-0">
    <p className="text-sm font-semibold text-blue-900 truncate">
      {refLigandFileName || 'Reference Ligand'}
    </p>
    <p className="text-[10px] text-blue-600 font-bold uppercase tracking-tight flex items-center gap-1">
        <CheckCircle2 className="w-3 h-3" aria-hidden />
        Processed
      </p>
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

            {pocketSelectionMethod === 'manual' && (
            <div className="pt-2">
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
                  className="flex-1 accent-primary-600"
                />
                <div className="w-20 px-2 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-center text-sm font-mono font-bold text-slate-700">
                  {parseFloat(bboxLength).toFixed(1)}
                </div>
              </div>
            </div>
            )}
          </div>
        )}

        {/* Step 3: Pharmacophore (Conditional) */}
        {(proteinFile || pharmacophoreFile) && dockingMode === 'Rigid Docking + Pharmacophore' && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 delay-150">
            <h3 className="text-base font-semibold text-slate-900 mb-2">3. Upload Pharmacophore</h3>
            <p className="text-xs text-slate-500 mb-3">Upload a pharmacophore file (.xyz, .json, .sdf) to guide docking.</p>
            {!pharmacophoreFile ? (
              <>
                <FileUpload
                  key={`pharm-${dockingMode}-${formResetKey}`}
                  onFilesUploaded={(files) => {
                    if (files.length > 0) handlePharmUpload(files[0]);
                    else {
                      setPharmacophoreFile(null);
                      setPharmacophoreToken(null);
                      onPharmacophoresChange?.([]);
                      onLigandContentChange?.(null);
                    }
                  }}
                  acceptedTypes={['.xyz', '.json', '.sdf']}
                  maxFiles={1}
                  maxSize={5 * 1024 * 1024}
                />
                <LoadFromJobPicker
                  acceptedExtensions={['.xyz', '.json', '.sdf']}
                  onFileLoaded={handlePharmUpload}
                  disabled={isSubmitting}
                />
              </>
            ) : (
              <div className="p-3 border border-blue-200 bg-blue-50/50 rounded-xl relative group">
                <button
                  onClick={() => {
                    setPharmacophoreFile(null);
                    setPharmacophoreToken(null);
                    onPharmacophoresChange?.([]);
                    onLigandContentChange?.(null);
                  }}
                  className="absolute top-2 right-2 p-1 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Remove pharmacophore file"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
                <div className="flex items-center gap-2.5">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-blue-900 truncate">
                      {pharmacophoreFile.name || 'Pharmacophore File'}
                    </p>
                    <p className="text-[10px] text-blue-600 font-bold uppercase tracking-tight flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" aria-hidden />
                      Processed
                    </p>
                  </div>
                </div>
              </div>
            )}
            {pharmacophores.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="text-sm font-semibold text-slate-800">Pharmacophore Selection</h4>
                <PharmacophoreSelector
                  pharmacophores={pharmacophores}
                  selectedIndices={selectedPharmacophoreIndices}
                  onSelectionChange={(indices) => onPharmacophoreSelectionChange?.(indices)}
                />
              </div>
            )}
          </div>
        )}

        {/* Step 4: Ligand Upload */}
        {(proteinFile || ligandFile) && (
          <div className="animate-in fade-in slide-in-from-top-4 duration-500 delay-300">
            <h3 className="text-base font-semibold text-slate-900 mb-2">
              {dockingMode === 'Rigid Docking + Pharmacophore' ? '4.' : '3.'} Upload Ligand
            </h3>
            <p className="text-xs text-slate-500 mb-3">Upload the ligand SDF file you want to dock.</p>
            {!ligandFile ? (
              <>
                <FileUpload
                  key={`ligand-${dockingMode}-${formResetKey}`}
                  onFilesUploaded={async (files) => {
                    if (files.length > 0) handleLigandUpload(files[0]);
                    else {
                      setLigandFile(null);
                      setLigandToken(null);
                      onLigandContentChange?.(null);
                      setBricsFragments([]);
                      fixedSelection.resetSelection();
                      setBricsRawSdf(null);
                      setFixStructureExpanded(false);
                    }
                  }}
                  acceptedTypes={['.sdf']}
                  maxFiles={1}
                  maxSize={10 * 1024 * 1024}
                />
                <LoadFromJobPicker
                  acceptedExtensions={['.sdf']}
                  onFileLoaded={handleLigandUpload}
                  disabled={isSubmitting}
                />
              </>
            ) : (
              <div className="p-3 border border-blue-200 bg-blue-50/50 rounded-xl relative group">
                <button
                  onClick={() => {
                    setLigandFile(null);
                    setLigandToken(null);
                    onLigandContentChange?.(null);
                    setBricsFragments([]);
                    fixedSelection.resetSelection();
                    setBricsRawSdf(null);
                    setFixStructureExpanded(false);
                  }}
                  className="absolute top-2 right-2 p-1 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Remove ligand file"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
                <div className="flex items-center gap-2.5">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-blue-900 truncate">
                      {ligandFile.name || 'Ligand File'}
                    </p>
                    <p className="text-[10px] text-blue-600 font-bold uppercase tracking-tight flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" aria-hidden />
                      Processed
                    </p>
                  </div>
                </div>
              </div>
            )}

            {dockingMode === 'Rigid Docking' && ligandFile && (bricsLoading || bricsFragments.length > 0) && (
              <FixedStructurePanel
                expanded={fixStructureExpanded}
                onExpandedChange={setFixStructureExpanded}
                bricsLoading={bricsLoading}
                bricsRawSdf={bricsRawSdf}
                bricsFragments={bricsFragments}
                mode={fixedSelection.mode}
                onModeChange={fixedSelection.switchMode}
                selectionAction={fixedSelection.selectionAction}
                onSelectionActionChange={fixedSelection.setSelectionAction}
                selectedFragmentIds={fixedSelection.selectedFragmentIds}
                mixedFragmentIds={fixedSelection.mixedFragmentIds}
                onFragmentSelectionChange={fixedSelection.setSelectedFragmentIds}
                onAddFragment={fixedSelection.addFragmentAtoms}
                onToggleFragment={fixedSelection.toggleFragmentAtoms}
                fixedCount={fixedSelection.fixedCount}
                totalAtomCount={totalAtomCount}
                onClear={fixedSelection.clearSelection}
                onInvert={fixedSelection.invertSelection}
              />
            )}
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

      <MetricsOptionsForm
        samplingMode={dockingMode}
        options={metricsOptions}
        onChange={setMetricsOptions}
      />

      {
        error && (
          <div className="p-4 bg-red-50/70 rounded-xl text-sm text-red-700 shadow-sm">
            {error}
          </div>
        )
      }

      <div className="flex gap-2">
      <button
        type="button"
        onClick={resetForm}
        disabled={isSubmitting}
        className="px-4 py-3.5 border border-slate-200 text-slate-700 rounded-xl font-semibold hover:bg-slate-50 disabled:opacity-50 transition-all"
      >
        Clear form
      </button>
      <button
        onClick={handleSubmit}
        disabled={isSubmitting}
        className="flex-1 px-5 py-3.5 bg-primary-600 text-white rounded-xl font-semibold hover:bg-primary-700 disabled:bg-slate-400 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md flex items-center justify-center gap-2"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Submitting...
          </>
        ) : (
          <>
            <Play className="w-5 h-5" aria-hidden />
            Run Docking
          </>
        )}
      </button>
      </div>
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

