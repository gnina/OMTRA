'use client';

import { useState, useEffect, useRef } from 'react';
import { apiClient } from '@/lib/api-client';
import type { SamplingMode, SamplingParams } from '@/types';
import { PharmacophoreViewer } from './PharmacophoreViewer';
import { FileUpload } from './FileUpload';
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface JobSubmissionFormProps {
  onJobSubmitted: (jobId: string) => void;
}

export function JobSubmissionForm({ onJobSubmitted }: JobSubmissionFormProps) {
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [samplingMode, setSamplingMode] = useState<SamplingMode>('Unconditional');
  const [seedInput, setSeedInput] = useState('42');
  const [nSamplesInput, setNSamplesInput] = useState('10');
  const [stepsInput, setStepsInput] = useState('100');
  const [nLigAtomsMeanInput, setNLigAtomsMeanInput] = useState('');
  const [nLigAtomsStdInput, setNLigAtomsStdInput] = useState('');
  const [autoAtomCount, setAutoAtomCount] = useState<number | null>(null);
  const [useCustomJobId, setUseCustomJobId] = useState(false);
  const [customJobId, setCustomJobId] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [uploadTokens, setUploadTokens] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formResetKey, setFormResetKey] = useState(0);

  // Pharmacophore state
  const [extractedPharmacophores, setExtractedPharmacophores] = useState<
    Array<{ type: string; position: [number, number, number] }>
  >([]);
  const [selectedPharmacophoreIndices, setSelectedPharmacophoreIndices] = useState<Set<number>>(new Set());
  const [ligandContent, setLigandContent] = useState<string | null>(null);
  const [proteinContent, setProteinContent] = useState<string | null>(null);
  const ATOM_STD_MARGIN = 0.15;
  const prevSamplingModeRef = useRef<SamplingMode>(samplingMode);

  const resetForm = () => {
    setSamplingMode('Unconditional');
    setSeedInput('42');
    setNSamplesInput('10');
    setStepsInput('100');
    setNLigAtomsMeanInput('');
    setNLigAtomsStdInput('');
    setAutoAtomCount(null);
    setUploadedFiles([]);
    setUploadTokens([]);
    setExtractedPharmacophores([]);
    setSelectedPharmacophoreIndices(new Set());
    setLigandContent(null);
    setProteinContent(null);
    setCustomJobId('');
    setUseCustomJobId(false);
    setError(null);
    setFormResetKey((key) => key + 1);
  };

  useEffect(() => {
    // Check API health
    const checkHealth = async () => {
      try {
        const isHealthy = await apiClient.healthCheck();
        setApiConnected(isHealthy);
        if (!isHealthy) {
          console.error('API health check failed. Check browser console for details.');
        }
      } catch (error) {
        console.error('API health check error:', error);
        setApiConnected(false);
      }
    };
    checkHealth();
  }, []);

  useEffect(() => {
    // Reset all form state when switching sampling modes
    if (prevSamplingModeRef.current !== samplingMode) {
      setUploadedFiles([]);
      setUploadTokens([]);
      setExtractedPharmacophores([]);
      setSelectedPharmacophoreIndices(new Set());
      setLigandContent(null);
      setProteinContent(null);
      setAutoAtomCount(null);
      setNLigAtomsMeanInput('');
      setNLigAtomsStdInput('');
      setFormResetKey((key) => key + 1); // Force FileUpload component to reset
      prevSamplingModeRef.current = samplingMode;
    }
  }, [samplingMode]);

  const fileSignature = (file: File) => `${file.name}:${file.size}:${file.lastModified}`;

  const handleFileUpload = async (files: File[]) => {
    setError(null);
    const newTokens: string[] = [];
    const prevFileMap = new Map(uploadedFiles.map((file, idx) => [fileSignature(file), idx]));
    const shouldExtract =
      samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned';
    let detectedAtomCount: number | null = null;

    try {
      for (const file of files) {
        const key = fileSignature(file);
        const existingIndex = prevFileMap.get(key);

        if (existingIndex !== undefined) {
          newTokens.push(uploadTokens[existingIndex]);
          continue;
        }

        if (file.size > 25 * 1024 * 1024) {
          setError(`File ${file.name} is too large (max 25MB)`);
          return;
        }

        const initResponse = await apiClient.initUpload();
        await apiClient.uploadFile(initResponse.upload_token, file);
        newTokens.push(initResponse.upload_token);

        const filename = file.name.toLowerCase();
        let cachedFileText: string | null = null;

        if (filename.endsWith('.sdf')) {
          cachedFileText = await file.text();
          const atoms = estimateAtomCountFromSdf(cachedFileText);
          if (atoms !== null && atoms > 0 && detectedAtomCount === null) {
            detectedAtomCount = atoms;
          }
        }

        if (shouldExtract && filename.endsWith('.sdf')) {
          try {
            const result = await apiClient.extractPharmacophore(file);
            setExtractedPharmacophores(result.pharmacophores);
            setSelectedPharmacophoreIndices(new Set());
            const ligandText = cachedFileText ?? (await file.text());
            setLigandContent(btoa(ligandText));
          } catch (err: any) {
            const detail = err?.response?.data?.detail ?? err.message;
            setError(`Failed to extract pharmacophore: ${detail}`);
          }
        }

        if (
          samplingMode === 'Protein+Pharmacophore-conditioned' &&
          (filename.endsWith('.pdb') || filename.endsWith('.cif'))
        ) {
          const content = await file.text();
          setProteinContent(btoa(content));
        }
      }

      setUploadedFiles(files);
      setUploadTokens(newTokens);

      const hasSdf = files.some((file) => file.name.toLowerCase().endsWith('.sdf'));
      if (hasSdf && detectedAtomCount !== null) {
        const derivedStd = Math.max(detectedAtomCount * ATOM_STD_MARGIN, 0.1);
        setAutoAtomCount(detectedAtomCount);
        setNLigAtomsMeanInput(detectedAtomCount.toString());
        setNLigAtomsStdInput(formatFloatInput(derivedStd));
      }
      if (!hasSdf) {
        setAutoAtomCount(null);
        setNLigAtomsMeanInput('');
        setNLigAtomsStdInput('');
        setExtractedPharmacophores([]);
        setSelectedPharmacophoreIndices(new Set());
        setLigandContent(null);
      }

      if (
        samplingMode === 'Protein+Pharmacophore-conditioned' &&
        !files.some((file) => file.name.toLowerCase().endsWith('.pdb') || file.name.toLowerCase().endsWith('.cif'))
      ) {
        setProteinContent(null);
      }
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? err.message;
      setError(`Upload failed: ${detail}`);
    }
  };

  const handleSubmit = async () => {
    setError(null);
    setIsSubmitting(true);

    try {
      // Validate pharmacophore selection
      if (
        (samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') &&
        extractedPharmacophores.length > 0 &&
        selectedPharmacophoreIndices.size === 0
      ) {
        setError('Please select at least one pharmacophore feature');
        setIsSubmitting(false);
        return;
      }

      // Convert pharmacophore to XYZ if needed
      let finalTokens = [...uploadTokens];
      if (
        (samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') &&
        extractedPharmacophores.length > 0 &&
        selectedPharmacophoreIndices.size > 0
      ) {
        const selectedList = Array.from(selectedPharmacophoreIndices);
        const xyzResult = await apiClient.pharmacophoreToXyz(
          extractedPharmacophores,
          selectedList,
          samplingMode !== 'Protein+Pharmacophore-conditioned'
        );

        // Upload XYZ file
        const initResponse = await apiClient.initUpload();
        const xyzBlob = new Blob([xyzResult.xyz_content], { type: 'text/plain' });
        const xyzFile = new File([xyzBlob], 'pharmacophore.xyz', { type: 'text/plain' });
        await apiClient.uploadFile(initResponse.upload_token, xyzFile);
        finalTokens.push(initResponse.upload_token);
      }

      // Validate protein-conditioned mode
      if (samplingMode === 'Protein-conditioned') {
        const hasProtein = uploadedFiles.some(f => f.name.toLowerCase().endsWith('.pdb') || f.name.toLowerCase().endsWith('.cif'));
        const hasLigand = uploadedFiles.some(f => f.name.toLowerCase().endsWith('.sdf'));
        if (!hasProtein || !hasLigand) {
          setError('Protein-conditioned mode requires both a protein file (PDB/CIF) and a reference ligand file (SDF)');
          setIsSubmitting(false);
          return;
        }
      }

      const parsedSamples = parseInt(nSamplesInput, 10);
      if (Number.isNaN(parsedSamples)) {
        setError('Number of samples must be a valid integer');
        setIsSubmitting(false);
        return;
      }
      if (parsedSamples < 1 || parsedSamples > 100) {
        setError('Number of samples must be between 1 and 100');
        setIsSubmitting(false);
        return;
      }

      const parsedSteps = parseInt(stepsInput, 10);
      if (Number.isNaN(parsedSteps)) {
        setError('Sampling steps must be a valid integer');
        setIsSubmitting(false);
        return;
      }
      if (parsedSteps < 10 || parsedSteps > 1000) {
        setError('Sampling steps must be between 10 and 1000');
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

      const meanProvided = nLigAtomsMeanInput.trim() !== '';
      const stdProvided = nLigAtomsStdInput.trim() !== '';
      if ((meanProvided && !stdProvided) || (!meanProvided && stdProvided)) {
        setError('Both mean and standard deviation must be provided together for atom count distribution');
        setIsSubmitting(false);
        return;
      }

      let parsedMean: number | null = null;
      let parsedStd: number | null = null;
      if (meanProvided && stdProvided) {
        parsedMean = Number(nLigAtomsMeanInput);
        parsedStd = Number(nLigAtomsStdInput);
        if (!Number.isFinite(parsedMean) || parsedMean < 4) {
          setError('Mean number of atoms must be at least 4');
          setIsSubmitting(false);
          return;
        }
        if (!Number.isFinite(parsedStd) || parsedStd <= 0) {
          setError('Standard deviation must be a positive number');
          setIsSubmitting(false);
          return;
        }
      }

      // Submit job
      const params: SamplingParams = {
        sampling_mode: samplingMode,
        seed: parsedSeed ?? null,
        n_samples: parsedSamples,
        steps: parsedSteps,
        n_lig_atoms_mean: parsedMean,
        n_lig_atoms_std: parsedStd,
      };

      const jobData = {
        params,
        uploads: finalTokens,
        job_id: useCustomJobId && customJobId.trim() ? customJobId.trim() : undefined,
      };

      const response = await apiClient.submitJob(jobData);
      
      resetForm();
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
            Sampling Mode
          </label>
          <select
            value={samplingMode}
            onChange={(e) => setSamplingMode(e.target.value as SamplingMode)}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          >
            <option value="Unconditional">Unconditional</option>
            <option value="Pharmacophore-conditioned">Pharmacophore-conditioned</option>
            <option value="Protein-conditioned">Protein-conditioned</option>
            <option value="Protein+Pharmacophore-conditioned">Protein+Pharmacophore-conditioned</option>
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
            max={2 ** 31 - 1}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Number of Samples
          </label>
          <input
            type="number"
            value={nSamplesInput}
            onChange={(e) => setNSamplesInput(e.target.value)}
            min={1}
            max={100}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Sampling Steps
          </label>
          <input
            type="number"
            value={stepsInput}
            onChange={(e) => setStepsInput(e.target.value)}
            min={10}
            max={1000}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

      </div>

      <div className="border-t border-slate-200/60 pt-6">
        <h3 className="text-base font-semibold text-slate-900 mb-3">Input Files</h3>
        {samplingMode === 'Unconditional' ? (
          <div className="p-4 bg-primary-50/70 rounded-xl text-sm text-primary-700 shadow-sm">
            No input files needed for unconditional generation
          </div>
        ) : (
          <FileUpload
            key={formResetKey}
            onFilesUploaded={handleFileUpload}
            acceptedTypes={
              samplingMode === 'Pharmacophore-conditioned'
                ? ['.xyz', '.sdf']
                : samplingMode === 'Protein-conditioned'
                ? ['.pdb', '.cif', '.sdf']
                : ['.pdb', '.cif', '.xyz', '.sdf']
            }
            maxFiles={3}
            maxSize={25 * 1024 * 1024}
          />
        )}
      </div>

      <div className="border-t border-slate-200/60 pt-6">
        <h4 className="text-sm font-semibold text-slate-700 mb-3">
          Atom Count Distribution (Optional)
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Mean Number of Atoms
            </label>
            <input
              type="number"
              value={nLigAtomsMeanInput}
              onChange={(e) => setNLigAtomsMeanInput(e.target.value)}
              min={4}
              step={1}
              placeholder="e.g., 25"
              className="w-full px-3 py-2 border border-slate-200 rounded-lg bg-white text-slate-900 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Standard Deviation
            </label>
            <input
              type="number"
              value={nLigAtomsStdInput}
              onChange={(e) => setNLigAtomsStdInput(e.target.value)}
              min={0.1}
              step={0.1}
              placeholder="e.g., 5.0"
              className="w-full px-3 py-2 border border-slate-200 rounded-lg bg-white text-slate-900 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
            />
          </div>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          {samplingMode === 'Unconditional'
            ? 'If you leave mean and standard deviation empty, the model will use dataset distribution for ligand sizes.'
            : autoAtomCount
            ? `Auto-filled using the ${autoAtomCount}-atom reference ligand. Std uses ${Math.round(
                ATOM_STD_MARGIN * 100
              )}% of # ligand atoms by default, and you can adjust either value.`
            : 'Upload a ligand SDF to auto-fill these values based on its atom count. You can still adjust them manually.'}
        </p>
      </div>

      {/* Pharmacophore Viewer */}
      {samplingMode !== 'Unconditional' &&
        samplingMode !== 'Protein-conditioned' &&
        extractedPharmacophores.length > 0 &&
        ligandContent && (
          <div className="border-t border-slate-200/60 pt-6">
            <h3 className="text-base font-semibold text-slate-900 mb-3">
              Pharmacophore Selection
            </h3>
            <div className="mb-3 text-sm text-slate-600">
              💡 Click spheres to select/deselect features
            </div>
            <PharmacophoreViewer
              ligandContent={ligandContent}
              pharmacophores={extractedPharmacophores.map((pharm, idx) => ({
                index: idx,
                type: pharm.type,
                x: pharm.position[0],
                y: pharm.position[1],
                z: pharm.position[2],
                color: getPharmacophoreColor(pharm.type),
                selected: selectedPharmacophoreIndices.has(idx),
              }))}
              selectedIndices={Array.from(selectedPharmacophoreIndices)}
              onSelectionChange={(indices) => setSelectedPharmacophoreIndices(new Set(indices))}
              proteinB64={proteinContent || undefined}
              proteinFormat={
                uploadedFiles.find(f => f.name.toLowerCase().endsWith('.pdb'))
                  ? 'pdb'
                  : uploadedFiles.find(f => f.name.toLowerCase().endsWith('.cif'))
                  ? 'cif'
                  : undefined
              }
            />
            {selectedPharmacophoreIndices.size > 0 ? (
              <div className="mt-3 p-3 bg-emerald-50/70 rounded-xl text-sm text-emerald-700 shadow-sm">
                ✓ {selectedPharmacophoreIndices.size} of {extractedPharmacophores.length} features selected
              </div>
            ) : (
              <div className="mt-3 p-3 bg-amber-50/70 rounded-xl text-sm text-amber-700 shadow-sm">
                ⚠️ No features selected. Click spheres in the viewer above.
              </div>
            )}
          </div>
        )}

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

      {error && (
        <div className="p-4 bg-red-50/70 rounded-xl text-sm text-red-700 shadow-sm">
          {error}
        </div>
      )}

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
            🚀 Run Sampling
          </>
        )}
      </button>
    </div>
  );
}

function getPharmacophoreColor(type: string): string {
  const colors: Record<string, string> = {
    Aromatic: 'purple',
    HydrogenDonor: '#f0f0f0',
    HydrogenAcceptor: 'orange',
    PositiveIon: 'blue',
    NegativeIon: 'red',
    Hydrophobic: 'green',
    Halogen: 'cyan',
  };
  return colors[type] || 'gray';
}

function estimateAtomCountFromSdf(content: string): number | null {
  const lines = content.split(/\r?\n/);
  const candidates: string[] = [];
  if (lines.length >= 4) {
    candidates.push(lines[3]);
  }
  const fallback = lines.find((line) => /^\s*\d+\s+\d+/.test(line));
  if (fallback) {
    candidates.push(fallback);
  }

  for (const line of candidates) {
    if (!line) continue;
    const firstField = line.trim().split(/\s+/)[0];
    const atomCount = parseInt(firstField, 10);
    if (Number.isFinite(atomCount) && atomCount > 0) {
      return atomCount;
    }
  }
  return null;
}

function formatFloatInput(value: number): string {
  if (Number.isInteger(value)) {
    return value.toString();
  }
  return value.toFixed(2).replace(/\.?0+$/, '');
}

