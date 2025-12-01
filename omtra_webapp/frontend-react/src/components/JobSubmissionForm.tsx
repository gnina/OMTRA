'use client';

import { useState, useEffect } from 'react';
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
  const [seed, setSeed] = useState<number>(42);
  const [nSamples, setNSamples] = useState<number>(10);
  const [steps, setSteps] = useState<number>(100);
  const [nLigAtomsMean, setNLigAtomsMean] = useState<number | null>(null);
  const [nLigAtomsStd, setNLigAtomsStd] = useState<number | null>(null);
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

  const resetForm = () => {
    setSamplingMode('Unconditional');
    setSeed(42);
    setNSamples(10);
    setSteps(100);
    setNLigAtomsMean(null);
    setNLigAtomsStd(null);
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
    // Reset pharmacophore state when switching modes
    if (samplingMode === 'Unconditional' || samplingMode === 'Protein-conditioned') {
      setExtractedPharmacophores([]);
      setSelectedPharmacophoreIndices(new Set());
      setLigandContent(null);
      setProteinContent(null);
    }
  }, [samplingMode]);

  const fileSignature = (file: File) => `${file.name}:${file.size}:${file.lastModified}`;

  const handleFileUpload = async (files: File[]) => {
    setError(null);
    const newTokens: string[] = [];
    const prevFileMap = new Map(uploadedFiles.map((file, idx) => [fileSignature(file), idx]));
    const shouldExtract =
      samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned';

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

        if (shouldExtract && filename.endsWith('.sdf')) {
          try {
            const result = await apiClient.extractPharmacophore(file);
            setExtractedPharmacophores(result.pharmacophores);
            setSelectedPharmacophoreIndices(new Set());
            const content = await file.text();
            setLigandContent(btoa(content));
          } catch (err: any) {
            const detail = err?.response?.data?.detail ?? err.message;
            setError(`Failed to extract pharmacophore: ${detail}`);
          }
        }

        if (samplingMode === 'Protein+Pharmacophore-conditioned' && (filename.endsWith('.pdb') || filename.endsWith('.cif'))) {
          const content = await file.text();
          setProteinContent(btoa(content));
        }
      }

      setUploadedFiles(files);
      setUploadTokens(newTokens);

      const hasSdf = files.some((file) => file.name.toLowerCase().endsWith('.sdf'));
      if (!hasSdf) {
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

      // Validate atoms parameters
      if ((nLigAtomsMean !== null && nLigAtomsStd === null) || (nLigAtomsMean === null && nLigAtomsStd !== null)) {
        setError('Both mean and standard deviation must be provided together for atom count distribution');
        setIsSubmitting(false);
        return;
      }

      // Submit job
      const params: SamplingParams = {
        sampling_mode: samplingMode,
        seed: seed || null,
        n_samples: nSamples,
        steps: steps,
        n_lig_atoms_mean: nLigAtomsMean || null,
        n_lig_atoms_std: nLigAtomsStd || null,
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
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
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
            value={nSamples}
            onChange={(e) => setNSamples(parseInt(e.target.value) || 1)}
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
            value={steps}
            onChange={(e) => setSteps(parseInt(e.target.value) || 10)}
            min={10}
            max={1000}
            className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
          />
        </div>

        <div className="border-t border-slate-200/60 pt-4">
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
                value={nLigAtomsMean ?? ''}
                onChange={(e) => setNLigAtomsMean(e.target.value ? parseFloat(e.target.value) : null)}
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
                value={nLigAtomsStd ?? ''}
                onChange={(e) => setNLigAtomsStd(e.target.value ? parseFloat(e.target.value) : null)}
                min={0.1}
                step={0.1}
                placeholder="e.g., 5.0"
                className="w-full px-3 py-2 border border-slate-200 rounded-lg bg-white text-slate-900 text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
              />
            </div>
          </div>
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

