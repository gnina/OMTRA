'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import {
    apiClient,
    encodeBase64Unicode
} from '@/lib/api-client';
import type { SamplingMode, MetricsOptions, BricsFragment } from '@/types';
import { DEFAULT_METRICS_OPTIONS } from '@/types';
import { FileUpload } from './FileUpload';
import { FixedStructurePanel } from './FixedStructurePanel';
import { MetricsOptionsForm } from './MetricsOptionsForm';
import { LoadFromJobPicker } from './LoadFromJobPicker';
import { PharmacophoreSelector } from './PharmacophoreSelector';
import { useFixedAtomSelection } from '@/hooks/useFixedAtomSelection';
import { AlertCircle, CheckCircle2, Loader2, Play, X, Settings2, Info, RefreshCw, ChevronDown, ChevronRight } from 'lucide-react';

interface JobSubmissionFormProps {
    initialSamplingMode?: SamplingMode;
    onSamplingModeChange?: (mode: SamplingMode) => void;
    onJobSubmitted: (jobId: string) => void;
    onProteinContentChange?: (content: string | null) => void;
    onProteinFormatChange?: (format: 'pdb' | 'cif' | undefined) => void;
    onLigandContentChange?: (content: string | null) => void;
    onPharmacophoresChange?: (pharmacophores: Array<{
        type: string; position: [number, number, number]
    }>) => void;
    onPharmacophoreSelectionChange?: (indices: number[]) => void;
    pharmacophores?: Array<{
        type: string; position: [number, number, number]
    }>;
    selectedPharmacophoreIndices?: number[];
    // Pocket Selection props (new)
    onPocketsDetected?: (pockets: any[]) => void;
    detectedPockets: any[];
    selectedPocketId: string | null;
    onPocketSelect: (id: string | null) => void;
    hiddenPocketIds?: string[];
    onHiddenPocketsChange?: (ids: string[]) => void;
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
    pharmacophoreTolerance: string;
    onPharmacophoreToleranceChange: (val: string) => void;
    bricsFragments: BricsFragment[];
    setBricsFragments: (frags: BricsFragment[]) => void;
    bricsRawSdf: string | null;
    setBricsRawSdf: (sdf: string | null) => void;
    fixStructureExpanded: boolean;
    setFixStructureExpanded: (v: boolean) => void;
    fixedSelection: ReturnType<typeof useFixedAtomSelection>;
    totalAtomCount: number;
}

export function JobSubmissionForm({
    initialSamplingMode = 'Unconditional',
    onSamplingModeChange,
    onJobSubmitted,
    onProteinContentChange,
    onProteinFormatChange,
    onLigandContentChange,
    onPharmacophoresChange,
    onPharmacophoreSelectionChange,
    onPocketsDetected,
    detectedPockets = [],
    selectedPocketId,
    onPocketSelect,
    hiddenPocketIds = [],
    onHiddenPocketsChange,
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
    pharmacophoreTolerance,
    onPharmacophoreToleranceChange,
    pharmacophores = [],
    selectedPharmacophoreIndices = [],
    bricsFragments,
    setBricsFragments,
    bricsRawSdf,
    setBricsRawSdf,
    fixStructureExpanded,
    setFixStructureExpanded,
    fixedSelection,
    totalAtomCount,
}: JobSubmissionFormProps) {
    const [apiConnected, setApiConnected] = useState<boolean | null>(null);
    const [samplingMode, setSamplingModeState] = useState<SamplingMode>(initialSamplingMode);

    const setSamplingMode = useCallback((mode: SamplingMode) => {
        setSamplingModeState(mode);
        onSamplingModeChange?.(mode);
    }, [onSamplingModeChange]);

    const [seedInput, setSeedInput] = useState('42');
    const [nSamplesInput, setNSamplesInput] = useState('10');
    const [stepsInput, setStepsInput] = useState('100');
    const [nLigAtomsMeanInput, setNLigAtomsMeanInput] = useState('');
    const [nLigAtomsStdInput, setNLigAtomsStdInput] = useState('');
    const [autoAtomCount, setAutoAtomCount] = useState<number | null>(null);
    const [useCustomJobId, setUseCustomJobId] = useState(false);
    const [customJobId, setCustomJobId] = useState('');
    // pharmacophoreTolerance moved to props

    // File state
    const [proteinFile, setProteinFile] = useState<File | null>(null);
    const [proteinToken, setProteinToken] = useState<string | null>(null);

    const [pharmacophoreFile, setPharmacophoreFile] = useState<File | null>(null);
    const [pharmacophoreToken, setPharmacophoreToken] = useState<string | null>(null);

    const [ligandFile, setLigandFile] = useState<File | null>(null);
    const [ligandToken, setLigandToken] = useState<string | null>(null);
    const [refLigandFile, setRefLigandFile] = useState<File | null>(null);

    const [uploadedFiles, setUploadedFiles] = useState<File[]>([]); // Deprecated
    const [uploadTokens, setUploadTokens] = useState<string[]>([]); // Deprecated

    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isDetectingPockets, setIsDetectingPockets] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [formResetKey, setFormResetKey] = useState(0);
    const [metricsOptions, setMetricsOptions] = useState<MetricsOptions>({ ...DEFAULT_METRICS_OPTIONS });


    // Pharmacophore state
    // Remove local pharmacophores state, use from props
    const [ligandContent, setLigandContent] = useState<string | null>(null); // For pharmacophore viewer
    // Remove local proteinContent state, use onProteinContentChange exclusively or similar logic to DockingForm
    const [proteinFormat, setProteinFormat] = useState<'pdb' | 'cif' | undefined>(undefined);
    const ATOM_STD_MARGIN = 0.15;
    const prevSamplingModeRef = useRef<SamplingMode>(samplingMode);

    // State for content viewing
    const [proteinContent, setProteinContent] = useState<string | null>(null);

    // BRICS fragment selection state (fragments/selectedIds/rawSdf lifted to parent)
    const [bricsLoading, setBricsLoading] = useState(false);

    const togglePocketHidden = useCallback((pocketId: string) => {
        const isHidden = hiddenPocketIds.includes(pocketId);
        const nextHidden = isHidden
            ? hiddenPocketIds.filter((id) => id !== pocketId)
            : [...hiddenPocketIds, pocketId];
        onHiddenPocketsChange?.(nextHidden);
        if (!isHidden && selectedPocketId === pocketId) {
            onPocketSelect(null);
        }
    }, [hiddenPocketIds, onHiddenPocketsChange, onPocketSelect, selectedPocketId]);

    const resetForm = useCallback(() => {
        setSeedInput('42');
        setNSamplesInput('10');
        setStepsInput('100');
        setNLigAtomsMeanInput('');
        setNLigAtomsStdInput('');
        setAutoAtomCount(null);
        onPharmacophoreToleranceChange('0.0');

        setProteinFile(null);
        setProteinToken(null);
        setPharmacophoreFile(null);
        setPharmacophoreToken(null);
        setLigandFile(null);
        setLigandToken(null);

        setPharmacophoreToken(null);
        onPharmacophoresChange?.([]);
        onPharmacophoreSelectionChange?.([]);
        setLigandContent(null);
        setProteinContent(null);
        setCustomJobId('');
        setUseCustomJobId(false);
        setError(null);
        setFormResetKey((key) => key + 1);

        // Notify parent to clear shared visual state
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
        setBricsFragments([]);
        fixedSelection.resetSelection();
        setBricsRawSdf(null);
        setFixStructureExpanded(false);
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
        setRefLigandToken,
        setRefLigandFileName,
        setPocketSelectionMethod,
        setManualCenter,
        setBboxLength,
        setLigandCenter,
        onPharmacophoreToleranceChange,
    ]);

    useEffect(() => {
        // Check API health
        const checkHealth = async () => {
            try {
                const isHealthy = await apiClient.healthCheck();
                setApiConnected(isHealthy);
            } catch (error) {
                setApiConnected(false);
            }
        };
        checkHealth();
    }, []);

    useEffect(() => {
        // Reset all form state when switching sampling modes
        if (prevSamplingModeRef.current !== samplingMode) {
            resetForm();
            prevSamplingModeRef.current = samplingMode;
        }
    }, [samplingMode, resetForm]);

    const uploadFileToApi = async (file: File): Promise<string> => {
        const initResponse = await apiClient.initUpload();
        await apiClient.uploadFile(initResponse.upload_token, file);
        return initResponse.upload_token;
    };

    const handleProteinUpload = async (file: File) => {
        setProteinFile(file);
        setProteinToken(null);
        setError(null);

        const filename = file.name.toLowerCase();
        try {
            const content = await file.text();
            const proteinB64 = encodeBase64Unicode(content);
            setProteinContent(proteinB64);
            onProteinContentChange?.(proteinB64);
            const format = filename.endsWith('.pdb') ? 'pdb' : 'cif';
            setProteinFormat(format);
            onProteinFormatChange?.(format);

            // Upload
            try {
                const token = await uploadFileToApi(file);
                setProteinToken(token);
            } catch (e) {
                console.error("Upload failed", e);
                setError("Failed to upload protein file");
                return;
            }

            // Detect pockets
            if (onPocketsDetected) {
                setIsDetectingPockets(true);
                try {
                    const result = await apiClient.detectPockets(file);
                    onPocketsDetected(result.pockets);
                } catch (err) {
                    console.error("Pocket detection failed", err);
                } finally {
                    setIsDetectingPockets(false);
                }
            }
        } catch (err: any) {
            setError(`Failed to process protein file: ${err.message}`);
        }
    };

    const handlePharmUpload = async (file: File) => {
        setPharmacophoreFile(file);
        setPharmacophoreToken(null);
        setError(null);

        try {
            // Upload
            const token = await uploadFileToApi(file);
            setPharmacophoreToken(token);

            // Extract pharmacophores using backend API for all formats (SDF, XYZ, JSON)
            try {
                const result = await apiClient.extractPharmacophore(file);
                onPharmacophoresChange?.(result.pharmacophores);
                onPharmacophoreSelectionChange?.([]);

                if (file.name.toLowerCase().endsWith('.sdf')) {
                    const content = await file.text();
                    const ligandB64 = encodeBase64Unicode(content);
                    setLigandContent(ligandB64);
                    onLigandContentChange?.(ligandB64);
                } else {
                    setLigandContent(null);
                    onLigandContentChange?.(null);
                }
            } catch (e) {
                console.error("Pharm extraction failed", e);
                setError(`Failed to extract pharmacophores: ${e instanceof Error ? e.message : 'Unknown error'}`);
            }
        } catch (err: any) {
            setError(`Failed to process pharmacophore file: ${err.message}`);
        }
    };

    const handleRefLigandUpload = async (file: File) => {
        setError(null);
        setBricsFragments([]);
        fixedSelection.resetSelection();
        setBricsRawSdf(null);
        setRefLigandFile(file);
        try {
            const token = await uploadFileToApi(file);
            setRefLigandToken?.(token);

            const content = await file.text();
            const base64Content = encodeBase64Unicode(content);
            setRefLigandContent?.(base64Content);
            setRefLigandFileName?.(file.name);
            setBricsRawSdf(content);

            const center = calculateCenterFromSdf(content);
            if (center) setLigandCenter(center);

            // Auto-fill atom count distribution from pocket ligand
            const atomCount = estimateAtomCountFromSdf(content);
            if (atomCount && atomCount > 0) {
                setNLigAtomsMeanInput(atomCount.toString());
                const suggestedStd = Math.min(atomCount * 0.2, 5.0);
                setNLigAtomsStdInput(suggestedStd.toFixed(1));
            }

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
            console.error('Ref ligand upload failed:', err);
            setError(`Failed to upload reference ligand: ${err.message}`);
        }
    };

    const handleLigandUpload = async (file: File) => {
        setRefLigandContent?.(null);
        setRefLigandToken?.(null);
        setRefLigandFileName?.(null);
        setRefLigandFile(null);
        setError(null);

        try {
            const token = await uploadFileToApi(file);
            setLigandToken(token);

            const content = await file.text();
            setLigandContent(encodeBase64Unicode(content)); // For viewer
            onLigandContentChange?.(encodeBase64Unicode(content));

            // Estimate atoms
            const atoms = estimateAtomCountFromSdf(content);
            if (atoms) {
                const derivedStd = Math.max(atoms * ATOM_STD_MARGIN, 0.1);
                setAutoAtomCount(atoms);
                setNLigAtomsMeanInput(atoms.toString());
                setNLigAtomsStdInput(formatFloatInput(derivedStd));
            }
        } catch (err: any) {
            setError(`Failed to process ligand file: ${err.message}`);
        }
    };

    const handleSubmit = async (e?: React.FormEvent | React.MouseEvent) => {
        e?.preventDefault();
        setError(null);

        // Snapshot fixed atoms before any await (pharmacophore upload, etc.)
        const capturedFixedIndices =
            samplingMode === 'Protein-conditioned' && fixedSelection.fixedCount > 0
                ? fixedSelection.getFixedAtomIndicesForSubmit()
                : undefined;

        setIsSubmitting(true);

        try {
            // Mode-specific validation
            if (samplingMode === 'Protein-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') {
                if (!proteinFile) throw new Error("Please upload a protein file in Step 1.");

                const hasPocket = (pocketSelectionMethod === 'detected' && selectedPocketId) ||
                    (pocketSelectionMethod === 'manual' && manualCenter.x && manualCenter.y && manualCenter.z) ||
                    (pocketSelectionMethod === 'ligand' && refLigandFile);

                if (!hasPocket) {
                    throw new Error("Please define a binding site in Step 2 (Select a pocket, upload a reference ligand, or enter coordinates).");
                }
            }

            if (samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') {
                const hasPharm = (pharmacophoreFile && !pharmacophores.length) || (pharmacophores.length > 0 && selectedPharmacophoreIndices.length > 0);
                if (!hasPharm) {
                    if (pharmacophores.length > 0) {
                        throw new Error("Please select at least one pharmacophore feature in the 3D viewer.");
                    } else {
                        throw new Error("Please upload a pharmacophore file.");
                    }
                }
            }

            const parsedSamples = parseInt(nSamplesInput, 10);
            if (Number.isNaN(parsedSamples) || parsedSamples < 1 || parsedSamples > 20) throw new Error("Samples must be 1-20");

            const parsedSteps = parseInt(stepsInput, 10);
            if (Number.isNaN(parsedSteps) || parsedSteps < 50 || parsedSteps > 300) throw new Error("Steps must be 50-300");

            let parsedMean: number | null = null;
            let parsedStd: number | null = null;
            if (nLigAtomsMeanInput && nLigAtomsStdInput) {
                parsedMean = Number(nLigAtomsMeanInput);
                parsedStd = Number(nLigAtomsStdInput);
                if (parsedMean < 4 || parsedMean > 100) throw new Error("Mean number of atoms must be between 4 and 100");
                if (parsedStd < 0) throw new Error("Standard deviation must be non-negative");
            }

            let finalTokens: string[] = [];
            let freshRefLigandToken: string | null = null;

            if (proteinFile) {
                const token = await uploadFileToApi(proteinFile);
                setProteinToken(token);
                finalTokens.push(token);
            }
            if (ligandFile) {
                const token = await uploadFileToApi(ligandFile);
                setLigandToken(token);
                finalTokens.push(token);
            }
            if (refLigandFile) {
                freshRefLigandToken = await uploadFileToApi(refLigandFile);
                setRefLigandToken?.(freshRefLigandToken);
                finalTokens.push(freshRefLigandToken);
            }

            // Handle pharmacophore - ALWAYS convert to XYZ with selection if pharmacophores were extracted
            if (pharmacophores.length > 0) {
                // If selection exists, use it; otherwise use ALL pharmacophores
                const selectedList = selectedPharmacophoreIndices.length > 0
                    ? Array.from(selectedPharmacophoreIndices)
                    : pharmacophores.map((_, i) => i);

                const xyzResult = await apiClient.pharmacophoreToXyz(
                    pharmacophores,
                    selectedList,
                    samplingMode !== 'Protein+Pharmacophore-conditioned'
                );
                // Upload generated XYZ with only selected features
                const initResponse = await apiClient.initUpload();
                const xyzBlob = new Blob([xyzResult.xyz_content], { type: 'text/plain' });
                const xyzFile = new File([xyzBlob], 'selected_pharmacophore.xyz', { type: 'text/plain' });
                await apiClient.uploadFile(initResponse.upload_token, xyzFile);
                finalTokens.push(initResponse.upload_token);
            } else if (pharmacophoreFile) {
                const token = await uploadFileToApi(pharmacophoreFile);
                setPharmacophoreToken(token);
                finalTokens.push(token);
            }

            const isProteinInvolving =
                samplingMode === 'Protein-conditioned' ||
                samplingMode === 'Protein+Pharmacophore-conditioned';


            let pocketSelectionParam = undefined;

            if (isProteinInvolving && pocketSelectionMethod === 'detected' && selectedPocketId) {
                const pocket = detectedPockets.find(p => p.id === selectedPocketId);
                const alphaSphereCenters = pocket?.alpha_sphere_centers;
                if (pocket && alphaSphereCenters.length > 0) {
                    pocketSelectionParam = {
                        type: 'coords',
                        value: alphaSphereCenters,
                    };
                } else if (pocket) {
                    pocketSelectionParam = {
                        type: 'center',
                        value: pocket.center,
                        bbox_length: parseFloat(bboxLength)
                    };
                }
            } else if (isProteinInvolving && pocketSelectionMethod === 'manual') {
                const center: [number, number, number] = [
                    parseFloat(manualCenter.x),
                    parseFloat(manualCenter.y),
                    parseFloat(manualCenter.z)
                ];
                if (center.some(isNaN)) throw new Error("Manual coordinates must be valid numbers");

                pocketSelectionParam = {
                    type: 'center',
                    value: center,
                    bbox_length: parseFloat(bboxLength)
                };
            } else if (isProteinInvolving && pocketSelectionMethod === 'ligand') {
                if (!refLigandFile) throw new Error("Reference ligand file is required for this pocket definition. Please upload one in Step 2.");

                pocketSelectionParam = {
                    type: 'file',
                    value: freshRefLigandToken,
                };
            }


            const fixedAtomIndicesForJob = capturedFixedIndices?.length ? capturedFixedIndices : undefined;

            // Submit job
            const params: any = {
                sampling_mode: samplingMode,
                seed: seedInput ? Number(seedInput) : null,
                n_samples: parsedSamples,
                steps: parsedSteps,
                n_lig_atoms_mean: parsedMean,
                n_lig_atoms_std: parsedStd,
                pocket_selection: pocketSelectionParam,
                pharmacophore_tolerance: 0.0,
                metrics_options: metricsOptions,
            };
            if (samplingMode === 'Protein-conditioned' && fixedAtomIndicesForJob) {
                params.fixed_atom_indices = fixedAtomIndicesForJob;
            }

            const jobData = {
                params,
                uploads: finalTokens,
                job_id: useCustomJobId && customJobId.trim() ? customJobId.trim() : undefined,
            };

            const response = await apiClient.submitJob(jobData);

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
                        Sampling Mode
                    </label>
                    <select
                        value={samplingMode}
                        onChange={(e) => {
                            const newMode = e.target.value as SamplingMode;
                            setSamplingMode(newMode);
                            // Clear viewer when switching to Unconditional
                            if (newMode === 'Unconditional') {
                                onProteinContentChange?.(null);
                                onProteinFormatChange?.(undefined);
                                onLigandContentChange?.(null);
                            }
                        }}
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
                            // Clamp to max 20
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
                            // Clamp to max 300
                            if (!isNaN(num) && num <= 300) {
                                setStepsInput(val);
                            }
                        }}
                        min={50}
                        max={300}
                        className="w-full px-3 py-2.5 border border-slate-200 rounded-xl bg-white text-slate-900 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors shadow-sm"
                    />

                </div>

                {/* Pharmacophore Tolerance Slider - Removed (not implemented yet) */}

            </div>

            <div className="border-t border-slate-200/60 pt-6 space-y-6">
                <h3 className="text-base font-semibold text-slate-900 mb-3">Input Files</h3>

                {samplingMode === 'Unconditional' && (
                    <div className="p-4 bg-primary-50/70 rounded-xl text-sm text-primary-700 shadow-sm">
                        No input files needed for unconditional generation
                    </div>
                )}

                {/* 1. Protein Upload (for Protein-conditioned modes) */}
                {(samplingMode === 'Protein-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') && (
                    <div>
                        <h4 className="text-sm font-semibold text-slate-700 mb-2">1. Upload Protein</h4>
                        <p className="text-xs text-slate-500 mb-2">Upload PDB or CIF file defining the target.</p>
                        {!proteinFile ? (
                            <>
                                <FileUpload
                                    key={`prot-${samplingMode}-${formResetKey}`}
                                    onFilesUploaded={(files) => {
                                        if (files.length > 0) handleProteinUpload(files[0]);
                                        else {
                                            setProteinFile(null);
                                            setProteinToken(null);
                                            onProteinContentChange?.(null);
                                        }
                                    }}
                                    acceptedTypes={['.pdb', '.cif']}
                                    maxFiles={1}
                                    maxSize={25 * 1024 * 1024}
                                />
                                {!proteinFile && <LoadFromJobPicker
                                    key={`load-prot-${samplingMode}-${formResetKey}`}
                                    acceptedExtensions={['.pdb', '.cif']}
                                    onFileLoaded={handleProteinUpload}
                                    disabled={isSubmitting}
                                />}
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
                            <div className="mt-2 text-xs text-blue-600 flex items-center">
                                <Loader2 className="w-3 h-3 animate-spin mr-1" /> Detecting pockets...
                            </div>
                        )}
                    </div>
                )}

                {/* 2. Pocket Selection (for Protein-conditioned modes) */}
                {(proteinFile || refLigandFile) && (samplingMode === 'Protein-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned') && (
                    <div className="animate-in fade-in slide-in-from-top-4 duration-500 space-y-4">
                        <h4 className="text-sm font-semibold text-slate-700 mb-2">2. Binding Site Definition</h4>
                        <p className="text-xs text-slate-500 mb-2">Recommend deselecting protein surface to view pocket.</p>
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
                            <div className="space-y-2">
                                <p className="text-xs text-slate-500 italic mb-2">
                                    Note: Defines the pocket as residues 8Å from alpha sphere centers.
                                </p>
                                {detectedPockets && detectedPockets.length > 0 ? (
                                    <div className="p-3 border border-emerald-200 bg-emerald-50/50 rounded-xl">
                                        <p className="text-xs text-emerald-800 font-medium flex items-center gap-1">
                                            <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden />
                                            Pockets detected.
                                        </p>
                                        <div className="mt-2 space-y-1.5">
                                            {detectedPockets.map((pocket: any, idx: number) => {
                                                const isHidden = hiddenPocketIds.includes(pocket.id);
                                                const isSelected = selectedPocketId === pocket.id;
                                                return (
                                                    <div
                                                        key={pocket.id}
                                                        className={`flex items-center justify-between gap-2 rounded-lg border px-2 py-1.5 text-xs ${isSelected
                                                            ? 'border-primary-300 bg-white text-primary-700'
                                                            : 'border-emerald-100 bg-white/70 text-slate-600'
                                                            } ${isHidden ? 'opacity-60' : ''}`}
                                                    >
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                if (!isHidden) onPocketSelect(isSelected ? null : pocket.id);
                                                            }}
                                                            className="min-w-0 flex-1 text-left"
                                                        >
                                                            <span className="font-semibold">Pocket {idx + 1}</span>
                                                            <span className="ml-1 font-mono text-[10px] text-slate-500">
                                                                score {typeof pocket.score === 'number' ? pocket.score.toFixed(3) : 'n/a'} · volume {typeof pocket.volume === 'number' ? pocket.volume.toFixed(1) : 'n/a'}
                                                            </span>
                                                        </button>
                                                        <button
                                                            type="button"
                                                            onClick={() => togglePocketHidden(pocket.id)}
                                                            className={`rounded-md px-2 py-1 font-semibold ${isHidden
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
                                            <div className="mt-1">
                                                <p className="text-xs text-slate-600">
                                                    Pocket <strong>{selectedPocketId}</strong> selected.
                                                </p>
                                                {detectedPockets.find(p => p.id === selectedPocketId) && (
                                                    <p className="text-xs text-slate-600">
                                                        Score: {typeof detectedPockets.find(p => p.id === selectedPocketId)?.score === 'number'
                                                            ? detectedPockets.find(p => p.id === selectedPocketId)?.score.toFixed(3)
                                                            : 'n/a'} · Volume: {typeof detectedPockets.find(p => p.id === selectedPocketId)?.volume === 'number'
                                                                ? detectedPockets.find(p => p.id === selectedPocketId)?.volume.toFixed(1)
                                                                : 'n/a'}
                                                    </p>
                                                )}
                                            </div>
                                        ) : (
                                            <p className="text-xs text-amber-600 animate-pulse font-medium mt-1">
                                                Select a pocket in the 3D viewer.
                                            </p>
                                        )}
                                    </div>
                                ) : (
                                    <div className="p-3 bg-slate-100 rounded-xl text-xs text-slate-500">
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
                                            key={`ref-ligand-${formResetKey}`}
                                            onFilesUploaded={(files) => {
                                                if (files.length > 0) handleRefLigandUpload(files[0]);
                                                else {
                                                    setRefLigandFile(null);
                                                    setRefLigandContent?.(null);
                                                    setRefLigandToken?.(null);
                                                    setRefLigandFileName?.(null);
                                                    setLigandCenter(null);
                                                    setBricsFragments([]);
                                                    fixedSelection.resetSelection();
                                                    setBricsRawSdf(null);
                                                    setFixStructureExpanded(false);
                                                }
                                            }}
                                            acceptedTypes={['.sdf']}
                                            maxFiles={1}
                                            maxSize={5 * 1024 * 1024}
                                        />
                                        {!refLigandFile && <LoadFromJobPicker
                                            key={`load-ref-ligand-${formResetKey}`}
                                            acceptedExtensions={['.sdf']}
                                            onFileLoaded={handleRefLigandUpload}
                                            disabled={isSubmitting}
                                        />}
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
                                                setBricsFragments([]);
                                                fixedSelection.resetSelection();
                                                setBricsRawSdf(null);
                                                setFixStructureExpanded(false);
                                            }}
                                            className="absolute top-2 right-2 p-1 text-blue-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                                            title="Remove reference ligand"
                                        >
                                            <X className="w-3.5 h-3.5" />
                                        </button>
                                        <div className="flex items-center gap-2.5">
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
                                            <div className="mt-2 pt-2 border-t border-blue-100/50">
                                                <p className="text-[9px] font-bold text-blue-400 uppercase tracking-tighter mb-0.5">Center</p>
                                                <p className="text-[10px] font-mono text-blue-700 bg-blue-100/30 px-1.5 py-0.5 rounded inline-block">
                                                    [{ligandCenter[0].toFixed(2)}, {ligandCenter[1].toFixed(2)}, {ligandCenter[2].toFixed(2)}]
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {/* BRICS Fragment Selection (only for Protein-conditioned, not Protein+Pharmacophore) */}
                                {samplingMode === 'Protein-conditioned' && refLigandFile && (bricsLoading || bricsFragments.length > 0) && (
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

                        {pocketSelectionMethod === 'manual' && (
                            <div className="space-y-3 p-3 border border-slate-200 rounded-xl bg-white shadow-sm">
                                <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Manual Center Coordinates</p>
                                <div className="grid grid-cols-3 gap-2">
                                    <div>
                                        <label className="block text-[10px] font-bold text-slate-400 mb-0.5">X</label>
                                        <input
                                            type="number"
                                            value={manualCenter.x}
                                            onChange={(e) => setManualCenter({ ...manualCenter, x: e.target.value })}
                                            className="w-full px-2 py-1 border border-slate-200 rounded-lg text-xs shadow-sm bg-slate-50 transition-all hover:bg-white focus:bg-white focus:ring-1 focus:ring-primary-500"
                                            step="0.1"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-[10px] font-bold text-slate-400 mb-0.5">Y</label>
                                        <input
                                            type="number"
                                            value={manualCenter.y}
                                            onChange={(e) => setManualCenter({ ...manualCenter, y: e.target.value })}
                                            className="w-full px-2 py-1 border border-slate-200 rounded-lg text-xs shadow-sm bg-slate-50 transition-all hover:bg-white focus:bg-white focus:ring-1 focus:ring-primary-500"
                                            step="0.1"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-[10px] font-bold text-slate-400 mb-0.5">Z</label>
                                        <input
                                            type="number"
                                            value={manualCenter.z}
                                            onChange={(e) => setManualCenter({ ...manualCenter, z: e.target.value })}
                                            className="w-full px-2 py-1 border border-slate-200 rounded-lg text-xs shadow-sm bg-slate-50 transition-all hover:bg-white focus:bg-white focus:ring-1 focus:ring-primary-500"
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

                {/* 3. Pharmacophore Upload (for Pharmacophore modes) */}
                {(samplingMode === 'Pharmacophore-conditioned' || (samplingMode === 'Protein+Pharmacophore-conditioned' && (proteinFile || pharmacophoreFile))) && (
                    <div className="animate-in fade-in slide-in-from-top-4 duration-500 delay-150">
                        <h4 className="text-sm font-semibold text-slate-700 mb-2">
                            {samplingMode === 'Protein+Pharmacophore-conditioned' ? '3.' : '1.'} Upload Pharmacophore
                        </h4>
                        <p className="text-xs text-slate-500 mb-2">Upload XYZ, JSON, or SDF (to extract features).</p>
                        {!pharmacophoreFile ? (
                            <>
                                <FileUpload
                                    key={`pharm-${samplingMode}-${formResetKey}`}
                                    onFilesUploaded={(files) => {
                                        if (files.length > 0) handlePharmUpload(files[0]);
                                        else {
                                            setPharmacophoreFile(null);
                                            setPharmacophoreToken(null);
                                            onPharmacophoresChange?.([]);
                                            onLigandContentChange?.(null);
                                            setLigandContent(null);
                                        }
                                    }}
                                    acceptedTypes={['.xyz', '.json', '.sdf']}
                                    maxFiles={1}
                                    maxSize={5 * 1024 * 1024}
                                />
                                <LoadFromJobPicker
                                    key={`load-pharm-${samplingMode}-${formResetKey}`}
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
                                        setLigandContent(null);
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
                    </div>
                )}


            </div>

            {
                samplingMode !== 'Unconditional' &&
                samplingMode !== 'Protein-conditioned' &&
                pharmacophores.length > 0 &&
                ligandContent && (
                    <div className="border-t border-slate-200/60 pt-6">
                        <h3 className="text-base font-semibold text-slate-900 mb-3">
                            Pharmacophore Selection
                        </h3>
                        <PharmacophoreSelector
                            pharmacophores={pharmacophores}
                            selectedIndices={selectedPharmacophoreIndices}
                            onSelectionChange={(indices) => onPharmacophoreSelectionChange?.(indices)}
                        />
                        {selectedPharmacophoreIndices.length > 0 ? (
                            <div className="mt-3 p-3 bg-emerald-50/70 rounded-xl text-sm text-emerald-700 shadow-sm border border-emerald-100 flex items-center gap-2">
                                <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden />
                                {selectedPharmacophoreIndices.length} of {pharmacophores.length} features selected
                            </div>
                        ) : (
                            <div className="mt-3 p-3 bg-amber-50/70 rounded-xl text-sm text-amber-700 shadow-sm border border-amber-100 flex items-center gap-2">
                                <AlertCircle className="w-4 h-4 shrink-0" aria-hidden />
                                No features selected. Select at least one feature.
                            </div>
                        )}
                    </div>
                )
            }

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
                            max={100}
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
                        ? 'If left empty, the model will use the dataset distribution.'
                        : autoAtomCount
                            ? `Auto-filled using the ${autoAtomCount}-atom reference ligand. Std uses ${Math.round(
                                ATOM_STD_MARGIN * 100
                            )}% of # ligand atoms by default, and you can adjust either value.`
                            : 'Upload a ligand SDF to auto-fill these values based on its atom count. You can still adjust them manually.'}
                </p>
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
                samplingMode={samplingMode}
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
                            Run Sampling
                        </>
                    )}
                </button>
            </div>
        </div >
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
        const fields = line.trim().split(/\s+/);
        if (fields.length >= 1) {
            const atomCount = parseInt(fields[0], 10);
            if (Number.isFinite(atomCount) && atomCount > 0) {
                return atomCount;
            }
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


function calculateCenterFromSdf(content: string): [number, number, number] | null {
    const lines = content.split('\n');
    let xSum = 0, ySum = 0, zSum = 0, count = 0;
    let atomBlock = false;

    for (const line of lines) {
        if (line.includes('V2000') || line.includes('V3000') || line.includes('MOL')) {
            atomBlock = true;
            continue;
        }
        if (atomBlock && line.trim().split(/\s+/).length >= 4) {
            const parts = line.trim().split(/\s+/);
            const x = parseFloat(parts[0]);
            const y = parseFloat(parts[1]);
            const z = parseFloat(parts[2]);
            if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
                xSum += x; ySum += y; zSum += z; count++;
            }
        }
        if (line.includes('M  END') || line.includes('$$$$')) break;
    }

    if (count > 0) {
        return [xSum / count, ySum / count, zSum / count];
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
            return [xFallback / countFallback, yFallback / countFallback, zFallback / countFallback];
        }
    }
    return null;
}


