'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { useFixedAtomSelection } from '@/hooks/useFixedAtomSelection';
import { JobSubmissionForm } from '@/components/JobSubmissionForm';
import { DockingForm } from '@/components/DockingForm';
import { JobList } from '@/components/JobList';
import { JobViewer } from '@/components/JobViewer';
import { HelpTab } from '@/components/HelpTab';
import { CentralSelectionViewer } from '@/components/CentralSelectionViewer';
import { AlertTriangle } from 'lucide-react';

import type { PocketInfo, BricsFragment } from '@/types';

export default function HomePage() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'submit' | 'jobs' | 'help'>('submit');
  const [workflowMode, setWorkflowMode] = useState<'denovo' | 'docking'>('denovo');
  const [denovoTask, setDenovoTask] = useState<any>('Unconditional');
  const [dockingTask, setDockingTask] = useState<any>('Rigid Docking');

  // Resizable sidebar state
  const [sidebarWidth, setSidebarWidth] = useState(33.33); // percentage
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  // Shared state for form and viewer
  const [proteinContent, setProteinContent] = useState<string | null>(null);
  const [proteinFormat, setProteinFormat] = useState<'pdb' | 'cif' | undefined>(undefined);
  const [ligandContent, setLigandContent] = useState<string | null>(null);
  const [extractedPharmacophores, setExtractedPharmacophores] = useState<
    Array<{ type: string; position: [number, number, number] }>
  >([]);

  // Docking-specific state
  const [detectedPockets, setDetectedPockets] = useState<PocketInfo[]>([]);
  const [selectedPocketId, setSelectedPocketId] = useState<string | null>(null);
  const [hiddenPocketIds, setHiddenPocketIds] = useState<string[]>([]);
  const [selectedPharmacophoreIndices, setSelectedPharmacophoreIndices] = useState<number[]>([]);

  // Pocket selection state (lifted)
  const [pocketSelectionMethod, setPocketSelectionMethod] = useState<'detected' | 'ligand' | 'manual'>('ligand');
  const [manualCenter, setManualCenter] = useState({ x: '0', y: '0', z: '0' });
  const [bboxLength, setBboxLength] = useState('15.0');
  const [ligandCenter, setLigandCenter] = useState<[number, number, number] | null>(null);
  const [refLigandContent, setRefLigandContent] = useState<string | null>(null);
  const [refLigandToken, setRefLigandToken] = useState<string | null>(null);
  const [refLigandFileName, setRefLigandFileName] = useState<string | null>(null);
  const [pharmacophoreTolerance, setPharmacophoreTolerance] = useState('0.0');

  // BRICS fragment state (lifted from forms)
  const [bricsFragments, setBricsFragments] = useState<BricsFragment[]>([]);
  const [bricsRawSdf, setBricsRawSdf] = useState<string | null>(null);
  const [fixStructureExpanded, setFixStructureExpanded] = useState(false);

  const totalAtomCount = useMemo(() => {
    const indices = new Set<number>();
    for (const frag of bricsFragments) {
      for (const i of frag.atom_indices) indices.add(i);
    }
    return indices.size;
  }, [bricsFragments]);

  const fixedSelection = useFixedAtomSelection(bricsFragments, totalAtomCount);

  // UI State

  // Load sidebar width from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('omtra-sidebar-width');
    if (saved) {
      const width = parseFloat(saved);
      if (width >= 20 && width <= 50) {
        setSidebarWidth(width);
      }
    }
  }, []);

  // Save sidebar width to localStorage
  useEffect(() => {
    localStorage.setItem('omtra-sidebar-width', sidebarWidth.toString());
  }, [sidebarWidth]);

  // Handle sidebar resize
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !sidebarRef.current) return;

      const containerWidth = window.innerWidth;
      const newWidth = (e.clientX / containerWidth) * 100;

      // Clamp between 20% and 50%
      if (newWidth >= 20 && newWidth <= 50) {
        setSidebarWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing) {
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Scroll to top when a job is selected
  useEffect(() => {
    if (selectedJobId) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [selectedJobId]);

  const handlePocketsDetected = (pockets: PocketInfo[]) => {
    setDetectedPockets(pockets);
    setHiddenPocketIds([]);
    setSelectedPocketId(null);
  };

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-slate-200/60 bg-white/80 backdrop-blur-md shadow-sm">
        <nav className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8">
          <div className="flex h-20 items-center gap-8">
            {/* Logo - Left Side */}
            <div className="flex items-center gap-3">
              <img src={`${process.env.NODE_ENV === 'production' ? '/omtra' : ''}/logo.png`} alt="OMTRA Logo" className="h-14 w-auto" />
              <div className="flex flex-col">
                <span className="text-3xl font-bold text-slate-900">OMTRA</span>
                <span className="text-sm text-slate-500 font-medium -mt-1">Generative Structure-Based Drug Design</span>
              </div>
            </div>
            {/* Navigation - After Logo */}
            <div className="flex items-center gap-6">
              <button
                onClick={() => {
                  setActiveTab('submit');
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`text-sm font-medium transition-colors ${activeTab === 'submit'
                  ? 'text-primary-600 font-semibold'
                  : 'text-slate-700 hover:text-primary-600'
                  }`}
              >
                Submit Job
              </button>
              <button
                onClick={() => {
                  if (activeTab === 'jobs') {
                    setSelectedJobId(null);
                  }
                  setActiveTab('jobs');
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`text-sm font-medium transition-colors ${activeTab === 'jobs'
                  ? 'text-primary-600 font-semibold'
                  : 'text-slate-700 hover:text-primary-600'
                  }`}
              >
                Jobs
              </button>
              <button
                onClick={() => {
                  setActiveTab('help');
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`text-sm font-medium transition-colors ${activeTab === 'help'
                  ? 'text-primary-600 font-semibold'
                  : 'text-slate-700 hover:text-primary-600'
                  }`}
              >
                Help
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Disclaimer Banner */}
      <div className="bg-amber-50 border-b border-amber-200/60">
        <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0 mt-0.5">
              <AlertTriangle className="w-5 h-5 text-amber-600" aria-hidden />
            </div>
            <div className="flex-1 text-sm text-amber-900">
              <p className="font-medium mb-1">Disclaimer</p>
              <p className="text-amber-800/90">
                This is a proof of concept server for demonstration purposes. All results are public, and jobs are automatically removed after 48 hours. For issues, feature requests, or to contribute, please visit our{' '}
                <a
                  href="https://github.com/gnina/OMTRA"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-semibold underline hover:text-amber-900 transition-colors"
                >
                  GitHub repository
                </a>
                {' '}and open an issue.
              </p>
            </div>
          </div>
        </div>
      </div>

      <main className="flex-1">
        {/* Submit Job Tab - Persisted */}
        <div className={activeTab === 'submit' ? 'block h-full' : 'hidden'}>
          <section className="flex h-[calc(100vh-8rem)] relative">
            {/* Left Sidebar - Form Controls */}
            <div
              ref={sidebarRef}
              className="border-r border-slate-200 bg-white flex-shrink-0 relative"
              style={{ width: `${sidebarWidth}%` }}
            >
              <div className="h-full overflow-y-auto">
                <div className="p-6">
                  {/* Workflow Mode Selector */}
                  <div className="mb-6">
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                      Workflow
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => {
                          if (workflowMode !== 'denovo') {
                            setWorkflowMode('denovo');
                            // Reset shared state
                            setProteinContent(null);
                            setProteinFormat(undefined);
                            setLigandContent(null);
                            setExtractedPharmacophores([]);
                            setDetectedPockets([]);
                            setSelectedPocketId(null);
                            setHiddenPocketIds([]);
                            setSelectedPharmacophoreIndices([]);
                            setPocketSelectionMethod('ligand');
                            setManualCenter({ x: '0', y: '0', z: '0' });
                            setBboxLength('15.0');
                            setLigandCenter(null);
                            setRefLigandContent(null);
                            setRefLigandToken(null);
                            setBricsFragments([]);
                            fixedSelection.resetSelection();
                            setBricsRawSdf(null);
                            setFixStructureExpanded(false);
                          }
                        }}
                        className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all ${workflowMode === 'denovo'
                          ? 'bg-primary-600 text-white shadow-md'
                          : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                          }`}
                      >
                        De Novo Design
                      </button>
                      <button
                        onClick={() => {
                          if (workflowMode !== 'docking') {
                            setWorkflowMode('docking');
                            // Reset shared state
                            setProteinContent(null);
                            setProteinFormat(undefined);
                            setLigandContent(null);
                            setExtractedPharmacophores([]);
                            setDetectedPockets([]);
                            setSelectedPocketId(null);
                            setHiddenPocketIds([]);
                            setSelectedPharmacophoreIndices([]);
                            setPocketSelectionMethod('ligand');
                            setManualCenter({ x: '0', y: '0', z: '0' });
                            setBboxLength('15.0');
                            setLigandCenter(null);
                            setRefLigandContent(null);
                            setRefLigandToken(null);
                            setBricsFragments([]);
                            fixedSelection.resetSelection();
                            setBricsRawSdf(null);
                            setFixStructureExpanded(false);
                          }
                        }}
                        className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all ${workflowMode === 'docking'
                          ? 'bg-primary-600 text-white shadow-md'
                          : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                          }`}
                      >
                        Docking
                      </button>
                    </div>
                  </div>

                  {/* Forms - Persisted via simple conditional or CSS? Using conditional for mode switch is standard, but if user wants sidebar consistency across modes, CSS is better. Given the shared protein state but distinct form inputs, unmounting removes inputs. User specifically complained about switching TABS (Submit vs Jobs). Switching MODES usually implies a reset. I will stick to persisting TABS only for now, unless directed otherwise. */}
                  {workflowMode === 'denovo' ? (
                    <JobSubmissionForm
                      initialSamplingMode={denovoTask}
                      onSamplingModeChange={setDenovoTask}
                      onJobSubmitted={setSelectedJobId}
                      onProteinContentChange={setProteinContent}
                      onProteinFormatChange={setProteinFormat}
                      onLigandContentChange={setLigandContent}
                      onPharmacophoresChange={setExtractedPharmacophores}
                      pharmacophores={extractedPharmacophores}
                      selectedPharmacophoreIndices={selectedPharmacophoreIndices}
                      onPharmacophoreSelectionChange={setSelectedPharmacophoreIndices}
                      onPocketsDetected={handlePocketsDetected}
                      detectedPockets={detectedPockets}
                      selectedPocketId={selectedPocketId}
                      onPocketSelect={setSelectedPocketId}
                      hiddenPocketIds={hiddenPocketIds}
                      onHiddenPocketsChange={setHiddenPocketIds}
                      pocketSelectionMethod={pocketSelectionMethod}
                      setPocketSelectionMethod={setPocketSelectionMethod}
                      manualCenter={manualCenter}
                      setManualCenter={setManualCenter}
                      bboxLength={bboxLength}
                      setBboxLength={setBboxLength}
                      ligandCenter={ligandCenter}
                      setLigandCenter={setLigandCenter}
                      refLigandContent={refLigandContent}
                      setRefLigandContent={setRefLigandContent}
                      refLigandToken={refLigandToken}
                      setRefLigandToken={setRefLigandToken}
                      refLigandFileName={refLigandFileName}
                      setRefLigandFileName={setRefLigandFileName}
                      pharmacophoreTolerance={pharmacophoreTolerance}
                      onPharmacophoreToleranceChange={setPharmacophoreTolerance}
                      bricsFragments={bricsFragments}
                      setBricsFragments={setBricsFragments}
                      bricsRawSdf={bricsRawSdf}
                      setBricsRawSdf={setBricsRawSdf}
                      fixStructureExpanded={fixStructureExpanded}
                      setFixStructureExpanded={setFixStructureExpanded}
                      fixedSelection={fixedSelection}
                      totalAtomCount={totalAtomCount}
                    />
                  ) : (
                    <DockingForm
                      initialDockingMode={dockingTask}
                      onDockingModeChange={setDockingTask}
                      onJobSubmitted={setSelectedJobId}
                      onProteinContentChange={setProteinContent}
                      onProteinFormatChange={setProteinFormat}
                      onLigandContentChange={setLigandContent}
                      onPocketsDetected={handlePocketsDetected}
                      detectedPockets={detectedPockets}
                      selectedPocketId={selectedPocketId}
                      onPocketSelect={setSelectedPocketId}
                      hiddenPocketIds={hiddenPocketIds}
                      onHiddenPocketsChange={setHiddenPocketIds}
                      onPharmacophoresChange={setExtractedPharmacophores}
                      pharmacophores={extractedPharmacophores}
                      selectedPharmacophoreIndices={selectedPharmacophoreIndices}
                      onPharmacophoreSelectionChange={setSelectedPharmacophoreIndices}
                      pocketSelectionMethod={pocketSelectionMethod}
                      setPocketSelectionMethod={setPocketSelectionMethod}
                      manualCenter={manualCenter}
                      setManualCenter={setManualCenter}
                      bboxLength={bboxLength}
                      setBboxLength={setBboxLength}
                      ligandCenter={ligandCenter}
                      setLigandCenter={setLigandCenter}
                      refLigandContent={refLigandContent}
                      setRefLigandContent={setRefLigandContent}
                      refLigandToken={refLigandToken}
                      setRefLigandToken={setRefLigandToken}
                      refLigandFileName={refLigandFileName}
                      setRefLigandFileName={setRefLigandFileName}
                      bricsFragments={bricsFragments}
                      setBricsFragments={setBricsFragments}
                      bricsRawSdf={bricsRawSdf}
                      setBricsRawSdf={setBricsRawSdf}
                      fixStructureExpanded={fixStructureExpanded}
                      setFixStructureExpanded={setFixStructureExpanded}
                      fixedSelection={fixedSelection}
                      totalAtomCount={totalAtomCount}
                    />
                  )}
                </div>
              </div>

              {/* Drag Handle - only the center tab is interactive */}
              <div className="absolute top-0 right-0 bottom-0 w-1">
                <div
                  className="absolute top-1/2 right-0 -translate-y-1/2 w-3 h-12 cursor-col-resize flex items-center justify-center -translate-x-[3px] z-10"
                  onMouseDown={() => setIsResizing(true)}
                >
                  <div className="w-1.5 h-12 bg-slate-300 hover:bg-primary-500 rounded-full transition-colors" />
                </div>
              </div>
            </div>

            {/* Center - Viewer for Selection */}
            <div className="flex-1 bg-slate-50 p-6 overflow-hidden">
              <div className="h-full bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6">
                <CentralSelectionViewer
                  visible={activeTab === 'submit'}
                  proteinContent={proteinContent || undefined}
                  proteinFormat={proteinFormat}
                  ligandContent={ligandContent || undefined}
                  pharmacophores={extractedPharmacophores.map((pharm, idx) => ({
                    index: idx,
                    type: pharm.type,
                    x: pharm.position[0],
                    y: pharm.position[1],
                    z: pharm.position[2],
                    color: getPharmacophoreColor(pharm.type),
                    selected: selectedPharmacophoreIndices.includes(idx),
                  }))}
                  selectedPharmacophoreIndices={selectedPharmacophoreIndices}
                  onPharmacophoreSelectionChange={setSelectedPharmacophoreIndices}
                  detectedPockets={detectedPockets}
                  selectedPocketId={selectedPocketId}
                  onPocketSelect={setSelectedPocketId}
                  hiddenPocketIds={hiddenPocketIds}
                  pocketSelectionMethod={pocketSelectionMethod}
                  manualCenter={manualCenter}
                  bboxLength={bboxLength}
                  ligandCenter={ligandCenter}
                  refLigandContent={refLigandContent || undefined}
                  pharmacophoreTolerance={parseFloat(pharmacophoreTolerance)}
                  bricsFragments={bricsFragments}
                  selectedFragmentIds={fixedSelection.selectedFragmentIds}
                  onFragmentSelectionChange={fixedSelection.setSelectedFragmentIds}
                  bricsRawSdf={bricsRawSdf || undefined}
                  fixStructureActive={!!bricsRawSdf && bricsFragments.length > 1 && ((workflowMode === 'denovo' && denovoTask === 'Protein-conditioned') || (workflowMode === 'docking' && dockingTask === 'Rigid Docking'))}
                  fixStructureMode={fixedSelection.mode}
                  selectionAction={fixedSelection.selectionAction}
                  selectedAtomIndices={fixedSelection.fixedAtomIndicesForSubmit}
                  onAtomClick={(idx) => fixedSelection.applyActionToAtoms([idx])}
                  onAtomsInBox={(indices) => fixedSelection.applyActionToAtoms(indices)}
                  onFragmentsInBox={(ids) => fixedSelection.toggleFragmentIdsInBox(ids)}
                  onToggleFragmentByAtom={fixedSelection.toggleFragmentByAtom}
                />
              </div>
            </div>
          </section>
        </div>

        {/* Jobs Tab */}
        <div className={activeTab === 'jobs' ? 'block' : 'hidden'}>
          <section className="py-8">
            <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8">
              <div className="bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6">
                {selectedJobId ? (
                  <JobViewer jobId={selectedJobId} onBack={() => setSelectedJobId(null)} />
                ) : (
                  <JobList onJobSelect={setSelectedJobId} />
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Help Tab */}
        <div className={activeTab === 'help' ? 'block' : 'hidden'}>
          <section className="py-8">
            <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 sm:px-6 lg:px-8">
              <div id="help" className="bg-white rounded-2xl shadow-[0_1px_3px_0_rgba(0,0,0,0.1),0_1px_2px_-1px_rgba(0,0,0,0.1)] p-6">
                <HelpTab />
              </div>
            </div>
          </section>
        </div>
      </main >

      {/* Footer */}
      < footer className="bg-slate-900 text-white mt-12" >
        <div className="mx-auto max-w-[95%] xl:max-w-[1400px] px-4 py-8 sm:px-6 lg:px-8">
          <div className="border-t border-slate-800 pt-6">
            <p className="text-sm text-slate-400 text-center">
              © {new Date().getFullYear()} OMTRA.{' '}
              <a
                href="https://github.com/gnina/OMTRA"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-300 hover:text-white underline transition-colors"
              >
                View on GitHub
              </a>
              {' '}•{' '}
              <a
                href="https://github.com/gnina/OMTRA/issues"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-300 hover:text-white underline transition-colors"
              >
                Report Issues
              </a>
            </p>
          </div>
        </div>
      </footer >
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
