'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { Download } from 'lucide-react';
import type { SamplingMode } from '@/types';

interface MolecularViewerProps {
  jobId: string;
  filename: string;
  samplingMode: SamplingMode;
  pocketSelection?: any;
  inputFilesList?: { files: Array<{ filename: string; size: number; extension: string }> };
  prefetchedContent?: string;
}

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export function MolecularViewer({
  jobId,
  filename,
  samplingMode,
  pocketSelection,
  inputFilesList: propInputFiles,
  prefetchedContent,
}: MolecularViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const proteinModelRef = useRef<any>(null);
  const pharmShapesRef = useRef<any[]>([]);
  const ligandModelRef = useRef<any>(null);
  const pharmacophoreAtomsRef = useRef<Array<{ x: number; y: number; z: number; color: string }> | null>(null);
  const [isSceneReady, setIsSceneReady] = useState(false);
  const surfaceIdRef = useRef<any>(null);
  const lastJobIdRef = useRef<string | null>(null);
  const isFetchingRef = useRef<boolean>(false);
  const loadIdRef = useRef<number>(0);

  // Styling state - simple toggles
  const [showSticks, setShowSticks] = useState(false);
  const [showSurface, setShowSurface] = useState(true);
  const [showBackbone, setShowBackbone] = useState(true);
  const [hasProtein, setHasProtein] = useState(false);

  // 1. Load active result file content
  useEffect(() => {
    // If we have prefetched content from parent, use it immediately (zero delay)
    if (prefetchedContent) {
      setFileContent(prefetchedContent);
      setIsLoading(false);
      return;
    }

    const loadFile = async () => {
      setIsLoading(true);
      try {
        const blob = await apiClient.downloadFile(jobId, filename);
        const text = await blob.text();
        setFileContent(text);
      } catch (err) {
        console.error('Failed to load file:', err);
      } finally {
        setIsLoading(false);
      }
    };
    loadFile();
  }, [jobId, filename, prefetchedContent]);

  // 2. Initialize Viewer
  useEffect(() => {
    if (!containerRef.current || typeof window === 'undefined') return;

    const load3Dmol = async () => {
      if (!window.$3Dmol) {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js';
        script.async = true;
        document.head.appendChild(script);
        await new Promise((resolve) => { script.onload = resolve; });
      }

      if (!containerRef.current) return;

      if (!viewerRef.current) {
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: 'white'
        });
        viewer.setSlab(-1000, 1000);
        viewerRef.current = viewer;
      }
    };
    load3Dmol();
  }, []);

  // 3. Job-Level Data Loading (Protein & Pharmacophores)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const viewerAny = viewer as any;

    const mode = samplingMode || '';
    const needsProtein = [
      'Protein-conditioned',
      'Protein+Pharmacophore-conditioned',
      'Rigid Docking',
      'Rigid Docking + Pharmacophore'
    ].includes(mode) || mode.toLowerCase().includes('protein') || mode.toLowerCase().includes('docking');

    const needsPharmacophore = [
      'Pharmacophore-conditioned',
      'Protein+Pharmacophore-conditioned',
      'Rigid Docking + Pharmacophore'
    ].includes(mode) || mode.toLowerCase().includes('pharmacophore');

    const loadStructuralData = async () => {
      const currentLoadId = Math.random();
      loadIdRef.current = currentLoadId;
      console.log(`[MolecularViewer] loadStructuralData starting (${currentLoadId}) for jobId: ${jobId}`);

      // Surgical Reset: Only clear if Job ID changed
      if (lastJobIdRef.current !== jobId) {
        console.log(`[MolecularViewer] Job ID changed, clearing all models`);
        viewer.clear();
        lastJobIdRef.current = jobId;
        proteinModelRef.current = null;
        pharmShapesRef.current = [];
        ligandModelRef.current = null;
      } else {
        // Surgically remove protein and pharm if they exist to replace them
        if (proteinModelRef.current) {
          console.log(`[MolecularViewer] Removing existing protein model before reload`);
          try { viewer.removeModel(proteinModelRef.current); } catch (e) { }
          proteinModelRef.current = null;
        }
        if (pharmShapesRef.current.length > 0) {
          try {
            pharmShapesRef.current.forEach((s: any) => s.remove());
          } catch (e) { }
          pharmShapesRef.current = [];
        }
      }

      setIsSceneReady(true);
      setHasProtein(false);

      // Fetch structural data inside (using propInputFiles as cache)
      let inputFilesList = propInputFiles;
      try {
        if (!inputFilesList && (needsProtein || needsPharmacophore)) {
          console.log(`[MolecularViewer] Fetching input files list...`);
          inputFilesList = await apiClient.listInputFiles(jobId);
        }
      } catch (err) {
        console.error('Failed to list input files:', err);
      }

      const proteinPromise = (async () => {
        // If we don't think we need protein based on mode, but we have input files, 
        // check if there's a protein file anyway as a fallback.
        if (!inputFilesList) return null;

        const protFile = inputFilesList.files.find(f => f.extension === '.pdb' || f.extension === '.cif');
        if (!protFile) return null;

        // If we have a protein file and we either need it or it's a docking/protein mode (robust check)
        if (needsProtein || mode.toLowerCase().includes('dock') || mode.toLowerCase().includes('protein')) {
          try {
            const protBlob = await apiClient.downloadInputFile(jobId, protFile.filename);
            const protText = await protBlob.text();
            return { text: protText, format: protFile.extension === '.pdb' ? 'pdb' : 'cif' };
          } catch (err) { console.error('Protein load failed:', err); }
        }
        return null;
      })();

      const needsPharmacophoreInner = needsPharmacophore;

      const pharmacophorePromise = (async () => {
        if (!needsPharmacophoreInner || !inputFilesList) return null;
        try {
          const pharmFile = inputFilesList.files.find(f => ['.xyz', '.json'].includes(f.extension.toLowerCase())) ||
            inputFilesList.files.find(f => f.extension.toLowerCase() === '.sdf');
          if (pharmFile) {
            const blob = await apiClient.downloadInputFile(jobId, pharmFile.filename);
            const text = await blob.text();
            return { text, extension: pharmFile.extension.toLowerCase() };
          }
        } catch (err) { console.error('Pharmacophore load failed:', err); }
        return null;
      })();

      const [proteinData, pharmData] = await Promise.all([proteinPromise, pharmacophorePromise]);

      if (loadIdRef.current !== currentLoadId) {
        console.log(`[MolecularViewer] Load interrupted by newer request (${currentLoadId})`);
        return;
      }

      if (proteinData) {
        console.log(`[MolecularViewer] Adding protein model (${proteinData.format})`);
        const model = viewer.addModel(proteinData.text, proteinData.format);
        proteinModelRef.current = model;
        setHasProtein(true);
        // Apply initial style respecting state
        const style: any = {};
        if (showBackbone) style.cartoon = { color: 'lightblue' };
        if (showSticks) style.stick = { radius: 0.15, colorscheme: 'lightgreyCarbon' };
        console.log(`[MolecularViewer] Setting initial protein style:`, style);
        viewer.setStyle({ model: model.getID() }, style);
      }

      if (pharmData) {
        console.log(`[MolecularViewer] Adding pharmacophore shapes`);
        let atoms: any[] = [];
        if (pharmData.extension === '.xyz') atoms = parsePharmacophoreXyz(pharmData.text);
        else if (pharmData.extension === '.json') atoms = parsePharmacophoreJson(pharmData.text);
        else if (pharmData.extension === '.sdf') atoms = parsePharmacophoreSdf(pharmData.text);
        pharmShapesRef.current = addPharmacophoreAtoms(viewer, atoms);
      }

      viewer.render();
      isFetchingRef.current = false;
    };

    loadStructuralData();
  }, [jobId, samplingMode, propInputFiles]);


  // 4. Molecule SWITCHING (Ligand only)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !fileContent || !isSceneReady) return;

    // Preserve camera view across molecule swap
    const currentView = viewer.getView();
    const hasExistingLigand = ligandModelRef.current !== null;

    // Targeted removal of old ligand to keep protein and camera steady
    if (hasExistingLigand) {
      try { viewer.removeModel(ligandModelRef.current); } catch (e) { }
    }

    const fileFormat = filename.split('.').pop()?.toLowerCase() || 'sdf';
    const ligandModel = viewer.addModel(fileContent, fileFormat);
    ligandModelRef.current = ligandModel;

    // Apply ligand style
    viewer.setStyle({ model: ligandModel }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });

    if (!hasExistingLigand) {
      // First model load: zoom to it
      viewer.zoomTo({ model: ligandModel });
    } else {
      // Subsequent swaps: restore previous camera view exactly
      viewer.setView(currentView);
    }

    viewer.render();
  }, [fileContent, filename, isSceneReady]);

  // 5. Handle Style Toggles (Cartoon, Sticks) - Unified
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;

    if (proteinModelRef.current) {
      const model = proteinModelRef.current;
      const modelId = model.getID();
      console.log(`[MolecularViewer] Style Effect: backbone=${showBackbone}, sticks=${showSticks}, modelId=${modelId}`);

      const style: any = {};
      if (showBackbone) style.cartoon = { color: 'lightblue' };
      if (showSticks) style.stick = { radius: 0.15, colorscheme: 'lightgreyCarbon' };

      console.log(`[MolecularViewer] Calling setStyle with:`, style);
      viewer.setStyle({ model: modelId }, style);

      // Keep clipping fix
      viewer.setSlab(-1000, 1000);
    }

    viewer.render();
  }, [showSticks, showBackbone, isSceneReady, hasProtein]);

  // 6. Handle Surface Toggles (Global Protein Surface)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;

    const updateSurface = () => {
      try {
        viewer.removeAllSurfaces();
        if (showSurface && proteinModelRef.current) {
          viewer.addSurface(
            window.$3Dmol.VDW,
            { opacity: 0.6, colorscheme: 'whiteCarbon' },
            { model: proteinModelRef.current }
          );
          viewer.setSlab(-1000, 1000);
        }
        viewer.render();
      } catch (e) {
        console.warn('Surface sync failed:', e);
      }
    };

    updateSurface();
  }, [showSurface, isSceneReady, hasProtein]);

  const handleDownload = async () => {
    try {
      const blob = await apiClient.downloadFile(jobId, filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Download failed');
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors shadow-sm"
        >
          <Download className="w-4 h-4" />
          Download
        </button>
      </div>
      <div
        ref={containerRef}
        style={{ width: '100%', height: '500px', position: 'relative' }}
        className="three-d-viewer-container border border-slate-200 rounded-2xl overflow-hidden shadow-inner bg-slate-50"
      >
        {/* No spinner needed for seamless experience */}

        {/* Style Controls Panel - Persistent to prevent flickering */}
        {hasProtein && (
          <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-200/60 p-3 space-y-2 z-20 text-sm">
            <div className="font-semibold text-slate-700 text-xs mb-2">Style</div>


            {hasProtein && (
              <>
                <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors">
                  <input
                    type="checkbox"
                    checked={showSticks}
                    onChange={(e) => setShowSticks(e.target.checked)}
                    className="w-3.5 h-3.5 rounded"
                  />
                  Sticks
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors">
                  <input
                    type="checkbox"
                    checked={showSurface}
                    onChange={(e) => setShowSurface(e.target.checked)}
                    className="w-3.5 h-3.5 rounded"
                  />
                  Surface
                </label>
                <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors">
                  <input
                    type="checkbox"
                    checked={showBackbone}
                    onChange={(e) => setShowBackbone(e.target.checked)}
                    className="w-3.5 h-3.5 rounded"
                  />
                  Backbone (Cartoon)
                </label>
              </>
            )}
          </div>
        )}
      </div>

      {(samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned' || samplingMode === 'Rigid Docking + Pharmacophore') && (
        <div className="mt-4 p-4 bg-slate-50 border border-slate-200 rounded-xl">
          <h4 className="text-sm font-semibold text-slate-700 mb-3">Pharmacophore Color Legend</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-purple-500" /> Aromatic</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-gray-200 border border-slate-300" /> Hydrogen Donor</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-orange-500" /> Hydrogen Acceptor</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500" /> Positive Ion</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-500" /> Negative Ion</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-green-500" /> Hydrophobic</div>
            <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-cyan-400" /> Halogen</div>
          </div>
        </div>
      )}
    </div>
  );
}

// Helper Functions
// Helper Functions

const PHARMACOPHORE_COLORS: Record<string, string> = {
  // Pharmit/Standard Types
  Aromatic: 'purple',
  HydrogenDonor: '#f0f0f0',
  HydrogenAcceptor: 'orange',
  PositiveIon: 'blue',
  NegativeIon: 'red',
  Hydrophobic: 'green',
  Halogen: 'cyan',

  // Element/Symbol Fallbacks
  P: 'purple',       // Aromatic
  S: '#f0f0f0',      // Donor (e.g. Sulfur/Donor)
  F: 'orange',       // Acceptor (e.g. Fluorine/Acceptor)
  N: 'blue',         // Nitrogen/Positive
  O: 'red',          // Oxygen/Negative
  C: 'green',        // Carbon/Hydrophobic
  Cl: 'cyan',        // Chlorine/Halogen

  // Generic Fallbacks
  Donor: '#f0f0f0',
  Acceptor: 'orange',
  PosIon: 'blue',
  NegIon: 'red',
  Zn: 'grey',        // Metal
  Mg: 'grey'
};

function parsePharmacophoreXyz(xyzContent: string) {
  const lines = xyzContent.split('\n');
  const atomLines = lines.slice(2);
  // Optional element mapping if using obscure Pharmit codes
  const elementToType: Record<string, string> = {
    P: 'Aromatic', S: 'HydrogenDonor', F: 'HydrogenAcceptor',
    N: 'PositiveIon', O: 'NegativeIon', C: 'Hydrophobic', Cl: 'Halogen'
  };

  return atomLines.map(line => line.trim()).filter(Boolean).map(line => {
    const parts = line.split(/\s+/);
    if (parts.length < 4) return null;
    const rawType = parts[0];
    const mappedType = elementToType[rawType] || rawType;
    const color = PHARMACOPHORE_COLORS[mappedType] || PHARMACOPHORE_COLORS[rawType] || 'gray';

    return {
      x: parseFloat(parts[1]),
      y: parseFloat(parts[2]),
      z: parseFloat(parts[3]),
      color
    };
  }).filter(Boolean) as Array<{ x: number; y: number; z: number; color: string }>;
}

function parsePharmacophoreJson(jsonContent: string) {
  try {
    const data = JSON.parse(jsonContent);
    // Handle array of points or object with points/features
    const features = Array.isArray(data) ? data : (data.points || data.features || []);

    if (!Array.isArray(features)) return [];

    return features.map((f: any) => {
      // Handle coordinate formats
      let x = 0, y = 0, z = 0;
      if (Array.isArray(f.position)) { [x, y, z] = f.position; }
      else if (typeof f.x === 'number') { x = f.x; y = f.y; z = f.z; }
      else if (f.center && Array.isArray(f.center)) { [x, y, z] = f.center; }

      const type = f.type || f.name || 'Unknown';
      const color = f.color || PHARMACOPHORE_COLORS[type] || 'gray';

      return { x, y, z, color };
    }).filter(p => !isNaN(p.x)); // simple filter
  } catch (e) {
    console.warn('Failed to parse pharmacophore JSON', e);
    return [];
  }
}

function parsePharmacophoreSdf(sdfContent: string) {
  const lines = sdfContent.split('\n');
  const atoms: Array<{ x: number; y: number; z: number; color: string }> = [];

  for (const line of lines) {
    // Look for atom lines (e.g. "   1.234   2.345   3.456 C ...")
    // Loose parsing: find 3 floats followed by letters
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 4) {
      const x = parseFloat(parts[0]);
      const y = parseFloat(parts[1]);
      const z = parseFloat(parts[2]);
      const sym = parts[3];

      if (!isNaN(x) && !isNaN(y) && !isNaN(z) && /^[A-Za-z]+$/.test(sym)) {
        // Map symbol to color
        const color = PHARMACOPHORE_COLORS[sym] || 'gray';
        atoms.push({ x, y, z, color });
      }
    }
  }
  return atoms;
}

// Helper wrapper to manage a pharmacophore shape's lifecycle
class PharmacophoreShape {
  private viewer: any;
  private atom: { x: number; y: number; z: number; color: string };
  private currentShape: any = null;
  private isSelected: boolean = false;

  constructor(viewer: any, atom: { x: number; y: number; z: number; color: string }) {
    this.viewer = viewer;
    this.atom = atom;
    this.render();
  }

  toggle() {
    this.isSelected = !this.isSelected;
    this.render();
  }

  render() {
    // Remove existing shape if any
    if (this.currentShape) {
      try {
        this.viewer.removeShape(this.currentShape);
      } catch (e) {
        console.warn('Failed to remove shape', e);
      }
    }

    const sphereOptions: any = {
      center: { x: this.atom.x, y: this.atom.y, z: this.atom.z },
      radius: 1.0, // Match typical size
      color: this.atom.color,
      clickable: true,
      callback: () => this.toggle(),
    };

    if (this.isSelected) {
      sphereOptions.alpha = 1.0;
      sphereOptions.wireframe = false;
    } else {
      sphereOptions.wireframe = true;
      sphereOptions.linewidth = 1.5;
    }

    this.currentShape = this.viewer.addSphere(sphereOptions);

    // Apply always-on-top style if selected
    if (this.isSelected && this.currentShape && this.currentShape.material) {
      try {
        const materials = Array.isArray(this.currentShape.material)
          ? this.currentShape.material
          : [this.currentShape.material];

        materials.forEach((mat: any) => {
          if (mat) {
            mat.depthTest = false;
            mat.depthWrite = false;
            mat.transparent = true;
            mat.opacity = 1.0;
          }
        });
      } catch (e) {
        console.warn('Failed to apply depth hack', e);
      }
    }

    this.viewer.render();
  }

  remove() {
    if (this.currentShape) {
      try {
        this.viewer.removeShape(this.currentShape);
      } catch (e) { }
    }
  }
}

function addPharmacophoreAtoms(viewer: any, atoms: Array<{ x: number; y: number; z: number; color: string }>) {
  return atoms.map(atom => new PharmacophoreShape(viewer, atom));
}
