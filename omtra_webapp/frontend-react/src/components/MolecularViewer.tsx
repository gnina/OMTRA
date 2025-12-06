'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { Download } from 'lucide-react';
import type { SamplingMode } from '@/types';

interface MolecularViewerProps {
  jobId: string;
  filename: string;
  samplingMode: SamplingMode;
}

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export function MolecularViewer({ jobId, filename, samplingMode }: MolecularViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const proteinDataRef = useRef<{ content: string; format: string } | null>(null);
  const pharmacophoreAtomsRef = useRef<Array<{ x: number; y: number; z: number; color: string }> | null>(null);
  const hasBuiltSceneRef = useRef(false);

  useEffect(() => {
    // Load file content
    const loadFile = async () => {
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
  }, [jobId, filename]);

  useEffect(() => {
    if (!containerRef.current || typeof window === 'undefined') return;

    const load3Dmol = async () => {
      if (!window.$3Dmol) {
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js';
        script.async = true;
        document.head.appendChild(script);
        await new Promise((resolve) => {
          script.onload = resolve;
        });
      }

      if (!containerRef.current) return;

      // Initialize viewer only once
      if (!viewerRef.current) {
        console.log('Creating new 3Dmol viewer');
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          defaultcolors: window.$3Dmol.rasmolElementColors,
        });
        // Set background color to white
        viewer.setBackgroundColor(0xffffff);
        viewerRef.current = viewer;
        (viewer as any)._initialized = true;
        console.log('Viewer created:', viewer);
      }

      const viewer = viewerRef.current;
      if (!viewer || !fileContent) {
        console.error('Viewer or fileContent missing:', { viewer: !!viewer, fileContent: !!fileContent });
        return;
      }

      // Determine whether we need protein / pharmacophore data
      const needsProtein =
        samplingMode === 'Protein-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned';
      const needsPharmacophore =
        samplingMode === 'Pharmacophore-conditioned' || samplingMode === 'Protein+Pharmacophore-conditioned';
      // Ensure protein/pharmacophore data is loaded once
      const ensureProteinData = async () => {
        if (!needsProtein || proteinDataRef.current) return;
        const inputFiles = await apiClient.listInputFiles(jobId);
        const protFile = inputFiles.files.find(
          (f) => f.extension === '.pdb' || f.extension === '.cif'
        );
        if (!protFile) return;
        const protBlob = await apiClient.downloadInputFile(jobId, protFile.filename);
        const protContent = await protBlob.text();
        const protFormat = protFile.extension === '.pdb' ? 'pdb' : 'cif';
        proteinDataRef.current = { content: protContent, format: protFormat };
      };

      const ensurePharmacophoreData = async () => {
        if (!needsPharmacophore || pharmacophoreAtomsRef.current) return;
        const inputFiles = await apiClient.listInputFiles(jobId);
        const xyzFile = inputFiles.files.find((f) => f.extension === '.xyz');
        if (!xyzFile) return;
        const xyzBlob = await apiClient.downloadInputFile(jobId, xyzFile.filename);
        const xyzContent = await xyzBlob.text();
        pharmacophoreAtomsRef.current = parsePharmacophoreXyz(xyzContent);
      };

      await Promise.all([
        ensureProteinData().catch((err) => console.error('Failed to ensure protein data:', err)),
        ensurePharmacophoreData().catch((err) => console.error('Failed to ensure pharmacophore data:', err)),
      ]);

      const hasProtein = !!proteinDataRef.current;
      const hasPharmacophore = !!pharmacophoreAtomsRef.current;
      const isFirstLoad = !hasBuiltSceneRef.current;

      console.log('MolecularViewer load:', { isFirstLoad, hasProtein, hasPharmacophore, filename });

      const viewerAny = viewer as any;

      const safeGetCamera = () => {
        if (!hasBuiltSceneRef.current) return null;
        try {
          return viewer.getView();
        } catch (err) {
          console.warn('Failed to capture camera view:', err);
          return null;
        }
      };

      const safeSetCamera = (camera: number[] | null, ligandModel: any) => {
        try {
          if (camera) {
            viewer.setView(camera);
          } else if (ligandModel) {
            viewer.zoomTo({ model: ligandModel });
          } else {
            viewer.zoomTo();
          }
        } catch (err) {
          console.warn('Failed to restore camera view:', err);
        }
      };

      const removeExistingLigandModel = () => {
        const existingLigand = viewerAny._ligandModel;
        if (!existingLigand) return;

        try {
          if (typeof viewer.removeModel === 'function') {
            viewer.removeModel(existingLigand);
          } else if (typeof existingLigand.remove === 'function') {
            existingLigand.remove();
          } else if (typeof viewer.removeAllModels === 'function') {
            viewer.removeAllModels();
            viewerAny._proteinModel = undefined;
            viewerAny._pharmacophoreShapes = undefined;
          } else {
            viewer.clear();
            viewerAny._proteinModel = undefined;
            viewerAny._pharmacophoreShapes = undefined;
          }
        } catch (err) {
          console.warn('Failed to remove existing ligand model:', err);
          viewer.clear();
          viewerAny._proteinModel = undefined;
          viewerAny._pharmacophoreShapes = undefined;
          viewerAny._proteinSurface = undefined;
          hasBuiltSceneRef.current = false;
        } finally {
          viewerAny._ligandModel = undefined;
        }
      };

      const ensureProteinModel = () => {
        if (!proteinDataRef.current || viewerAny._proteinModel) return;
        const proteinModel = viewer.addModel(
          proteinDataRef.current.content,
          proteinDataRef.current.format
        );
        viewer.setStyle({ model: proteinModel }, { cartoon: { color: 'lightblue' } });
        viewerAny._proteinModel = proteinModel;
      };

      const ensurePharmacophoreShapes = () => {
        if (!pharmacophoreAtomsRef.current || viewerAny._pharmacophoreShapes) return;
        viewerAny._pharmacophoreShapes = addPharmacophoreAtoms(
          viewer,
          pharmacophoreAtomsRef.current
        );
      };

      const rebuildProteinSurface = (ligandModel: any) => {
        if (!viewerAny._proteinModel || viewerAny._proteinSurface) return;

        try {
          const surface = viewer.addSurface(
            window.$3Dmol.VDW,
            { opacity: 0.6, colorscheme: 'whiteCarbon' },
            {
              model: viewerAny._proteinModel,
              within: { distance: 6.0, sel: { model: ligandModel } },
            }
          );
          viewerAny._proteinSurface = surface;
        } catch (err) {
          console.error('Failed to add protein surface:', err);
        }
      };

      const updateScene = () => {
        const camera = safeGetCamera();
        removeExistingLigandModel();

        const fileFormat = filename.split('.').pop()?.toLowerCase() || 'sdf';
        const ligandModel = viewer.addModel(fileContent, fileFormat);
        viewer.setStyle({ model: ligandModel }, { stick: { radius: 0.15 } });
        viewerAny._ligandModel = ligandModel;

        ensureProteinModel();
        ensurePharmacophoreShapes();
        rebuildProteinSurface(ligandModel);

        safeSetCamera(camera, ligandModel);
        viewer.render();
        hasBuiltSceneRef.current = true;
      };

      updateScene();
    };

    if (!isLoading && fileContent) {
      load3Dmol();
    }

    return () => {
      // Don't cleanup - keep viewer alive across filename changes
    };
  }, [fileContent, jobId, filename, samplingMode, isLoading]);

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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading molecule...</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <Download className="w-4 h-4" />
          Download
        </button>
      </div>
      <div
        ref={containerRef}
        style={{ width: '100%', height: '500px', position: 'relative' }}
        className="three-d-viewer-container border border-gray-300 rounded-lg overflow-hidden"
      />
      {(samplingMode === 'Pharmacophore-conditioned' ||
        samplingMode === 'Protein+Pharmacophore-conditioned') && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold mb-2">Pharmacophore Color Legend</h4>
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div>🟣 <strong>Aromatic</strong> - Purple</div>
            <div>⚪ <strong>Hydrogen Donor</strong> - White</div>
            <div>🟠 <strong>Hydrogen Acceptor</strong> - Orange</div>
            <div>🔵 <strong>Positive Ion</strong> - Blue</div>
            <div>🔴 <strong>Negative Ion</strong> - Red</div>
            <div>🟢 <strong>Hydrophobic</strong> - Green</div>
            <div>🔷 <strong>Halogen</strong> - Cyan</div>
          </div>
        </div>
      )}
    </div>
  );
}

function parsePharmacophoreXyz(xyzContent: string) {
  const lines = xyzContent.split('\n');
  const atomLines = lines.slice(2); // skip header

  const elementToType: Record<string, string> = {
    P: 'Aromatic',
    S: 'HydrogenDonor',
    F: 'HydrogenAcceptor',
    N: 'PositiveIon',
    O: 'NegativeIon',
    C: 'Hydrophobic',
    Cl: 'Halogen',
  };

  const typeToColor: Record<string, string> = {
    Aromatic: 'purple',
    HydrogenDonor: '#f0f0f0',
    HydrogenAcceptor: 'orange',
    PositiveIon: 'blue',
    NegativeIon: 'red',
    Hydrophobic: 'green',
    Halogen: 'cyan',
  };

  return atomLines
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(/\s+/);
      if (parts.length < 4) return null;
      const element = parts[0];
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]);
      const z = parseFloat(parts[3]);
      const pharmType = elementToType[element] || 'Unknown';
      const color = typeToColor[pharmType] || 'gray';
      return { x, y, z, color };
    })
    .filter(Boolean) as Array<{ x: number; y: number; z: number; color: string }>;
}

function addPharmacophoreAtoms(
  viewer: any,
  atoms: Array<{ x: number; y: number; z: number; color: string }>
) {
  return atoms.map((atom) =>
    viewer.addSphere({
      center: { x: atom.x, y: atom.y, z: atom.z },
      radius: 1.0,
      color: atom.color,
      wireframe: true,
      linewidth: 1.5,
    })
  );
}

