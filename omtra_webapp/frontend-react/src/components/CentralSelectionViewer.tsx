'use client';

import { useEffect, useRef, useState } from 'react';
import type { BricsFragment } from '@/types';

type ProteinStyle = 'cartoon' | 'surface' | 'sticks' | 'backbone';
type LigandStyle = 'stick' | 'sphere' | 'line';

declare global {
  interface Window {
    $3Dmol: any;
  }
}

const FRAGMENT_COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
  '#0891b2', '#ca8a04', '#db2777', '#4f46e5', '#65a30d',
];
const DIMMED_COLOR = '#d1d5db';

interface CentralSelectionViewerProps {
  proteinContent?: string;
  proteinFormat?: 'pdb' | 'cif';
  ligandContent?: string;
  detectedPockets?: Array<{
    id: string;
    center: [number, number, number];
    bbox_length: number;
    score?: number;
    alpha_sphere_centers?: [number, number, number][];
    alpha_sphere_radii?: number[];
  }>;
  selectedPocketId?: string | null;
  onPocketSelect?: (pocketId: string | null) => void;
  hiddenPocketIds?: string[];
  pharmacophores?: Array<{
    index: number;
    type: string;
    x: number;
    y: number;
    z: number;
    color: string;
    selected: boolean;
  }>;
  selectedPharmacophoreIndices?: number[];
  onPharmacophoreSelectionChange?: (indices: number[]) => void;
  pocketSelectionMethod?: 'detected' | 'ligand' | 'manual';
  manualCenter?: { x: string; y: string; z: string };
  bboxLength?: string;
  ligandCenter?: [number, number, number] | null;
  refLigandContent?: string | null;
  pharmacophoreTolerance?: number;
  bricsFragments?: BricsFragment[];
  selectedFragmentIds?: number[];
  onFragmentSelectionChange?: (ids: number[]) => void;
  bricsRawSdf?: string;
}

export function CentralSelectionViewer({
  proteinContent,
  proteinFormat,
  ligandContent,
  detectedPockets = [],
  selectedPocketId,
  onPocketSelect,
  hiddenPocketIds = [],
  pharmacophores = [],
  selectedPharmacophoreIndices = [],
  onPharmacophoreSelectionChange,
  pocketSelectionMethod = 'detected',
  manualCenter = { x: '0', y: '0', z: '0' },
  bboxLength = '15.0',
  ligandCenter,
  refLigandContent,
  pharmacophoreTolerance,
  bricsFragments = [],
  selectedFragmentIds = [],
  onFragmentSelectionChange,
  bricsRawSdf,
}: CentralSelectionViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const surfaceRef = useRef<any>(null);
  const [isSceneReady, setIsSceneReady] = useState(false);

  // Styling state - simple toggles
  const [showSticks, setShowSticks] = useState(false);
  const [showSurface, setShowSurface] = useState(false);
  const [showBackbone, setShowBackbone] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);

  // Initialize Viewer (Once)
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

      if (!viewerRef.current) {
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          defaultcolors: window.$3Dmol.rasmolElementColors,
        });

        const viewerAny = viewer as any;
        if (viewerAny.camera) {
          viewerAny.camera.near = 0.01;
          viewerAny.camera.far = 1000000;
          viewerAny.camera.updateProjectionMatrix();
        }

        viewerRef.current = viewer;
      }
    };

    load3Dmol();
  }, []);

  // Handle Model Loading (Persistent Models to prevent flickering)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const viewerAny = viewer as any;

    const hasAnyContent = !!(proteinContent || ligandContent || refLigandContent || (pharmacophores && pharmacophores.length > 0));

    if (!hasAnyContent) {
      viewer.clear();
      setIsSceneReady(false);
      viewerAny._proteinModel = undefined;
      viewerAny._ligandModel = undefined;
      viewerAny._refLigandModel = undefined;
      viewerAny._proteinData = undefined;
      viewerAny._ligandData = undefined;
      viewerAny._refData = undefined;
      return;
    }

    // Capture view to keep camera steady if needed
    const currentView = viewer.getView();
    let shouldZoom = false;
    let zoomTarget: any = undefined; // undefined = zoom to all, or a specific model

    // 1. Protein Layer
    if (proteinContent && proteinFormat) {
      const proteinData = atob(proteinContent);
      if (viewerAny._proteinData !== proteinData) {
        if (viewerAny._proteinModel) {
          try { viewer.removeModel(viewerAny._proteinModel); } catch (e) { console.warn(e); }
        }
        const protModel = viewer.addModel(proteinData, proteinFormat);
        viewerAny._proteinModel = protModel;
        viewerAny._proteinData = proteinData;
        shouldZoom = true;

        // Style protein ATOM records (cartoon/sticks per toggle)
        const protStyle: any = {};
        if (showBackbone) protStyle.cartoon = { color: 'lightblue' };
        if (showSticks) protStyle.stick = { radius: 0.15, colorscheme: 'lightgreyCarbon' };
        viewer.setStyle({ model: protModel, hetflag: false }, protStyle);
        // HETATM records (cofactors, small molecules in PDB) always as sticks
        viewer.setStyle({ model: protModel, hetflag: true }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
        if (showSurface) viewer.addSurface(window.$3Dmol.VDW, { opacity: 0.6, colorscheme: 'whiteCarbon' }, { model: protModel });

      }
    } else if (viewerAny._proteinModel) {
      // Protein was removed
      try {
        viewer.removeModel(viewerAny._proteinModel);
        viewer.removeAllSurfaces();
      } catch (e) { console.warn(e); }
      viewerAny._proteinModel = undefined;
      viewerAny._proteinData = undefined;
      shouldZoom = true;
    }

    const hasBrics = bricsFragments.length > 1 && !!bricsRawSdf;
    const bricsB64 = hasBrics ? btoa(bricsRawSdf) : null;
    const fragsOnLigand = hasBrics && !!ligandContent && bricsB64 === ligandContent;
    const fragsOnRef = hasBrics && !fragsOnLigand && !!refLigandContent && bricsB64 === refLigandContent;

    const setupFragmentClicks = (model: any) => {
      viewer.setClickable({ model }, true, (atom: any) => {
        if (!onFragmentSelectionChange) return;
        const fragId = findFragmentForAtom(atom.index, bricsFragments);
        if (fragId === null) return;
        const newIds = selectedFragmentIds.includes(fragId)
          ? selectedFragmentIds.filter((id) => id !== fragId)
          : [...selectedFragmentIds, fragId];
        onFragmentSelectionChange(newIds);
      });
    };

    // 2. Ligand Layer
    if (ligandContent) {
      const ligandData = atob(ligandContent);
      if (viewerAny._ligandData !== ligandData) {
        if (viewerAny._ligandModel) {
          try { viewer.removeModel(viewerAny._ligandModel); } catch (e) { console.warn(e); }
        }
        const ligModel = viewer.addModel(ligandData, 'sdf');
        viewerAny._ligandModel = ligModel;
        viewerAny._ligandData = ligandData;

        if (fragsOnLigand) {
          applyFragmentStyles(viewer, ligModel, bricsFragments, selectedFragmentIds);
          setupFragmentClicks(ligModel);
        } else {
          viewer.setStyle({ model: ligModel }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
        }
        shouldZoom = true;
        zoomTarget = { model: ligModel };
      }
    } else if (viewerAny._ligandModel) {
      try { viewer.removeModel(viewerAny._ligandModel); } catch (e) { console.warn(e); }
      viewerAny._ligandModel = undefined;
      viewerAny._ligandData = undefined;
      if (!viewerAny._proteinModel) shouldZoom = true;
    }

    // 3. Reference Ligand (Pocket Ligand)
    if (refLigandContent) {
      const refData = atob(refLigandContent);
      if (viewerAny._refData !== refData) {
        if (viewerAny._refLigandModel) {
          try { viewer.removeModel(viewerAny._refLigandModel); } catch (e) { console.warn(e); }
        }
        const refModel = viewer.addModel(refData, 'sdf');
        viewerAny._refLigandModel = refModel;
        viewerAny._refData = refData;

        if (fragsOnRef) {
          applyFragmentStyles(viewer, refModel, bricsFragments, selectedFragmentIds);
          setupFragmentClicks(refModel);
        } else {
          viewer.setStyle({ model: refModel }, { stick: { radius: 0.2, colorscheme: 'lightgreyCarbon', color: 'lime' } });
        }
        shouldZoom = true;
        zoomTarget = { model: refModel };
      }
    } else if (viewerAny._refLigandModel) {
      try { viewer.removeModel(viewerAny._refLigandModel); } catch (e) { console.warn(e); }
      viewerAny._refLigandModel = undefined;
      viewerAny._refData = undefined;
      if (!viewerAny._proteinModel) shouldZoom = true;
    }

    // Smart Zoom Logic — center on ligand when available, otherwise fit all
    if (shouldZoom || !isSceneReady) {
      if (zoomTarget) {
        viewer.zoomTo(zoomTarget);
      } else {
        viewer.zoomTo();
      }
    } else {
      viewer.setView(currentView);
    }

    viewer.render();
    if (!isSceneReady) {
      setIsSceneReady(true);
    }
  }, [proteinContent, proteinFormat, ligandContent, refLigandContent, bricsRawSdf, pharmacophores, bricsFragments, selectedFragmentIds, onFragmentSelectionChange]);

  // Update fragment styles when selection changes (without reloading model)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady || bricsFragments.length <= 1) return;
    const viewerAny = viewer as any;

    // Pick the model that owns the fragments (match bricsRawSdf to model content)
    const b64 = bricsRawSdf ? btoa(bricsRawSdf) : null;
    const onLig = b64 && ligandContent && b64 === ligandContent;
    const model = onLig ? viewerAny._ligandModel : viewerAny._refLigandModel;
    if (!model) return;

    applyFragmentStyles(viewer, model, bricsFragments, selectedFragmentIds);
    viewer.setClickable({ model }, true, (atom: any) => {
      if (!onFragmentSelectionChange) return;
      const fragId = findFragmentForAtom(atom.index, bricsFragments);
      if (fragId === null) return;
      const newIds = selectedFragmentIds.includes(fragId)
        ? selectedFragmentIds.filter((id) => id !== fragId)
        : [...selectedFragmentIds, fragId];
      onFragmentSelectionChange(newIds);
    });
  }, [selectedFragmentIds, bricsFragments, isSceneReady, onFragmentSelectionChange, ligandContent, refLigandContent, bricsRawSdf]);

  // 1. Handle Style Toggles (Cartoon, Sticks) - Lightweight
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;
    const viewerAny = viewer as any;

    if (viewerAny._proteinModel) {
      const protStyle: any = {};
      if (showBackbone) protStyle.cartoon = { color: 'lightblue' };
      if (showSticks) protStyle.stick = { radius: 0.15, colorscheme: 'lightgreyCarbon' };
      viewer.setStyle({ model: viewerAny._proteinModel, hetflag: false }, protStyle);
      viewer.setStyle({ model: viewerAny._proteinModel, hetflag: true }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
    }

    viewer.render();
  }, [showSticks, showBackbone, isSceneReady]);

  // 2. Handle Shape Updates (Boxes, Pharmacophore spheres)
  useEffect(() => {
    const viewer = viewerRef.current;
    console.log('Shape update requested:', { isSceneReady, nPharms: pharmacophores.length, nPockets: detectedPockets.length });
    if (!viewer || !isSceneReady) return;

    viewer.removeAllShapes();
    console.log(`Adding ${pharmacophores.length} pharmacophore spheres to scene`);

    // Pocket visualization
    const parsedBboxLen = parseFloat(bboxLength);
    const POCKET_COLORS = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];
    const DETECTED_POCKET_REFERENCE_RADIUS = 8.0;
    if (showBoxes) {
      if (pocketSelectionMethod === 'detected') {
        // Render pockets as alpha sphere blobs
        detectedPockets.forEach((pocket, pocketIdx) => {
          if (hiddenPocketIds.includes(pocket.id)) return;
          const isSelected = pocket.id === selectedPocketId;
          const color = POCKET_COLORS[pocketIdx % POCKET_COLORS.length];
          const alphaSphereCenters = pocket.alpha_sphere_centers;

          const clickHandler = () => {
            if (onPocketSelect) onPocketSelect(isSelected ? null : pocket.id);
          };

          if (alphaSphereCenters && alphaSphereCenters.length > 0) {
            for (const centerCoords of alphaSphereCenters) {
              viewer.addSphere({
                center: { x: centerCoords[0], y: centerCoords[1], z: centerCoords[2] },
                radius: DETECTED_POCKET_REFERENCE_RADIUS,
                color,
                alpha: isSelected ? 0.8 : 0.5,
                clickable: true,
                callback: clickHandler,
              });
            }
          } else {
            // Fallback: single sphere at pocket center
            viewer.addSphere({
              center: { x: pocket.center[0], y: pocket.center[1], z: pocket.center[2] },
              radius: parsedBboxLen / 2,
              color,
              alpha: isSelected ? 0.8 : 0.5,
              clickable: true,
              callback: clickHandler,
            });
          }
        });

      } else if (pocketSelectionMethod === 'manual') {
        const center = [parseFloat(manualCenter.x), parseFloat(manualCenter.y), parseFloat(manualCenter.z)];
        if (!center.some(isNaN)) {
          const boxOptions: any = {
            center: { x: center[0], y: center[1], z: center[2] },
            dimensions: { w: parsedBboxLen, h: parsedBboxLen, d: parsedBboxLen },
            color: 'yellow',
            opacity: 0.6,
            wireframe: false,
          };
          viewer.addBox(boxOptions);
          viewer.addBox({ ...boxOptions, wireframe: true, color: 'black', opacity: 1.0 });
        }
      }
    }

    // Pharmacophores
    pharmacophores.forEach((pharm) => {
      const isSelected = selectedPharmacophoreIndices.includes(pharm.index);

      const clickHandler = () => {
        if (onPharmacophoreSelectionChange) {
          const newIndices = [...selectedPharmacophoreIndices];
          const idx = newIndices.indexOf(pharm.index);
          if (idx >= 0) newIndices.splice(idx, 1);
          else newIndices.push(pharm.index);
          onPharmacophoreSelectionChange(newIndices);
        }
      };

      const center = { x: pharm.x, y: pharm.y, z: pharm.z };

      // Calculate radius based on tolerance
      // user example: std=1.0 -> radius=3.0
      // Default (tolerance=0) -> radius=1.0
      const visualRadius = 1.0 + (pharmacophoreTolerance || 0) * 2.0;

      const sphereOptions: any = {
        center,
        radius: visualRadius,
        color: pharm.color,
        clickable: true,
        callback: clickHandler
      };

      if (isSelected) {
        // Selected: Solid opaque sphere
        sphereOptions.alpha = 1.0;
        sphereOptions.wireframe = false;
      } else {
        // Unselected: Wireframe
        sphereOptions.wireframe = true;
        sphereOptions.linewidth = 1.5;
      }

      const shape = viewer.addSphere(sphereOptions);

      // Force Always-On-Top for visibility through bbox/protein
      if (shape && shape.material) {
        try {
          const materials = Array.isArray(shape.material) ? shape.material : [shape.material];
          materials.forEach((mat: any) => {
            if (mat) {
              mat.depthTest = false;
              mat.depthWrite = false;
              mat.transparent = true;
              if (isSelected) mat.opacity = 1.0;
            }
          });
        } catch (e) { console.warn(e); }
      }
    });

    // Re-add fragment wireframe spheres (removeAllShapes above clears them)
    (viewer as any)._fragSpheres = [];
    if (bricsFragments.length > 1 && selectedFragmentIds.length > 0) {
      const b64 = bricsRawSdf ? btoa(bricsRawSdf) : null;
      const onLig = b64 && ligandContent && b64 === ligandContent;
      const fragModel = onLig ? (viewer as any)._ligandModel : (viewer as any)._refLigandModel;
      if (fragModel) {
        const atoms = fragModel.selectedAtoms({});
        for (const frag of bricsFragments) {
          if (!selectedFragmentIds.includes(frag.id)) continue;
          const color = FRAGMENT_COLORS[frag.id % FRAGMENT_COLORS.length];
          for (const atomIdx of frag.atom_indices) {
            const atom = atoms.find((a: any) => a.index === atomIdx);
            if (!atom) continue;
            const shape = viewer.addSphere({
              center: { x: atom.x, y: atom.y, z: atom.z },
              radius: 0.35,
              color,
              wireframe: true,
              linewidth: 1.5,
            });
            (viewer as any)._fragSpheres.push(shape);
          }
        }
      }
    }

    viewer.render();

    // For pharmacophore-only mode, zoom to fit pharmacophores after rendering
    const hasPharmacophoresOnly = pharmacophores.length > 0 && !proteinContent && !ligandContent && !refLigandContent;
    if (hasPharmacophoresOnly) {
      viewer.zoomTo();
    }
  }, [
    isSceneReady,
    detectedPockets,
    selectedPocketId,
    hiddenPocketIds,
    pharmacophores,
    selectedPharmacophoreIndices,
    pocketSelectionMethod,
    manualCenter,
    bboxLength,
    onPocketSelect,
    onPharmacophoreSelectionChange,
    pharmacophoreTolerance,
    showBoxes,
    bricsFragments,
    selectedFragmentIds,
    bricsRawSdf,
    ligandContent,
  ]);

  // 3. Handle Surface Toggles
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;
    const viewerAny = viewer as any;

    const applySurface = () => {
      try {
        viewer.removeAllSurfaces();
        if (showSurface && viewerAny._proteinModel) {
          viewer.addSurface(window.$3Dmol.VDW, {
            opacity: 0.6,
            colorscheme: 'whiteCarbon',
          }, { model: viewerAny._proteinModel });
        }
        viewer.render();
      } catch (e) {
        console.warn('Surface sync failed:', e);
      }
    };

    applySurface();
  }, [
    showSurface,
    isSceneReady,
    proteinContent, // Re-run when protein changes to update surface
  ]);

  return (
    <div className="h-full w-full relative">
      <div ref={containerRef} className="h-full w-full" style={{ minHeight: '500px' }} />

      {proteinContent && (
        <div className="absolute top-4 right-4 bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg border border-slate-200/60 p-3 space-y-2 z-20 text-sm">
          <div className="font-semibold text-slate-700 text-xs mb-2">Style</div>
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
          <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer hover:bg-slate-50 py-1 px-2 rounded-lg transition-colors">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(e) => setShowBoxes(e.target.checked)}
              className="w-3.5 h-3.5 rounded"
            />
            Pockets
          </label>
        </div>
      )}

      {!proteinContent && !ligandContent && !refLigandContent && pharmacophores.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
          <div className="text-center text-slate-500 bg-white/80 px-6 py-4 rounded-lg">
            <p className="text-lg font-medium mb-2">Viewer</p>
            <p className="text-sm">Upload files to visualize/select conditional information</p>
          </div>
        </div>
      )}
    </div>
  );
}

function applyFragmentStyles(
  viewer: any,
  model: any,
  fragments: BricsFragment[],
  selectedIds: number[]
) {
  // Base: standard element-colored sticks (unchanged from default look)
  viewer.setStyle({ model }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });

  // Clean up previous fragment spheres
  if ((viewer as any)._fragSpheres) {
    for (const s of (viewer as any)._fragSpheres) {
      try { viewer.removeShape(s); } catch (_) {}
    }
  }
  (viewer as any)._fragSpheres = [];

  if (selectedIds.length === 0) return;

  const atoms = model.selectedAtoms({});
  for (const frag of fragments) {
    if (!selectedIds.includes(frag.id)) continue;
    const color = FRAGMENT_COLORS[frag.id % FRAGMENT_COLORS.length];
    for (const atomIdx of frag.atom_indices) {
      const atom = atoms.find((a: any) => a.index === atomIdx);
      if (!atom) continue;
      const shape = viewer.addSphere({
        center: { x: atom.x, y: atom.y, z: atom.z },
        radius: 0.35,
        color,
        wireframe: true,
        linewidth: 1.5,
      });
      (viewer as any)._fragSpheres.push(shape);
    }
  }
}

function findFragmentForAtom(atomIndex: number, fragments: BricsFragment[]): number | null {
  for (const frag of fragments) {
    if (frag.atom_indices.includes(atomIndex)) return frag.id;
  }
  return null;
}
