'use client';

import { useEffect, useRef, useState } from 'react';

type ProteinStyle = 'cartoon' | 'surface' | 'sticks' | 'backbone';
type LigandStyle = 'stick' | 'sphere' | 'line';

declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface CentralSelectionViewerProps {
  proteinContent?: string; // base64 encoded protein file content
  proteinFormat?: 'pdb' | 'cif';
  ligandContent?: string; // base64 encoded ligand file content
  detectedPockets?: Array<{
    id: string;
    center: [number, number, number];
    bbox_length: number;
    score?: number;
  }>;
  selectedPocketId?: string | null;
  onPocketSelect?: (pocketId: string) => void;
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
}

export function CentralSelectionViewer({
  proteinContent,
  proteinFormat,
  ligandContent,
  detectedPockets = [],
  selectedPocketId,
  onPocketSelect,
  pharmacophores = [],
  selectedPharmacophoreIndices = [],
  onPharmacophoreSelectionChange,
  pocketSelectionMethod = 'detected',
  manualCenter = { x: '0', y: '0', z: '0' },
  bboxLength = '15.0',
  ligandCenter,
  refLigandContent,
  pharmacophoreTolerance,
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
        shouldZoom = true; // Always zoom when protein changes (it's the main context)

        // Style protein ATOM records (cartoon/sticks per toggle)
        const protStyle: any = {};
        if (showBackbone) protStyle.cartoon = { color: 'lightblue' };
        if (showSticks) protStyle.stick = { radius: 0.15, colorscheme: 'lightgreyCarbon' };
        viewer.setStyle({ model: protModel, hetflag: false }, protStyle);
        // HETATM records (cofactors, small molecules in PDB) always as sticks
        viewer.setStyle({ model: protModel, hetflag: true }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
        if (showSurface) viewer.addSurface(window.$3Dmol.VDW, { opacity: 0.6, colorscheme: 'whiteCarbon', depthWrite: true }, { model: protModel });
      }
    } else if (viewerAny._proteinModel) {
      // Protein was removed
      try {
        viewer.removeModel(viewerAny._proteinModel);
        viewer.removeAllSurfaces();
        // Remove bounding box shapes if they exist
        if (viewerAny._bboxShapes) {
          viewerAny._bboxShapes.forEach((s: any) => { try { viewer.removeShape(s); } catch (e) { } });
          viewerAny._bboxShapes = undefined;
        }
      } catch (e) { console.warn(e); }
      viewerAny._proteinModel = undefined;
      viewerAny._proteinData = undefined;
      shouldZoom = true; // Re-center on remaining items
    }

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
        viewer.setStyle({ model: ligModel }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
        if (!viewerAny._proteinModel) shouldZoom = true; // Zoom if no protein to anchor
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
        viewer.setStyle({ model: refModel }, { stick: { radius: 0.2, colorscheme: 'lightgreyCarbon', color: 'lime' } });
        if (!viewerAny._proteinModel) shouldZoom = true;
      }
    } else if (viewerAny._refLigandModel) {
      try { viewer.removeModel(viewerAny._refLigandModel); } catch (e) { console.warn(e); }
      viewerAny._refLigandModel = undefined;
      viewerAny._refData = undefined;
      if (!viewerAny._proteinModel) shouldZoom = true;
    }

    // Smart Zoom Logic
    if (shouldZoom || !isSceneReady) {
      viewer.zoomTo();
    } else {
      viewer.setView(currentView);
    }

    viewer.render();
    if (!isSceneReady) {
      setIsSceneReady(true);
    }
  }, [proteinContent, proteinFormat, ligandContent, refLigandContent, pharmacophores]);

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
      // HETATM always as sticks
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

    // Bounding Boxes
    const parsedBboxLen = parseFloat(bboxLength);
    if (showBoxes) {
      if (pocketSelectionMethod === 'detected') {
        // Helper to create a thick wireframe using cylinders
        const createThickBox = (
          center: { x: number, y: number, z: number },
          dims: { w: number, h: number, d: number },
          color: string,
          radius: number = 0.01,
          callback?: () => void
        ) => {
          const { x, y, z } = center;
          const { w, h, d } = dims;
          const hw = w / 2;
          const hh = h / 2;
          const hd = d / 2;

          // 8 Corners
          const c1 = { x: x - hw, y: y - hh, z: z - hd };
          const c2 = { x: x + hw, y: y - hh, z: z - hd };
          const c3 = { x: x + hw, y: y + hh, z: z - hd };
          const c4 = { x: x - hw, y: y + hh, z: z - hd };
          const c5 = { x: x - hw, y: y - hh, z: z + hd };
          const c6 = { x: x + hw, y: y - hh, z: z + hd };
          const c7 = { x: x + hw, y: y + hh, z: z + hd };
          const c8 = { x: x - hw, y: y + hh, z: z + hd };

          // 12 Edges
          const edges = [
            [c1, c2], [c2, c3], [c3, c4], [c4, c1], // Front face (z-minus)
            [c5, c6], [c6, c7], [c7, c8], [c8, c5], // Back face (z-plus)
            [c1, c5], [c2, c6], [c3, c7], [c4, c8]  // Connecting edges
          ];

          edges.forEach(([start, end]) => {
            viewer.addCylinder({
              start, end,
              radius,
              color,
              fromCap: 1, toCap: 1, // Rounded caps
              clickable: !!callback, // Add click ability if callback exists
              callback: callback
            });
          });
        };

        // Pass 1: Render all UNSELECTED pockets first
        detectedPockets.forEach((pocket) => {
          if (pocket.id === selectedPocketId) return; // Skip selected for now

          const center = { x: pocket.center[0], y: pocket.center[1], z: pocket.center[2] };
          const dims = { w: parsedBboxLen, h: parsedBboxLen, d: parsedBboxLen };

          // 1. Thick Visual Wireframe (Cylinders) - Now Clickable!
          createThickBox(center, dims, 'black', 0.035, () => {
            if (onPocketSelect) onPocketSelect(pocket.id);
          });

          // 2. Standard Wireframe Box (Fallback Interaction)
          // This ensures that if the cylinders are missed, the wire outline is still clickable.
          // Wireframes do not occlude the view.
          viewer.addBox({
            center,
            dimensions: dims,
            color: 'black',
            wireframe: true,
            clickable: true,
            opacity: 1.0, // Fully visible thin lines (hidden inside thick cylinders usually)
            callback: () => {
              if (onPocketSelect) onPocketSelect(pocket.id);
            }
          });
        });

        // Pass 2: Render SELECTED pocket last
        const selectedPocket = detectedPockets.find(p => p.id === selectedPocketId);
        if (selectedPocket) {
          // Selected: High alpha (0.7)
          const boxOptions: any = {
            center: { x: selectedPocket.center[0], y: selectedPocket.center[1], z: selectedPocket.center[2] },
            dimensions: { w: parsedBboxLen, h: parsedBboxLen, d: parsedBboxLen },
            color: 'yellow',
            opacity: 0.8,
            wireframe: false,
            clickable: true,
            callback: () => {
              if (onPocketSelect) onPocketSelect(selectedPocket.id);
            }
          };
          const shape = viewer.addBox(boxOptions);

          // Force selected to be visible but slightly transparent
          if (shape) {
            setTimeout(() => {
              try {
                const setProps = (obj: any) => {
                  if (!obj) return;
                  if (obj.renderOrder !== undefined) obj.renderOrder = 9999;
                  if (obj.material) {
                    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                    mats.forEach((m: any) => {
                      m.depthTest = false;
                      m.depthWrite = false;
                      m.transparent = true;
                      m.opacity = 0.8;
                      m.needsUpdate = true;
                    });
                  }
                  if (obj.children) obj.children.forEach(setProps);
                  if (obj.mesh) setProps(obj.mesh);
                };
                setProps(shape);
                viewer.render();
              } catch (e) { }
            }, 50);
          }

          // 3. Thick Visual Wireframe for Selected (Cylinders)
          const center = { x: selectedPocket.center[0], y: selectedPocket.center[1], z: selectedPocket.center[2] };
          const dims = { w: parsedBboxLen, h: parsedBboxLen, d: parsedBboxLen };
          createThickBox(center, dims, 'black', 0.035);
        }

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
    pharmacophores, // Re-run if pharmacophores array changes
    selectedPharmacophoreIndices,
    pocketSelectionMethod,
    manualCenter,
    bboxLength,
    onPocketSelect,
    onPharmacophoreSelectionChange,
    pharmacophoreTolerance,
    showBoxes
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
            depthWrite: true
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
            Bounding Boxes
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
