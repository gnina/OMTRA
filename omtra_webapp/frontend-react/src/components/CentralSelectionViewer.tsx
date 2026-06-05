'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { BricsFragment } from '@/types';
import { decodeBase64Unicode, encodeBase64Unicode } from '@/lib/api-client';
import type { FixStructureMode, SelectionAction } from '@/hooks/useFixedAtomSelection';
import {
  atomsInSelectionBox,
  getViewerPointerElement,
  normalizeRect,
  pointerLocalFromEvent,
} from '@/lib/boxSelect3d';

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
const FIXED_ATOM_COLOR = '#f59e0b';
const PROTEIN_SURFACE_STYLE = {
  opacity: 0.6,
  colorscheme: 'whiteCarbon',
};
const DETECTED_POCKET_REFERENCE_RADIUS = 8.0;
const POCKET_OPACITY_UNSELECTED = 0.65;
const POCKET_OPACITY_SELECTED = 0.85;

function visitShapeMaterials(shape: any, fn: (mat: any, obj: any) => void) {
  const visit = (obj: any) => {
    if (!obj) return;
    if (obj.material) {
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((mat: any) => fn(mat, obj));
    }
    obj.children?.forEach(visit);
    if (obj.mesh) visit(obj.mesh);
  };
  visit(shape);
}

/** 3Dmol addSphere returns GLShape; opacity must touch renderedShapeObj after render(). */
function stylePocketSphere(glShape: any, colorHex: string, opacity: number) {
  if (!glShape) return;
  glShape.opacity = opacity;
  if (glShape.stylespec) {
    glShape.stylespec.alpha = opacity;
    glShape.stylespec.opacity = opacity;
  }
  const applyToMaterials = (root: any) => {
    visitShapeMaterials(root, (mat, obj) => {
      if (obj.renderOrder !== undefined) obj.renderOrder = 2000;
      if (mat.color?.setStyle) mat.color.setStyle(colorHex);
      mat.opacity = opacity;
      mat.transparent = opacity < 1;
      mat.depthTest = false;
      mat.depthWrite = false;
      mat.needsUpdate = true;
    });
  };
  applyToMaterials(glShape.renderedShapeObj);
  applyToMaterials(glShape.shapeObj);
}

interface CentralSelectionViewerProps {
  visible?: boolean;
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
  fixStructureActive?: boolean;
  fixStructureMode?: FixStructureMode;
  selectionAction?: SelectionAction;
  selectedAtomIndices?: number[];
  onAtomClick?: (atomIndex: number) => void;
  onAtomsInBox?: (indices: number[]) => void;
  onFragmentsInBox?: (fragmentIds: number[]) => void;
  onToggleFragmentByAtom?: (atomIndex: number) => void;
}

export function CentralSelectionViewer({
  visible = true,
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
  fixStructureActive = false,
  fixStructureMode = 'fragment',
  selectionAction = 'toggle',
  selectedAtomIndices = [],
  onAtomClick,
  onAtomsInBox,
  onFragmentsInBox,
  onToggleFragmentByAtom,
}: CentralSelectionViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [isSceneReady, setIsSceneReady] = useState(false);
  const boxDragRef = useRef<{ active: boolean; x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [boxOverlay, setBoxOverlay] = useState<{ left: number; top: number; width: number; height: number } | null>(null);
  const suppressClickRef = useRef(false);
  const fixedAtomSet = useRef(new Set(selectedAtomIndices));
  fixedAtomSet.current = new Set(selectedAtomIndices);
  const onPocketSelectRef = useRef(onPocketSelect);
  const selectedPocketIdRef = useRef(selectedPocketId);
  const pocketShapeGroupsRef = useRef<Array<{ pocketId: string; color: string; shapes: any[] }>>([]);
  onPocketSelectRef.current = onPocketSelect;
  selectedPocketIdRef.current = selectedPocketId;

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

  // Re-render when tab becomes visible (WebGL canvas goes blank under CSS display:none)
  useEffect(() => {
    if (visible && viewerRef.current && isSceneReady) {
      viewerRef.current.render();
    }
  }, [visible, isSceneReady]);

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
      const proteinData = decodeBase64Unicode(proteinContent);
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
    const bricsB64 = hasBrics ? encodeBase64Unicode(bricsRawSdf as string) : null;
    const fragsOnLigand = hasBrics && !!ligandContent && bricsB64 === ligandContent;
    const fragsOnRef = hasBrics && !fragsOnLigand && !!refLigandContent && bricsB64 === refLigandContent;

    const setupFixStructureClicks = (model: any) => {
      if (!fixStructureActive) return;
      viewer.setClickable({ model }, true, (atom: any) => {
        if (suppressClickRef.current || boxDragRef.current?.active) return;
        if (fixStructureMode === 'atom') {
          onAtomClick?.(atom.index);
        } else if (onToggleFragmentByAtom) {
          onToggleFragmentByAtom(atom.index);
        } else if (onFragmentSelectionChange) {
          const fragId = findFragmentForAtom(atom.index, bricsFragments);
          if (fragId === null) return;
          const newIds = selectedFragmentIds.includes(fragId)
            ? selectedFragmentIds.filter((id) => id !== fragId)
            : [...selectedFragmentIds, fragId];
          onFragmentSelectionChange(newIds);
        }
      });
    };

    // 2. Ligand Layer
    if (ligandContent) {
      const ligandData = decodeBase64Unicode(ligandContent);
      if (viewerAny._ligandData !== ligandData) {
        if (viewerAny._ligandModel) {
          try { viewer.removeModel(viewerAny._ligandModel); } catch (e) { console.warn(e); }
        }
        const ligModel = viewer.addModel(ligandData, 'sdf');
        viewerAny._ligandModel = ligModel;
        viewerAny._ligandData = ligandData;

        if (fragsOnLigand || fixStructureActive) {
          applyFixStructureStyles(
            viewer,
            ligModel,
            bricsFragments,
            selectedFragmentIds,
            fixedAtomSet.current,
            fixStructureMode,
            fixStructureActive,
          );
          setupFixStructureClicks(ligModel);
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
      const refData = decodeBase64Unicode(refLigandContent);
      if (viewerAny._refData !== refData) {
        if (viewerAny._refLigandModel) {
          try { viewer.removeModel(viewerAny._refLigandModel); } catch (e) { console.warn(e); }
        }
        const refModel = viewer.addModel(refData, 'sdf');
        viewerAny._refLigandModel = refModel;
        viewerAny._refData = refData;

        if (fragsOnRef || fixStructureActive) {
          applyFixStructureStyles(
            viewer,
            refModel,
            bricsFragments,
            selectedFragmentIds,
            fixedAtomSet.current,
            fixStructureMode,
            fixStructureActive,
          );
          setupFixStructureClicks(refModel);
        } else {
          viewer.setStyle({ model: refModel }, { stick: { radius: 0.2, colorscheme: 'lightgreyCarbon' } });
        }
        shouldZoom = true;
        zoomTarget = { model: refModel };
      }
    } else if (viewerAny._refLigandModel) {
      try { viewer.removeModel(viewerAny._refLigandModel); } catch (e) { console.warn(e); }
      viewerAny._refLigandModel = undefined;
      viewerAny._refData = undefined;
      // After reference ligand removal, zoom to remaining ligand if present, else protein
      if (viewerAny._ligandModel) {
        shouldZoom = true;
        zoomTarget = { model: viewerAny._ligandModel };
      } else if (viewerAny._proteinModel) {
        shouldZoom = true;
        zoomTarget = { model: viewerAny._proteinModel };
      } else {
        shouldZoom = true;
      }
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
  }, [proteinContent, proteinFormat, ligandContent, refLigandContent, bricsRawSdf, pharmacophores, bricsFragments, selectedFragmentIds, selectedAtomIndices, fixStructureActive, fixStructureMode, onFragmentSelectionChange, onAtomClick, onToggleFragmentByAtom]);

  const getFixStructureModel = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return null;
    const viewerAny = viewer as any;
    const b64 = bricsRawSdf ? encodeBase64Unicode(bricsRawSdf) : null;
    const onLig = b64 && ligandContent && b64 === ligandContent;
    if (onLig && viewerAny._ligandModel) return viewerAny._ligandModel;
    if (viewerAny._refLigandModel) return viewerAny._refLigandModel;
    return viewerAny._ligandModel ?? viewerAny._refLigandModel ?? null;
  }, [bricsRawSdf, ligandContent]);

  // Update fix-structure styles when selection changes (without reloading model)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady || !fixStructureActive) return;
    const model = getFixStructureModel();
    if (!model) return;

    applyFixStructureStyles(
      viewer,
      model,
      bricsFragments,
      selectedFragmentIds,
      fixedAtomSet.current,
      fixStructureMode,
      fixStructureActive,
    );
    viewer.setClickable({ model }, true, (atom: any) => {
      if (suppressClickRef.current || boxDragRef.current?.active) return;
      if (fixStructureMode === 'atom') {
        onAtomClick?.(atom.index);
      } else if (onToggleFragmentByAtom) {
        onToggleFragmentByAtom(atom.index);
      } else if (onFragmentSelectionChange) {
        const fragId = findFragmentForAtom(atom.index, bricsFragments);
        if (fragId === null) return;
        const newIds = selectedFragmentIds.includes(fragId)
          ? selectedFragmentIds.filter((id) => id !== fragId)
          : [...selectedFragmentIds, fragId];
        onFragmentSelectionChange(newIds);
      }
    });
    viewer.render();
  }, [selectedFragmentIds, selectedAtomIndices, bricsFragments, isSceneReady, fixStructureActive, fixStructureMode, onFragmentSelectionChange, onAtomClick, onToggleFragmentByAtom, getFixStructureModel]);

  // Shift+drag box selection (coordinates aligned with 3Dmol click picking)
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !fixStructureActive || !isSceneReady) return;

    const pointerEl = getViewerPointerElement(container);

    const onMouseDown = (e: MouseEvent) => {
      if (!e.shiftKey) return;
      e.preventDefault();
      e.stopPropagation();
      const { x, y } = pointerLocalFromEvent(e, pointerEl);
      boxDragRef.current = { active: true, x0: x, y0: y, x1: x, y1: y };
      setBoxOverlay({ left: x, top: y, width: 0, height: 0 });
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!boxDragRef.current?.active) return;
      if (!e.shiftKey) {
        boxDragRef.current = null;
        setBoxOverlay(null);
        return;
      }
      const { x, y } = pointerLocalFromEvent(e, pointerEl);
      boxDragRef.current.x1 = x;
      boxDragRef.current.y1 = y;
      const rect = normalizeRect(boxDragRef.current.x0, boxDragRef.current.y0, x, y);
      setBoxOverlay(rect);
    };

    const onMouseUp = (e: MouseEvent) => {
      if (!boxDragRef.current?.active) return;
      const drag = boxDragRef.current;
      boxDragRef.current = null;
      setBoxOverlay(null);

      if (!e.shiftKey) return;

      const rect = normalizeRect(drag.x0, drag.y0, drag.x1, drag.y1);
      if (rect.width < 4 && rect.height < 4) return;

      suppressClickRef.current = true;
      setTimeout(() => { suppressClickRef.current = false; }, 0);

      const viewer = viewerRef.current;
      const model = getFixStructureModel();
      if (!viewer || !model) return;

      const atomIndices = atomsInSelectionBox(viewer, model, pointerEl, rect);
      if (atomIndices.length === 0) return;

      if (fixStructureMode === 'atom') {
        onAtomsInBox?.(atomIndices);
      } else {
        const fragIds = new Set<number>();
        for (const idx of atomIndices) {
          const fid = findFragmentForAtom(idx, bricsFragments);
          if (fid !== null) fragIds.add(fid);
        }
        onFragmentsInBox?.([...fragIds]);
      }
    };

    container.addEventListener('mousedown', onMouseDown, true);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      container.removeEventListener('mousedown', onMouseDown, true);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [fixStructureActive, isSceneReady, fixStructureMode, bricsFragments, onAtomsInBox, onFragmentsInBox, getFixStructureModel]);

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

  // 2. Protein surface
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;
    const viewerAny = viewer as any;

    try {
      viewer.removeAllSurfaces();
      if (showSurface && viewerAny._proteinModel) {
        viewer.addSurface(
          window.$3Dmol.VDW,
          PROTEIN_SURFACE_STYLE,
          { model: viewerAny._proteinModel },
        );
      }
      viewer.render();
    } catch (e) {
      console.warn('Surface sync failed:', e);
    }
  }, [showSurface, isSceneReady, proteinContent]);

  const pocketsLayoutKey = detectedPockets
    .map(
      (p) =>
        `${p.id}:${hiddenPocketIds.includes(p.id) ? 1 : 0}:${p.alpha_sphere_centers?.length ?? 0}`,
    )
    .join('|');
  const pharmacophoreLayoutKey = pharmacophores
    .map((p) => `${p.index}:${p.x}:${p.y}:${p.z}:${p.type}:${p.color}`)
    .join('|');

  const applyPocketSelectionStyle = (viewer: any, selectedId: string | null | undefined) => {
    for (const group of pocketShapeGroupsRef.current) {
      const opacity =
        group.pocketId === selectedId ? POCKET_OPACITY_SELECTED : POCKET_OPACITY_UNSELECTED;
      for (const glShape of group.shapes) {
        stylePocketSphere(glShape, group.color, opacity);
      }
    }
    viewer.render();
  };

  // 3. Shapes (pockets, pharmacophores, fragment markers) — does not touch surfaces
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady) return;

    viewer.removeAllShapes();

    // Pocket visualization
    const parsedBboxLen = parseFloat(bboxLength);
    const POCKET_COLORS = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'];
    pocketShapeGroupsRef.current = [];
    if (showBoxes) {
      if (pocketSelectionMethod === 'detected') {
        // Render pockets as alpha sphere blobs
        detectedPockets.forEach((pocket, pocketIdx) => {
          if (hiddenPocketIds.includes(pocket.id)) return;
          const color = POCKET_COLORS[pocketIdx % POCKET_COLORS.length];
          const alphaSphereCenters = pocket.alpha_sphere_centers;
          const sphereOpacity =
            pocket.id === selectedPocketId ? POCKET_OPACITY_SELECTED : POCKET_OPACITY_UNSELECTED;
          const pocketId = pocket.id;
          const shapes: any[] = [];

          const clickHandler = () => {
            const current = selectedPocketIdRef.current;
            onPocketSelectRef.current?.(current === pocketId ? null : pocketId);
          };

          if (alphaSphereCenters && alphaSphereCenters.length > 0) {
            for (const centerCoords of alphaSphereCenters) {
              const shape = viewer.addSphere({
                center: { x: centerCoords[0], y: centerCoords[1], z: centerCoords[2] },
                radius: DETECTED_POCKET_REFERENCE_RADIUS,
                color,
                alpha: sphereOpacity,
                clickable: true,
                callback: clickHandler,
              });
              stylePocketSphere(shape, color, sphereOpacity);
              shapes.push(shape);
            }
          } else {
            // Fallback: single sphere at pocket center
            const shape = viewer.addSphere({
              center: { x: pocket.center[0], y: pocket.center[1], z: pocket.center[2] },
              radius: parsedBboxLen / 2,
              color,
              alpha: sphereOpacity,
              clickable: true,
              callback: clickHandler,
            });
            stylePocketSphere(shape, color, sphereOpacity);
            shapes.push(shape);
          }
          pocketShapeGroupsRef.current.push({ pocketId, color, shapes });
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

    // Re-add fixed-atom / fragment wireframe spheres (removeAllShapes above clears them)
    (viewer as any)._fragSpheres = [];
    if (fixStructureActive && selectedAtomIndices.length > 0) {
      const b64 = bricsRawSdf ? encodeBase64Unicode(bricsRawSdf) : null;
      const onLig = b64 && ligandContent && b64 === ligandContent;
      const fragModel = onLig ? (viewer as any)._ligandModel : (viewer as any)._refLigandModel;
      if (fragModel) {
        const atoms = fragModel.selectedAtoms({});
        const fixedSet = new Set(selectedAtomIndices);
        for (const atom of atoms) {
          if (!fixedSet.has(atom.index)) continue;
          const shape = viewer.addSphere({
            center: { x: atom.x, y: atom.y, z: atom.z },
            radius: 0.35,
            color: FIXED_ATOM_COLOR,
            wireframe: true,
            linewidth: 1.5,
          });
          (viewer as any)._fragSpheres.push(shape);
        }
      }
    } else if (bricsFragments.length > 1 && selectedFragmentIds.length > 0) {
      const b64 = bricsRawSdf ? encodeBase64Unicode(bricsRawSdf) : null;
      const onLig = b64 && ligandContent && b64 === ligandContent;
      const fragModel = onLig ? (viewer as any)._ligandModel : (viewer as any)._refLigandModel;
      if (fragModel) {
        const atoms = fragModel.selectedAtoms({});
        for (const frag of bricsFragments) {
          if (!selectedFragmentIds.includes(frag.id)) continue;
          for (const atomIdx of frag.atom_indices) {
            const atom = atoms.find((a: any) => a.index === atomIdx);
            if (!atom) continue;
            const shape = viewer.addSphere({
              center: { x: atom.x, y: atom.y, z: atom.z },
              radius: 0.35,
              color: FIXED_ATOM_COLOR,
              wireframe: true,
              linewidth: 1.5,
            });
            (viewer as any)._fragSpheres.push(shape);
          }
        }
      }
    }

    viewer.render();
    if (pocketSelectionMethod === 'detected' && pocketShapeGroupsRef.current.length > 0) {
      applyPocketSelectionStyle(viewer, selectedPocketId);
    }

    // For pharmacophore-only mode, zoom to fit pharmacophores after rendering
    const hasPharmacophoresOnly = pharmacophores.length > 0 && !proteinContent && !ligandContent && !refLigandContent;
    if (hasPharmacophoresOnly) {
      viewer.zoomTo();
    }
  }, [
    isSceneReady,
    pocketsLayoutKey,
    pharmacophoreLayoutKey,
    selectedPharmacophoreIndices,
    pocketSelectionMethod,
    manualCenter,
    bboxLength,
    pharmacophoreTolerance,
    showBoxes,
    bricsFragments,
    selectedFragmentIds,
    selectedAtomIndices,
    fixStructureActive,
    fixStructureMode,
    bricsRawSdf,
    ligandContent,
    refLigandContent,
  ]);

  // Pocket highlight only (after shapes exist + renderedShapeObj is built)
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !isSceneReady || pocketSelectionMethod !== 'detected') return;
    if (pocketShapeGroupsRef.current.length === 0) return;
    applyPocketSelectionStyle(viewer, selectedPocketId);
  }, [selectedPocketId, isSceneReady, pocketSelectionMethod]);

  return (
    <div className="h-full w-full relative">
      <div ref={containerRef} className="h-full w-full" style={{ minHeight: '500px' }} />
      {boxOverlay && (
        <div
          className="absolute pointer-events-none border-2 border-dashed border-primary-500 bg-primary-500/10 z-30"
          style={{
            left: boxOverlay.left,
            top: boxOverlay.top,
            width: boxOverlay.width,
            height: boxOverlay.height,
          }}
        />
      )}
      {fixStructureActive && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 px-3 py-1.5 rounded-full bg-slate-900/75 text-white text-[11px] pointer-events-none">
          Click to toggle · Shift+drag to box-select
        </div>
      )}

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

function applyFixStructureStyles(
  viewer: any,
  model: any,
  fragments: BricsFragment[],
  selectedFragmentIds: number[],
  fixedAtoms: Set<number>,
  mode: FixStructureMode,
  active: boolean,
) {
  viewer.setStyle({ model }, { stick: { radius: 0.15, colorscheme: 'lightgreyCarbon' } });
}

function findFragmentForAtom(atomIndex: number, fragments: BricsFragment[]): number | null {
  for (const frag of fragments) {
    if (frag.atom_indices.includes(atomIndex)) return frag.id;
  }
  return null;
}
