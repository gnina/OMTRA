'use client';

import { useEffect, useRef } from 'react';

declare global {
  interface Window {
    $3Dmol: any;
  }
}

interface PharmacophoreViewerProps {
  ligandContent: string; // base64
  pharmacophores: Array<{
    index: number;
    type: string;
    position: [number, number, number];
    color: string;
    selected: boolean;
  }>;
  selectedIndices: number[];
  onSelectionChange: (indices: number[]) => void;
  proteinB64?: string;
  proteinFormat?: string;
}

export function PharmacophoreViewer({
  ligandContent,
  pharmacophores,
  selectedIndices,
  onSelectionChange,
  proteinB64,
  proteinFormat,
}: PharmacophoreViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const selectionRef = useRef<Set<number>>(new Set(selectedIndices));

  useEffect(() => {
    selectionRef.current = new Set(selectedIndices);
    // Update spheres when selection changes externally
    if (viewerRef.current && (viewerRef.current as any).updateSphereAppearance && (viewerRef.current as any).sphereShapes) {
      const updateFn = (viewerRef.current as any).updateSphereAppearance;
      const sphereShapes = (viewerRef.current as any).sphereShapes;
      Object.keys(sphereShapes).forEach((idxStr) => {
        const idx = parseInt(idxStr);
        // We pass the isSelected state from selectedIndices prop
        // The updateSphereAppearance function inside the viewer will check against internal state
        updateFn(idx, selectedIndices.includes(idx));
      });
    }
  }, [selectedIndices]);

  // 1. Initialize Viewer
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

      if (!containerRef.current || viewerRef.current) return;

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
      viewer.render();
    };

    load3Dmol();
  }, []);

  // 2. Handle Model and Shape Updating
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || typeof window === 'undefined') return;
    const viewerAny = viewer as any;

    // A. Handle Ligand
    if (ligandContent) {
      try {
        const ligandData = atob(ligandContent);
        if (viewerAny._ligandData !== ligandData) {
          if (viewerAny._ligandModel) viewer.removeModel(viewerAny._ligandModel);
          const model = viewer.addModel(ligandData, 'sdf');
          viewer.setStyle({ model: model.getID() }, { stick: { radius: 0.15 } });
          viewerAny._ligandModel = model;
          viewerAny._ligandData = ligandData;
          viewer.zoomTo({ model });
        }
      } catch (err) {
        console.error('Failed to load ligand:', err);
      }
    } else if (viewerAny._ligandModel) {
      viewer.removeModel(viewerAny._ligandModel);
      viewerAny._ligandModel = undefined;
      viewerAny._ligandData = undefined;
      viewer.render();
    }

    // B. Handle Protein
    if (proteinB64 && proteinFormat) {
      try {
        const proteinData = atob(proteinB64);
        if (viewerAny._proteinData !== proteinData) {
          if (viewerAny._proteinModel) {
            viewer.removeModel(viewerAny._proteinModel);
            viewer.removeAllSurfaces();
          }
          const model = viewer.addModel(proteinData, proteinFormat);
          viewer.setStyle({ chain: 'A' }, { cartoon: { color: 'lightblue' } });
          viewerAny._proteinModel = model;
          viewerAny._proteinData = proteinData;

          // Add surface
          try {
            const pocketSelection = { model: -1, within: { distance: 6.0, sel: { model: viewerAny._ligandModel || 0 } } };
            viewer.addSurface(window.$3Dmol.VDW, { opacity: 0.6, colorscheme: 'whiteCarbon' }, pocketSelection);
          } catch (e) { }
        }
      } catch (err) {
        console.error('Failed to load protein:', err);
      }
    } else if (viewerAny._proteinModel) {
      viewer.removeModel(viewerAny._proteinModel);
      viewer.removeAllSurfaces();
      viewerAny._proteinModel = undefined;
      viewerAny._proteinData = undefined;
      viewer.render();
    }

    // C. Handle Pharmacophores
    viewer.removeAllShapes();
    const sphereShapes: { [key: number]: any } = {};

    const updateSphereAppearance = (index: number, isSelected: boolean) => {
      const sphereInfo = sphereShapes[index];
      if (!sphereInfo) return;
      if (sphereInfo.isSelected === isSelected) return;

      if (sphereInfo.shape) {
        try { viewer.removeShape(sphereInfo.shape); } catch (e) { }
      }

      const pharm = pharmacophores.find((p) => p.index === index);
      if (!pharm) return;

      const sphereOptions: any = {
        center: sphereInfo.center,
        radius: 1.0,
        color: pharm.color,
        clickable: true,
        callback: sphereInfo.clickHandler,
      };

      if (isSelected) {
        sphereOptions.alpha = 1.0;
        sphereOptions.wireframe = false;
      } else {
        sphereOptions.wireframe = true;
        sphereOptions.linewidth = 1.5;
      }

      const newShape = viewer.addSphere(sphereOptions);
      if (newShape && newShape.material) {
        try {
          const mats = Array.isArray(newShape.material) ? newShape.material : [newShape.material];
          mats.forEach((mat: any) => {
            mat.depthTest = false;
            mat.depthWrite = false;
            mat.transparent = true;
            if (isSelected) mat.opacity = 1.0;
          });
        } catch (e) { }
      }

      sphereShapes[index].shape = newShape;
      sphereShapes[index].isSelected = isSelected;
      viewer.render();
    };

    pharmacophores.forEach((pharm) => {
      const isSelected = selectedIndices.includes(pharm.index);
      const clickHandler = () => {
        const newSelection = new Set(selectionRef.current);
        if (newSelection.has(pharm.index)) newSelection.delete(pharm.index);
        else newSelection.add(pharm.index);
        selectionRef.current = newSelection;
        onSelectionChange(Array.from(newSelection));
        updateSphereAppearance(pharm.index, newSelection.has(pharm.index));
      };

      const center = { x: pharm.position[0], y: pharm.position[1], z: pharm.position[2] };
      const sphereOptions: any = {
        center,
        radius: 1.0,
        color: pharm.color,
        clickable: true,
        callback: clickHandler
      };

      if (isSelected) {
        sphereOptions.alpha = 1.0;
        sphereOptions.wireframe = false;
      } else {
        sphereOptions.wireframe = true;
        sphereOptions.linewidth = 1.5;
      }

      const shape = viewer.addSphere(sphereOptions);
      if (shape && shape.material) {
        try {
          const mats = Array.isArray(shape.material) ? shape.material : [shape.material];
          mats.forEach((mat: any) => {
            mat.depthTest = false;
            mat.depthWrite = false;
            mat.transparent = true;
            if (isSelected) mat.opacity = 1.0;
          });
        } catch (e) { }
      }

      sphereShapes[pharm.index] = { shape, center, isSelected, clickHandler };
    });

    viewerAny.updateSphereAppearance = updateSphereAppearance;
    viewerAny.sphereShapes = sphereShapes;
    viewer.render();
  }, [ligandContent, proteinB64, proteinFormat, onSelectionChange, pharmacophores, selectedIndices]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '500px', position: 'relative' }}
      className="three-d-viewer-container border border-gray-300 rounded-lg overflow-hidden"
    />
  );
}
