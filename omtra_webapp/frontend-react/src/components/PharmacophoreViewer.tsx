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

  useEffect(() => {
    if (!containerRef.current || typeof window === 'undefined') return;

    // CRITICAL: Only initialize viewer once (global viewer pattern)
    if (viewerRef.current) {
      // Viewer already exists - check if we need to add/update protein
      const viewer = viewerRef.current;
      const hasProtein = (viewer as any)._hasProtein || false;

      // If protein data is provided but not loaded yet, load it now
      if (proteinB64 && proteinFormat && !hasProtein) {
        try {
          const proteinData = atob(proteinB64);
          viewer.addModel(proteinData, proteinFormat);
          viewer.setStyle({ chain: 'A' }, { cartoon: { color: 'lightblue' } });

          // Add surface for protein
          try {
            const pocketSelection = {
              model: -1,
              within: { distance: 6.0, sel: { model: 0 } },
            };
            viewer.addSurface(window.$3Dmol.VDW, { opacity: 0.6, colorscheme: 'whiteCarbon' }, pocketSelection);
          } catch (err) {
            console.error('Failed to add protein surface:', err);
          }

          (viewer as any)._hasProtein = true;

          // Update camera clipping planes after adding protein
          const viewerAny = viewer as any;
          if (viewerAny.camera) {
            viewerAny.camera.near = 0.1;
            viewerAny.camera.far = 10000;
            viewerAny.camera.updateProjectionMatrix();
          }

          viewer.render();
        } catch (err) {
          console.error('Failed to load protein:', err);
        }
      }
      return; // Viewer already exists, don't re-initialize
    }

    // Dynamically load 3Dmol
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

      // Load ligand
      try {
        // ligandContent is already base64, decode it
        const ligandData = atob(ligandContent);
        viewer.addModel(ligandData, 'sdf');
        viewer.setStyle({ model: 0 }, { stick: { radius: 0.15 } });
      } catch (err) {
        console.error('Failed to load ligand:', err);
        // Try loading as raw content if base64 decode fails
        try {
          viewer.addModel(ligandContent, 'sdf');
          viewer.setStyle({ model: 0 }, { stick: { radius: 0.15 } });
        } catch (err2) {
          console.error('Failed to load ligand as raw content:', err2);
        }
      }

      // Load protein if provided
      if (proteinB64 && proteinFormat) {
        try {
          const proteinData = atob(proteinB64);
          viewer.addModel(proteinData, proteinFormat);
          viewer.setStyle({ chain: 'A' }, { cartoon: { color: 'lightblue' } });

          // Add surface for protein
          try {
            const pocketSelection = {
              model: -1,
              within: { distance: 6.0, sel: { model: 0 } },
            };
            viewer.addSurface(window.$3Dmol.VDW, { opacity: 0.6, colorscheme: 'whiteCarbon' }, pocketSelection);
          } catch (err) {
            console.error('Failed to add protein surface:', err);
          }

          (viewer as any)._hasProtein = true;
        } catch (err) {
          console.error('Failed to load protein:', err);
        }
      }

      // Store sphere shapes for updating
      const sphereShapes: { [key: number]: any } = {};

      // Function to update sphere appearance
      const updateSphereAppearance = (index: number, isSelected: boolean) => {
        const sphereInfo = sphereShapes[index];
        if (!sphereInfo || !viewer) return;

        // Check if internal state matches requested state
        if (sphereInfo.isSelected === isSelected) return;

        // Remove old shape(s)
        if (sphereInfo.shapes) {
          sphereInfo.shapes.forEach((s: any) => { try { viewer.removeShape(s); } catch (e) { } });
        }
        if (sphereInfo.shape) {
          try { viewer.removeShape(sphereInfo.shape); } catch (e) { }
        }

        const pharm = pharmacophores.find((p) => p.index === index);
        if (!pharm) return;

        const center = sphereInfo.center;

        const sphereOptions: any = {
          center,
          radius: 1.0,
          color: sphereInfo.color,
          clickable: true,
          callback: sphereInfo.clickHandler,
        };

        if (isSelected) {
          // Selected: Solid opaque sphere + Always on top
          sphereOptions.alpha = 1.0;
          sphereOptions.wireframe = false;
        } else {
          // Unselected: Wireframe only
          sphereOptions.wireframe = true;
          sphereOptions.linewidth = 1.5;
        }

        const newShape = viewer.addSphere(sphereOptions);

        // Apply always-on-top to ALL pharmacophores
        if (newShape && newShape.material) {
          try {
            const materials = Array.isArray(newShape.material) ? newShape.material : [newShape.material];
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

        // Update tracking
        sphereShapes[index] = {
          shape: newShape,
          shapes: [newShape], // keep array structure just in case
          center: sphereInfo.center,
          color: sphereInfo.color,
          clickHandler: sphereInfo.clickHandler,
          isSelected: isSelected
        };

        viewer.render();
      };

      // Add pharmacophore spheres (store click handlers for later updates)
      pharmacophores.forEach((pharm) => {
        const isSelected = selectionRef.current.has(pharm.index);
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
          // Selected: Solid opaque sphere
          sphereOptions.alpha = 1.0;
          sphereOptions.wireframe = false;
        } else {
          // Unselected: Wireframe
          sphereOptions.wireframe = true;
          sphereOptions.linewidth = 1.5;
        }

        const shape = viewer.addSphere(sphereOptions);

        // Apply always-on-top to ALL pharmacophores
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

        // Store sphere info for updates
        sphereShapes[pharm.index] = {
          shape: shape,
          shapes: [shape],
          center,
          color: pharm.color,
          clickHandler: clickHandler,
          isSelected
        };
      });

      // Only zoom on initial load (zoom to ligand model 0)
      viewer.zoomTo({ model: 0 });

      if (viewerAny.camera) {
        viewerAny.camera.near = 0.01;
        viewerAny.camera.far = 1000000;
        viewerAny.camera.updateProjectionMatrix();
      }

      viewer.render();

      // Store viewer and update function for later use
      (viewer as any).updateSphereAppearance = updateSphereAppearance;
      (viewer as any).sphereShapes = sphereShapes;
    };

    load3Dmol();

    // Don't clean up on unmount - let React handle it naturally
    // The viewer should persist as long as the component is mounted
    return () => {
      // Only cleanup if component is actually unmounting (not just re-rendering)
      // We'll let React handle the DOM cleanup
    };
  }, [ligandContent, proteinB64, proteinFormat, onSelectionChange, pharmacophores]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '500px', position: 'relative' }}
      className="three-d-viewer-container border border-gray-300 rounded-lg overflow-hidden"
    />
  );
}
