'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { BricsFragment } from '@/types';

export type FixStructureMode = 'fragment' | 'atom';
export type SelectionAction = 'toggle' | 'add' | 'remove';

function indicesToSet(indices: Iterable<number>): Set<number> {
  return new Set(indices);
}

function atomIndicesFromFragments(fragments: BricsFragment[], fragmentIds: number[]): Set<number> {
  const idSet = new Set(fragmentIds);
  const atoms = new Set<number>();
  for (const frag of fragments) {
    if (!idSet.has(frag.id)) continue;
    for (const idx of frag.atom_indices) atoms.add(idx);
  }
  return atoms;
}

function fullySelectedFragmentIds(fragments: BricsFragment[], selectedAtoms: Set<number>): number[] {
  return fragments
    .filter((frag) => frag.atom_indices.length > 0 && frag.atom_indices.every((i) => selectedAtoms.has(i)))
    .map((f) => f.id);
}

function mixedFragmentIds(fragments: BricsFragment[], selectedAtoms: Set<number>): number[] {
  return fragments
    .filter((frag) => {
      const n = frag.atom_indices.filter((i) => selectedAtoms.has(i)).length;
      return n > 0 && n < frag.atom_indices.length;
    })
    .map((f) => f.id);
}

export function useFixedAtomSelection(bricsFragments: BricsFragment[], totalAtomCount: number) {
  const [mode, setMode] = useState<FixStructureMode>('fragment');
  const [selectionAction, setSelectionAction] = useState<SelectionAction>('toggle');
  const [selectedAtomIndices, setSelectedAtomIndices] = useState<Set<number>>(() => new Set());

  const selectedFragmentIds = useMemo(
    () => fullySelectedFragmentIds(bricsFragments, selectedAtomIndices),
    [bricsFragments, selectedAtomIndices]
  );

  const mixedFragmentIdsList = useMemo(
    () => mixedFragmentIds(bricsFragments, selectedAtomIndices),
    [bricsFragments, selectedAtomIndices]
  );

  const fixedCount = selectedAtomIndices.size;

  const fixedAtomIndicesForSubmit = useMemo(
    () => [...selectedAtomIndices].sort((a, b) => a - b),
    [selectedAtomIndices]
  );

  /** Latest indices for job submit (updated every render; safe across async gaps). */
  const fixedAtomIndicesRef = useRef<number[]>(fixedAtomIndicesForSubmit);
  useEffect(() => {
    fixedAtomIndicesRef.current = fixedAtomIndicesForSubmit;
  }, [fixedAtomIndicesForSubmit]);

  const getFixedAtomIndicesForSubmit = useCallback(
    () => [...fixedAtomIndicesRef.current],
    []
  );

  const switchMode = useCallback(
    (newMode: FixStructureMode) => {
      setMode((prev) => {
        if (prev === newMode) return prev;
        // Fragment → atom: selectedAtomIndices already holds union from fragments
        // Atom → fragment: keep atom set; chips show fully-selected fragments only
        return newMode;
      });
    },
    []
  );

  const setSelectedFragmentIds = useCallback(
    (ids: number[]) => {
      const nextFullIds = new Set(ids);
      setSelectedAtomIndices((prev) => {
        const prevFullIds = new Set(fullySelectedFragmentIds(bricsFragments, prev));
        const next = new Set(prev);

        // Only strip atoms from fragments that were fully selected and are now deselected.
        // Partial (mixed) fragment selections from atom mode must be preserved.
        for (const frag of bricsFragments) {
          if (prevFullIds.has(frag.id) && !nextFullIds.has(frag.id)) {
            for (const idx of frag.atom_indices) next.delete(idx);
          }
        }

        // Fully selected fragments always contribute all of their atoms.
        for (const frag of bricsFragments) {
          if (nextFullIds.has(frag.id)) {
            for (const idx of frag.atom_indices) next.add(idx);
          }
        }

        return next;
      });
    },
    [bricsFragments]
  );

  const addFragmentAtoms = useCallback(
    (fragId: number) => {
      const frag = bricsFragments.find((f) => f.id === fragId);
      if (!frag) return;
      setSelectedAtomIndices((prev) => {
        const next = new Set(prev);
        for (const idx of frag.atom_indices) next.add(idx);
        return next;
      });
    },
    [bricsFragments]
  );

  /** Toggle all atoms in a fragment (sidebar chips in atom mode). Clears partial selections too. */
  const toggleFragmentAtoms = useCallback(
    (fragId: number) => {
      const frag = bricsFragments.find((f) => f.id === fragId);
      if (!frag) return;
      setSelectedAtomIndices((prev) => {
        const hasAny = frag.atom_indices.some((i) => prev.has(i));
        const next = new Set(prev);
        if (hasAny) {
          for (const idx of frag.atom_indices) next.delete(idx);
        } else {
          for (const idx of frag.atom_indices) next.add(idx);
        }
        return next;
      });
    },
    [bricsFragments]
  );

  const toggleFragmentByAtom = useCallback(
    (atomIndex: number) => {
      const frag = bricsFragments.find((f) => f.atom_indices.includes(atomIndex));
      if (!frag) return;
      setSelectedAtomIndices((prev) => {
        const allIn = frag.atom_indices.every((i) => prev.has(i));
        const next = new Set(prev);
        if (allIn) {
          for (const i of frag.atom_indices) next.delete(i);
        } else {
          for (const i of frag.atom_indices) next.add(i);
        }
        return next;
      });
    },
    [bricsFragments]
  );

  const applyActionToAtoms = useCallback(
    (indices: number[], action: SelectionAction = selectionAction) => {
      if (indices.length === 0) return;
      setSelectedAtomIndices((prev) => {
        const next = new Set(prev);
        for (const idx of indices) {
          if (action === 'toggle') {
            if (next.has(idx)) next.delete(idx);
            else next.add(idx);
          } else if (action === 'add') {
            next.add(idx);
          } else {
            next.delete(idx);
          }
        }
        return next;
      });
    },
    [selectionAction]
  );

  const toggleFragmentIdsInBox = useCallback(
    (fragmentIds: number[]) => {
      if (fragmentIds.length === 0) return;
      setSelectedAtomIndices((prev) => {
        const next = new Set(prev);
        for (const fragId of fragmentIds) {
          const frag = bricsFragments.find((f) => f.id === fragId);
          if (!frag) continue;
          const allIn = frag.atom_indices.every((i) => prev.has(i));
          if (allIn) {
            for (const i of frag.atom_indices) next.delete(i);
          } else {
            for (const i of frag.atom_indices) next.add(i);
          }
        }
        return next;
      });
    },
    [bricsFragments]
  );

  const clearSelection = useCallback(() => setSelectedAtomIndices(new Set()), []);

  const invertSelection = useCallback(() => {
    if (totalAtomCount <= 0) return;
    setSelectedAtomIndices((prev) => {
      const next = new Set<number>();
      for (let i = 0; i < totalAtomCount; i++) {
        if (!prev.has(i)) next.add(i);
      }
      return next;
    });
  }, [totalAtomCount]);

  const resetSelection = useCallback(() => {
    setSelectedAtomIndices(new Set());
    setMode('fragment');
    setSelectionAction('toggle');
  }, []);

  return {
    mode,
    switchMode,
    selectionAction,
    setSelectionAction,
    selectedAtomIndices,
    setSelectedAtomIndices,
    selectedFragmentIds,
    mixedFragmentIds: mixedFragmentIdsList,
    fixedCount,
    fixedAtomIndicesForSubmit,
    getFixedAtomIndicesForSubmit,
    setSelectedFragmentIds,
    addFragmentAtoms,
    toggleFragmentAtoms,
    toggleFragmentByAtom,
    applyActionToAtoms,
    toggleFragmentIdsInBox,
    clearSelection,
    invertSelection,
    resetSelection,
    indicesToSet,
  };
}
