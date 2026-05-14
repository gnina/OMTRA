'use client';

import { useCallback } from 'react';
import type { BricsFragment } from '@/types';
import { Info, CheckSquare, Square } from 'lucide-react';

const FRAGMENT_COLORS = [
  '#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c',
  '#0891b2', '#ca8a04', '#db2777', '#4f46e5', '#65a30d',
];

interface FragmentSelectorProps {
  fragments: BricsFragment[];
  selectedFragmentIds: number[];
  onSelectionChange: (ids: number[]) => void;
}

export function FragmentSelector({
  fragments,
  selectedFragmentIds,
  onSelectionChange,
}: FragmentSelectorProps) {
  const isSelected = useCallback(
    (id: number) => selectedFragmentIds.includes(id),
    [selectedFragmentIds]
  );

  const toggleFragment = useCallback(
    (id: number) => {
      onSelectionChange(
        isSelected(id)
          ? selectedFragmentIds.filter((fid) => fid !== id)
          : [...selectedFragmentIds, id]
      );
    },
    [selectedFragmentIds, isSelected, onSelectionChange]
  );

  const selectAll = useCallback(() => {
    onSelectionChange(fragments.map((f) => f.id));
  }, [fragments, onSelectionChange]);

  const clearAll = useCallback(() => {
    onSelectionChange([]);
  }, [onSelectionChange]);

  if (fragments.length <= 1) {
    return (
      <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-700 flex items-start gap-2">
        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>
          No BRICS-decomposable bonds found. Fragment selection is not available.
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-2.5">
      <div className="p-2.5 bg-blue-50/60 border border-blue-200 rounded-xl text-[11px] text-blue-700 flex items-start gap-2">
        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>
          Click fragments in the viewer or toggle below. Selected fragments stay fixed during generation.
        </span>
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        {fragments.map((frag) => {
          const sel = isSelected(frag.id);
          const color = FRAGMENT_COLORS[frag.id % FRAGMENT_COLORS.length];
          return (
            <button
              key={frag.id}
              onClick={() => toggleFragment(frag.id)}
              className={`
                inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium
                border transition-all duration-150
                ${sel
                  ? 'border-current shadow-sm'
                  : 'border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:bg-slate-50'
                }
              `}
              style={sel ? { color, borderColor: color, backgroundColor: `${color}10` } : undefined}
            >
              {sel ? (
                <CheckSquare className="w-3 h-3" />
              ) : (
                <Square className="w-3 h-3" />
              )}
              Frag {frag.id}
              <span className="text-[10px] opacity-70">({frag.num_atoms})</span>
            </button>
          );
        })}

        <span className="mx-1 text-slate-300">|</span>

        <button
          onClick={selectAll}
          className="text-[10px] text-primary-600 hover:text-primary-800 font-semibold transition-colors"
        >
          All
        </button>
        <button
          onClick={clearAll}
          className="text-[10px] text-slate-500 hover:text-slate-700 font-semibold transition-colors"
        >
          None
        </button>
      </div>

      {selectedFragmentIds.length > 0 && (
        <p className="text-[10px] text-slate-500">
          {selectedFragmentIds.length} of {fragments.length} fragments fixed
        </p>
      )}
    </div>
  );
}
