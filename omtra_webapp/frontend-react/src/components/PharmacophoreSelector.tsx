'use client';

import { useCallback } from 'react';
import { Info, CheckSquare, Square } from 'lucide-react';

const PHARM_COLORS: Record<string, string> = {
  Aromatic: '#9333ea',
  HydrogenDonor: '#2563eb',
  HydrogenAcceptor: '#f97316',
  PositiveIon: '#ca8a04',
  NegativeIon: '#0891b2',
  Hydrophobic: '#65a30d',
  Halogen: '#06b6d4',
};

interface PharmFeature {
  type: string;
  position: [number, number, number];
}

interface PharmacophoreSelectorProps {
  pharmacophores: PharmFeature[];
  selectedIndices: number[];
  onSelectionChange: (indices: number[]) => void;
}

export function PharmacophoreSelector({
  pharmacophores,
  selectedIndices,
  onSelectionChange,
}: PharmacophoreSelectorProps) {
  const isSelected = useCallback(
    (idx: number) => selectedIndices.includes(idx),
    [selectedIndices],
  );

  const toggle = useCallback(
    (idx: number) => {
      onSelectionChange(
        isSelected(idx)
          ? selectedIndices.filter((i) => i !== idx)
          : [...selectedIndices, idx],
      );
    },
    [selectedIndices, isSelected, onSelectionChange],
  );

  const selectAll = useCallback(() => {
    onSelectionChange(pharmacophores.map((_, i) => i));
  }, [pharmacophores, onSelectionChange]);

  const clearAll = useCallback(() => {
    onSelectionChange([]);
  }, [onSelectionChange]);

  if (pharmacophores.length === 0) return null;

  return (
    <div className="space-y-2.5 mt-2">
      <div className="p-2.5 bg-blue-50/60 border border-blue-200 rounded-xl text-[11px] text-blue-700 flex items-start gap-2">
        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>Click spheres in the 3D viewer or toggle features below.</span>
      </div>

      <div className="flex items-center gap-1.5 flex-wrap">
        {pharmacophores.map((pharm, idx) => {
          const sel = isSelected(idx);
          const color = PHARM_COLORS[pharm.type] || '#64748b';
          return (
            <button
              key={idx}
              type="button"
              onClick={() => toggle(idx)}
              className={`
                inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium
                border transition-all duration-150
                ${sel
                  ? 'border-current shadow-sm'
                  : 'border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:bg-slate-50'
                }
              `}
              style={sel ? { color: color, borderColor: color, backgroundColor: `${color}20` } : undefined}
            >
              {sel ? (
                <CheckSquare className="w-3 h-3" />
              ) : (
                <Square className="w-3 h-3" style={{ color }} />
              )}
              {pharm.type} #{idx + 1}
            </button>
          );
        })}
      </div>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={selectAll}
          className="text-[10px] font-semibold text-primary-600 hover:text-primary-700"
        >
          Select all
        </button>
        <button
          type="button"
          onClick={clearAll}
          className="text-[10px] font-semibold text-slate-500 hover:text-slate-700"
        >
          Clear all
        </button>
      </div>
    </div>
  );
}
