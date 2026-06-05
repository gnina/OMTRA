'use client';

import { ChevronDown, ChevronRight, Loader2 } from 'lucide-react';
import { FragmentSelector } from '@/components/FragmentSelector';
import type { BricsFragment } from '@/types';
import type { FixStructureMode, SelectionAction } from '@/hooks/useFixedAtomSelection';

interface FixedStructurePanelProps {
  expanded: boolean;
  onExpandedChange: (v: boolean) => void;
  bricsLoading: boolean;
  bricsRawSdf: string | null;
  bricsFragments: BricsFragment[];
  mode: FixStructureMode;
  onModeChange: (mode: FixStructureMode) => void;
  selectionAction: SelectionAction;
  onSelectionActionChange: (action: SelectionAction) => void;
  selectedFragmentIds: number[];
  mixedFragmentIds: number[];
  onFragmentSelectionChange: (ids: number[]) => void;
  onAddFragment: (fragId: number) => void;
  onToggleFragment: (fragId: number) => void;
  fixedCount: number;
  totalAtomCount: number;
  onClear: () => void;
  onInvert: () => void;
}

export function FixedStructurePanel({
  expanded,
  onExpandedChange,
  bricsLoading,
  bricsRawSdf,
  bricsFragments,
  mode,
  onModeChange,
  selectionAction,
  onSelectionActionChange,
  selectedFragmentIds,
  mixedFragmentIds,
  onFragmentSelectionChange,
  onAddFragment,
  onToggleFragment,
  fixedCount,
  totalAtomCount,
  onClear,
  onInvert,
}: FixedStructurePanelProps) {
  const allFixed = totalAtomCount > 0 && fixedCount >= totalAtomCount;

  return (
    <div className="mt-3 border border-slate-200 rounded-xl overflow-hidden">
      <button
        type="button"
        onClick={() => onExpandedChange(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <span className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
          Fix atoms
          <span className="text-[10px] font-normal text-slate-400">(optional)</span>
          {fixedCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 bg-primary-100 text-primary-700 rounded-full text-[10px] font-bold">
              {fixedCount}
            </span>
          )}
        </span>
        {expanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
        )}
      </button>
      {expanded && (
        <div className="p-3 border-t border-slate-200 space-y-3">
          <p className="text-[11px] text-slate-600">
            Fixed atoms keep the same position during generation.
          </p>

          <div className="flex rounded-lg border border-slate-200 p-0.5 bg-slate-50">
            <button
              type="button"
              onClick={() => onModeChange('fragment')}
              className={`flex-1 px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${mode === 'fragment' ? 'bg-white text-primary-700 shadow-sm' : 'text-slate-500'
                }`}
            >
              By fragment
            </button>
            <button
              type="button"
              onClick={() => onModeChange('atom')}
              className={`flex-1 px-2 py-1 text-[11px] font-medium rounded-md transition-colors ${mode === 'atom' ? 'bg-white text-primary-700 shadow-sm' : 'text-slate-500'
                }`}
            >
              By atom
            </button>
          </div>

          {mode === 'atom' && (
            <div className="space-y-1">
              <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Selection action</p>
              <div className="flex gap-1">
                {(['toggle', 'add', 'remove'] as SelectionAction[]).map((action) => (
                  <button
                    key={action}
                    type="button"
                    onClick={() => onSelectionActionChange(action)}
                    className={`px-2 py-1 rounded-lg text-[11px] font-medium capitalize border transition-colors ${selectionAction === action
                      ? 'border-primary-300 bg-primary-50 text-primary-800'
                      : 'border-slate-200 text-slate-500 hover:bg-slate-50'
                      }`}
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          )}

          {bricsLoading ? (
            <div className="flex items-center gap-2 text-xs text-slate-500 py-4 justify-center">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Analyzing fragments...
            </div>
          ) : bricsRawSdf ? (
            <FragmentSelector
              fragments={bricsFragments}
              selectedFragmentIds={selectedFragmentIds}
              mixedFragmentIds={mixedFragmentIds}
              onSelectionChange={onFragmentSelectionChange}
              onAddFragment={onAddFragment}
              onToggleFragment={onToggleFragment}
              chipMode={mode === 'atom' ? 'add' : 'toggle'}
            />
          ) : null}

          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-600 font-medium">
              {fixedCount} atom{fixedCount !== 1 ? 's' : ''} fixed
            </span>
            <div className="flex gap-2">
              <button type="button" onClick={onClear} className="text-slate-500 hover:text-slate-800 font-semibold">
                Clear
              </button>
              {totalAtomCount > 0 && (
                <button type="button" onClick={onInvert} className="text-slate-500 hover:text-slate-800 font-semibold">
                  Invert
                </button>
              )}
            </div>
          </div>

          {allFixed && (
            <p className="text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1.5">
              All atoms fixed — nothing left to generate.
            </p>
          )}
        </div>
      )}
    </div>
  );
}