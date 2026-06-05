'use client';

import { useEffect, useMemo, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import type { MetricsOptions, SamplingMode } from '@/types';
import { DEFAULT_METRICS_OPTIONS } from '@/types';
import { Loader2, Settings2, X } from 'lucide-react';

const BASIC_COLUMNS = new Set(['sample_name', 'n_atoms', 'is_connected', 'mw', 'logp', 'tpsa']);

const COLUMN_ORDER = [
  'sample_name',
  'n_atoms',
  'is_connected',
  'pb_valid',
  'pb_failing_checks',
  'vina_score',
  'strain',
  'logp',
  'tpsa',
  'mw',
  'clashes',
  'HBAcceptor',
  'HBDonor',
  'Hydrophobic'
];

const PROTEIN_MODES: SamplingMode[] = [
  'Protein-conditioned',
  'Protein+Pharmacophore-conditioned',
  'Rigid Docking',
  'Rigid Docking + Pharmacophore',
];

function allowedMetricColumns(
  metricsOptions: MetricsOptions | undefined,
  samplingMode: SamplingMode,
): Set<string> {
  const opts = metricsOptions ?? DEFAULT_METRICS_OPTIONS;
  const allowed = new Set(BASIC_COLUMNS);
  if (opts.posebusters) {
    allowed.add('pb_valid');
    allowed.add('pb_failing_checks');
  }
  const isProtein = PROTEIN_MODES.includes(samplingMode);
  if (isProtein && opts.posecheck) {
    allowed.add('clashes');
    allowed.add('HBAcceptor');
    allowed.add('HBDonor');
    allowed.add('Hydrophobic');
  }
  if (opts.strain) {
    allowed.add('strain');
  }
  if (isProtein && opts.vina) {
    allowed.add('vina_score');
  }
  return allowed;
}

interface MetricsTableProps {
  jobId: string;
  onRowSelect: (index: number) => void;
  selectedIndex?: number | null;
  samplingMode: SamplingMode;
  metricsOptions?: MetricsOptions;
}

export function MetricsTable({
  jobId,
  onRowSelect,
  selectedIndex,
  samplingMode,
  metricsOptions,
}: MetricsTableProps) {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [showColumnMenu, setShowColumnMenu] = useState(false);
  const [visibleColumns, setVisibleColumns] = useState<Set<string>>(new Set());

  // Sync selectedRow with selectedIndex prop
  useEffect(() => {
    if (selectedIndex !== undefined && selectedIndex !== null) {
      setSelectedRow(selectedIndex);
    }
  }, [selectedIndex]);
  const [sortConfig, setSortConfig] = useState<{ column: string | null; direction: 'asc' | 'desc' }>({
    column: null,
    direction: 'desc',
  });

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const blob = await apiClient.downloadFile(jobId, 'per_molecule_metrics.json');
        const text = await blob.text();
        const data = JSON.parse(text);
        setMetrics(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error('Failed to load metrics:', err);
        setMetrics([]);
      } finally {
        setIsLoading(false);
      }
    };

    loadMetrics();
  }, [jobId]);

  const allowedCols = useMemo(
    () => allowedMetricColumns(metricsOptions, samplingMode),
    [metricsOptions, samplingMode],
  );

  const filteredMetrics = metrics.map((m, originalIndex) => {
    const {
      n_connected_components,
      qed,
      is_invalid,
      smiles,
      Warning,
      PiStacking,
      is_reference,
      ...rest
    } = m;
    
    if (rest.molecular_weight !== undefined) {
      rest.mw = rest.molecular_weight;
      delete rest.molecular_weight;
    }

    const row: Record<string, unknown> = { __originalIndex: originalIndex };
    for (const [key, val] of Object.entries(rest)) {
      if (allowedCols.has(key)) {
        row[key] = val;
      }
    }
    return row;
  });

  const allColumns = Object.keys(filteredMetrics[0] || {}).filter((col) => col !== '__originalIndex');

  const shouldShowPbFailingChecks = useMemo(() => {
    return filteredMetrics.some((m) => {
      const checks = m.pb_failing_checks;
      return Array.isArray(checks) && checks.length > 0;
    });
  }, [filteredMetrics]);

  const availableColumns = useMemo(() => {
    return allColumns.filter((col) => {
      if (col === 'pb_failing_checks' && !shouldShowPbFailingChecks) {
        return false;
      }
      return true;
    });
  }, [allColumns, shouldShowPbFailingChecks]);

  // Initialize visible columns on first load (all columns visible by default)
  useEffect(() => {
    if (availableColumns.length > 0 && visibleColumns.size === 0) {
      setVisibleColumns(new Set(availableColumns));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableColumns]);

  // Get columns to display (respecting visibility settings, but always include sample_name)
  const columns = useMemo(() => {
    const cols = availableColumns.filter((col) => {
      // sample_name is always visible
      if (col === 'sample_name') return true;
      return visibleColumns.has(col);
    });
    return cols.sort((a, b) => {
      const idxA = COLUMN_ORDER.indexOf(a);
      const idxB = COLUMN_ORDER.indexOf(b);
      const posA = idxA !== -1 ? idxA : COLUMN_ORDER.length;
      const posB = idxB !== -1 ? idxB : COLUMN_ORDER.length;
      if (posA !== posB) return posA - posB;
      return a.localeCompare(b);
    });
  }, [availableColumns, visibleColumns]);

  const toggleColumn = (column: string) => {
    setVisibleColumns((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(column)) {
        newSet.delete(column);
      } else {
        newSet.add(column);
      }
      return newSet;
    });
  };

  const sortedMetrics = useMemo(() => {
    if (!sortConfig.column) return filteredMetrics;
    const column = sortConfig.column;
    const direction = sortConfig.direction === 'asc' ? 1 : -1;

    return [...filteredMetrics].sort((a, b) => {
      const valA = a[column];
      const valB = b[column];

      if (valA === undefined || valA === null) return 1;
      if (valB === undefined || valB === null) return -1;

      if (typeof valA === 'number' && typeof valB === 'number') {
        return (valA - valB) * direction;
      }

      const stringA = String(valA);
      const stringB = String(valB);
      return stringA.localeCompare(stringB, undefined, { numeric: true, sensitivity: 'base' }) * direction;
    });
  }, [filteredMetrics, sortConfig]);

  const handleSort = (column: string) => {
    setSortConfig((prev) => {
      if (prev.column === column) {
        return { column, direction: prev.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { column, direction: 'desc' };
    });
  };

  const handleRowClick = (originalIndex: number) => {
    setSelectedRow(originalIndex);
    onRowSelect(originalIndex);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
      </div>
    );
  }

  if (filteredMetrics.length === 0) {
    return (
      <div className="p-6 bg-slate-50/50 rounded-xl text-sm text-slate-600 text-center shadow-sm">
        No metrics available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-900">Metrics</h3>
        <div className="relative">
          <button
            type="button"
            onClick={() => setShowColumnMenu(!showColumnMenu)}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors shadow-sm"
          >
            <Settings2 className="w-4 h-4" />
            Columns
          </button>
          {showColumnMenu && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setShowColumnMenu(false)}
              />
              <div className="absolute right-0 mt-2 w-64 bg-white border border-slate-200/60 rounded-lg shadow-lg z-20 max-h-96 overflow-y-auto">
                <div className="p-3 border-b border-slate-200 flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-slate-900">Show/Hide Columns</h4>
                  <button
                    type="button"
                    onClick={() => setShowColumnMenu(false)}
                    className="text-slate-400 hover:text-slate-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="p-2">
                  {availableColumns.map((col) => {
                    const isRequired = col === 'sample_name';
                    return (
                      <label
                        key={col}
                        className={`flex items-center gap-2 px-2 py-1.5 hover:bg-slate-50 rounded ${isRequired ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'
                          }`}
                      >
                        <input
                          type="checkbox"
                          checked={visibleColumns.has(col) || isRequired}
                          disabled={isRequired}
                          onChange={() => !isRequired && toggleColumn(col)}
                          className="w-4 h-4 text-primary-600 border-slate-300 rounded focus:ring-primary-500 disabled:opacity-50"
                        />
                        <span className="text-sm text-slate-700">
                          {col}
                          {isRequired && <span className="text-xs text-slate-400 ml-1">(required)</span>}
                        </span>
                      </label>
                    );
                  })}
                </div>
                <div className="p-2 border-t border-slate-200">
                  <button
                    type="button"
                    onClick={() => {
                      setVisibleColumns(new Set(availableColumns));
                      setShowColumnMenu(false);
                    }}
                    className="w-full text-sm text-primary-600 hover:text-primary-700 font-medium"
                  >
                    Show All
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
      <div className="text-sm text-slate-600 mb-3 bg-blue-50/70 rounded-xl px-4 py-2 shadow-sm">
        💡 Click on any row to view it above
      </div>
      <div className="overflow-x-auto rounded-xl shadow-sm bg-white">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              {columns.map((col) => {
                const isSorted = sortConfig.column === col;
                return (
                  <th
                    key={col}
                    scope="col"
                    aria-sort={
                      isSorted ? (sortConfig.direction === 'asc' ? 'ascending' : 'descending') : 'none'
                    }
                    className={`px-4 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-slate-700 ${col === 'sample_name' ? 'sticky left-0 bg-slate-50 z-10' : ''
                      }`}
                  >
                    <button
                      type="button"
                      onClick={() => handleSort(col)}
                      className="flex items-center gap-1.5 hover:text-primary-700 focus:outline-none transition-colors"
                    >
                      {col}
                      <span className="text-[10px] text-slate-400">
                        {isSorted ? (sortConfig.direction === 'asc' ? '↑' : '↓') : '↕'}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {sortedMetrics.map((row) => {
              const originalIndex = (row.__originalIndex as number) ?? 0;
              const isSelected = selectedRow === originalIndex;
              const isRef = metrics[originalIndex]?.is_reference === true;
              return (
                <tr
                  key={originalIndex}
                  onClick={() => handleRowClick(originalIndex)}
                  className={`cursor-pointer transition-colors ${isSelected
                    ? 'bg-primary-100/70 hover:bg-primary-100/70 border-l-4 border-primary-600'
                    : isRef
                      ? 'bg-amber-50/70 hover:bg-amber-100/50'
                      : 'hover:bg-slate-50'
                    }`}
                  style={isSelected ? { borderLeftColor: '#0284c7' } : undefined}
                >
                  {columns.map((col) => (
                    <td
                      key={col}
                      className={`px-4 py-3.5 text-sm text-slate-900 ${col === 'sample_name'
                        ? `sticky left-0 z-10 font-mono ${isSelected ? 'bg-primary-100/70' : isRef ? 'bg-amber-50/70' : 'bg-white'}`
                        : isSelected ? 'bg-primary-100/70' : ''
                        }`}
                    >
                      {row[col] !== null && row[col] !== undefined
                        ? typeof row[col] === 'number'
                          ? col === 'mw'
                            ? (row[col] as number).toFixed(2)
                            : Number.isInteger(row[col]) || (row[col] as number) % 1 === 0
                              ? String(Math.round(row[col] as number))
                              : (row[col] as number).toFixed(4)
                          : Array.isArray(row[col])
                            ? row[col].length > 0
                              ? row[col].join(', ')
                              : '-'
                            : String(row[col])
                        : '-'}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}


