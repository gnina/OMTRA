'use client';

import type { MetricsOptions, SamplingMode } from '@/types';

const PROTEIN_MODES: SamplingMode[] = [
  'Protein-conditioned',
  'Protein+Pharmacophore-conditioned',
  'Rigid Docking',
  'Rigid Docking + Pharmacophore',
];

interface MetricsOptionsFormProps {
  samplingMode: SamplingMode;
  options: MetricsOptions;
  onChange: (options: MetricsOptions) => void;
}

export function MetricsOptionsForm({ samplingMode, options, onChange }: MetricsOptionsFormProps) {
  const isProteinInvolving = PROTEIN_MODES.includes(samplingMode);

  const toggle = (key: keyof MetricsOptions) => {
    onChange({ ...options, [key]: !options[key] });
  };

  return (
    <div className="border-t border-slate-200/60 pt-6">
      <h4 className="text-sm font-semibold text-slate-700 mb-3">Evaluation Metrics</h4>
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={options.posebusters}
            onChange={() => toggle('posebusters')}
            className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
          />
          PoseBusters validity
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={options.strain}
            onChange={() => toggle('strain')}
            className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
          />
          Strain energy
        </label>
        {isProteinInvolving && (
          <>
            <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.posecheck}
                onChange={() => toggle('posecheck')}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              PoseCheck (clashes + interaction counts)
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.vina}
                onChange={() => toggle('vina')}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              Vina score (GNINA)
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-700 cursor-pointer">
              <input
                type="checkbox"
                checked={options.poseview}
                onChange={() => toggle('poseview')}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              PoseView interaction diagram
            </label>
          </>
        )}
      </div>
    </div>
  );
}
