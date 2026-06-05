'use client';

export function HelpTab() {
  return (
    <div className="prose max-w-none">
      <div className="mb-8 border-b border-slate-200 pb-6 text-center">
        <h2 className="text-4xl font-extrabold text-slate-900 mb-2 tracking-tight">
          OMTRA Help Center
        </h2>
      </div>

      <div className="space-y-10">
        {/* 1. Workflow Modes */}
        <section className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
          <h3 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-primary-100 text-primary-600 rounded-lg text-sm">01</span>
            Workflow Modes
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                🚀 De Novo Design
              </h4>
              <p className="text-sm text-slate-600">
                Generate new chemical structures from scratch.
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                🧬 Docking
              </h4>
              <p className="text-sm text-slate-600">
                Dock existing ligands into a protein pocket.
              </p>
            </div>
          </div>
        </section>

        {/* 2. Pocket Selection */}
        <section className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
          <h3 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-emerald-100 text-emerald-600 rounded-lg text-sm">02</span>
            Defining the Pocket
          </h3>
          <ul className="space-y-4">
            <li className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold text-sm">A</div>
              <div>
                <strong className="text-slate-900">Reference Ligand</strong>
                <p className="text-sm text-slate-600 mt-1">
                  Upload a ligand file (.sdf). The pocket is defined within 8Å of this ligand.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold text-sm">B</div>
              <div>
                <strong className="text-slate-900">Detected Pockets</strong>
                <p className="text-sm text-slate-600 mt-1">
                  Upload a protein to auto-detect binding sites using{' '}
                  <a
                    href="https://github.com/cch1999/pocketeer"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary-600 hover:text-primary-700 underline font-medium"
                  >
                    Pocketeer
                  </a>
                  . Select one to define the pocket.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold text-sm">C</div>
              <div>
                <strong className="text-slate-900">Manual Coordinates</strong>
                <p className="text-sm text-slate-600 mt-1">
                  Manually set X, Y, Z center and box size.
                </p>
              </div>
            </li>
          </ul>
        </section>

        {/* 3. Sampling Conditions */}
        <section className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
          <h3 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-amber-100 text-amber-600 rounded-lg text-sm">03</span>
            Sampling
          </h3>
          <div className="space-y-4 text-slate-700">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2">Adding pharmacophore constraints</h4>
              <ul className="list-disc pl-5 text-sm space-y-1">
                <li><strong>Upload a pharmacophore file</strong>: Upload an SDF file and select pharmacophore spheres interactively. Alternatively, upload a JSON file with pharmacophore features exported from <a href="https://pharmit.csb.pitt.edu" target="_blank" rel="noopener noreferrer" className="text-primary-600 underline">Pharmit</a> or using <a href="https://github.com/gnina/OMTRA/blob/main/omtra/scripts/mol2pharm.py" target="_blank" rel="noopener noreferrer" className="text-primary-600 underline">this script</a>.</li>
                <li><strong>Select constraints</strong>: Pharmacophores appear as wireframe spheres in the 3D viewer. Click any sphere to select it as a conditioning constraint (solid = selected). We recommend selecting fewer than 8.</li>
              </ul>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2">Fix structure (partial generation)</h4>
              <ul className="list-disc pl-5 text-sm space-y-1">
                <li><strong>By fragment or by atom</strong>: For protein-conditioned de novo and rigid docking, open <em>Fix structure</em> on the reference/docking SDF. Use BRICS fragments for quick scaffold locking, or switch to per-atom mode for arbitrary subsets.</li>
                <li><strong>3D selection</strong>: Click to toggle; hold <kbd className="px-1 py-0.5 bg-slate-200 rounded text-xs">Shift</kbd> and drag to box-select.</li>
                <li><strong>Results</strong>: Fixed atoms are highlighted in amber in the output 3D viewer.</li>
              </ul>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2">Key Parameters</h4>
              <ul className="list-disc pl-5 text-sm space-y-1">
                <li><strong>Sampling Steps</strong>: Number of integration steps. Higher values improve quality (we used 200 in our paper).</li>
                <li><strong>Atom Count Distribution</strong>: Sets the distribution of atom counts for generated molecules. Auto-populated when a reference ligand is used to define the pocket.</li>
              </ul>
            </div>
          </div>
        </section>

        {/* 4. Output Data */}
        <section className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm">
          <h3 className="text-2xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-purple-100 text-purple-600 rounded-lg text-sm">04</span>
            Job Output
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                📊 Metrics Table
              </h4>
              <p className="text-sm text-slate-600">
                Each generated molecule is scored with drug-likeness properties (QED, LogP, molecular weight) and, for protein-conditioned jobs, docking metrics (Vina score, clashes, HB interactions). Click any row to load that molecule in the 3D viewer.
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                🧪 3D Viewer &amp; Downloads
              </h4>
              <p className="text-sm text-slate-600">
                View generated molecules in 3D alongside the protein and binding pocket. Download individual molecules as SDF files, or view 2D interaction diagrams (PoseView) for protein-conditioned jobs.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
