'use client';

export function HelpTab() {
  return (
    <div className="prose max-w-none">
      <div className="mb-8 border-b border-slate-200 pb-6 text-center">
        <h2 className="text-4xl font-extrabold text-slate-900 mb-2 tracking-tight">
          OMTRA Help Center
        </h2>
        <p className="text-slate-500 text-lg">Master the sampling and docking workflows</p>
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
                Generate new chemical structures from scratch or matching specific constraints.
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                🧬 Docking
              </h4>
              <p className="text-sm text-slate-600">
                Dock existing ligands into a protein pocket using GNINA minimization.
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
                <strong className="text-slate-900">Detected Pockets</strong>
                <p className="text-sm text-slate-600 mt-1">
                  Upload a protein to auto-detect binding sites (yellow boxes). Click a box to select it.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-bold text-sm">B</div>
              <div>
                <strong className="text-slate-900">Reference Ligand</strong>
                <p className="text-sm text-slate-600 mt-1">
                  Upload a simplified ligand file (.sdf). The pocket is defined within 8Å of this ligand.
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
            Sampling & Pharmacophores
          </h3>
          <div className="space-y-4 text-slate-700">
            <p className="text-sm">
              You can constrain generation using pharmacophores derived from a reference ligand.
            </p>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2">3D Viewer Interaction</h4>
              <ul className="list-disc pl-5 text-sm space-y-1">
                <li><strong>Unselected Pharmacophores</strong>: Displayed as <strong>Wireframe</strong> spheres.</li>
                <li><strong>Selected Pharmacophores</strong>: Click to select. We recommend selecting less than 8 pharmacophores as conditioning information.</li>
                <li><strong>Visibility</strong>: All pharmacophores are &quot;Always On Top&quot; and visible through the protein surface.</li>
              </ul>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2">Key Parameters</h4>
              <ul className="list-disc pl-5 text-sm space-y-1">
                <li><strong>Sampling Steps</strong>: Controls denoising iterations. Higher values (e.g., 200) improve quality.</li>
                <li><strong>Pharmacophores</strong>: Select a subset (&lt; 8 recommended) as structural constraints.</li>
                <li><strong>Atom Distribution</strong>: Sets the target element composition for the generated molecules.</li>
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
                📄 Generated Molecules
              </h4>
              <p className="text-sm text-slate-600">
                Download top-ranked molecules in SDF format. Results include confidence scores and property predictions.
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
              <h4 className="font-bold text-slate-900 mb-2 flex items-center gap-2">
                📊 Analysis
              </h4>
              <p className="text-sm text-slate-600">
                View 2D interaction diagrams (PoseView) and 3D binding poses directly in the browser.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
