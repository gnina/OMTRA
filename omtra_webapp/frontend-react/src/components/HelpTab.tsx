'use client';

export function HelpTab() {
  return (
    <div className="prose max-w-none">
      <div className="mb-8">
        <h2 className="text-3xl font-semibold text-slate-900 mb-2">
          How to Use OMTRA Molecule Sampler
        </h2>
        <p className="text-slate-600">Step-by-step guide</p>
      </div>

      <div className="space-y-8">
        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">1. Set Parameters</h3>
          <ul className="list-disc list-inside space-y-2 text-slate-700">
            <li>
              <strong>Sampling Mode</strong>: Choose from:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                <li><strong>Unconditional</strong>: Generate molecules without constraints</li>
                <li><strong>Pharmacophore-conditioned</strong>: Generate molecules matching specific pharmacophore features</li>
                <li><strong>Protein-conditioned</strong>: Generate molecules for a specific protein binding site</li>
                <li><strong>Protein+Pharmacophore-conditioned</strong>: Combine protein and pharmacophore constraints</li>
              </ul>
            </li>
            <li>
              <strong>Random Seed</strong>: Set for reproducible results (0 to 2³¹-1). Use the same seed to regenerate identical molecules.
            </li>
            <li>
              <strong>Number of Samples</strong>: Number of molecules to generate (1-100).
            </li>
            <li>
              <strong>Sampling Steps</strong>: Recommended: 100-200
            </li>
            <li>
              <strong>Atom Count Distribution (Optional)</strong>: Control the size of generated molecules:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                <li>Mean: Average number of atoms per molecule (minimum 4)</li>
                <li>Standard Deviation: Variability in atom count (minimum 0.1)</li>
              </ul>
            </li>
          </ul>
        </section>

        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">2. Upload Input Files</h3>
          <div className="space-y-3 text-slate-700">
            <div>
              <strong>Unconditional Mode</strong>: No files required
            </div>
            <div>
              <strong>Pharmacophore-conditioned Mode</strong>:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                <li>Upload <strong>XYZ pharmacophore files</strong> directly, or</li>
                <li>Upload <strong>SDF ligand files</strong> to automatically extract pharmacophore features</li>
                <li>Accepted formats: .xyz, .sdf</li>
              </ul>
            </div>
            <div>
              <strong>Protein-conditioned Mode</strong>:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                <li>Requires <strong>both</strong> a protein structure file (PDB or CIF) and a reference ligand file (SDF)</li>
                <li>The ligand is used to identify the binding pocket location</li>
                <li>Accepted formats: .pdb, .cif, .sdf</li>
              </ul>
            </div>
            <div>
              <strong>Protein+Pharmacophore-conditioned Mode</strong>:
              <ul className="list-disc list-inside ml-6 mt-1 space-y-1">
                <li>Upload a protein structure file (PDB or CIF)</li>
                <li>Upload either pharmacophore XYZ files or ligand SDF files (for extraction)</li>
                <li>Accepted formats: .pdb, .cif, .xyz, .sdf</li>
              </ul>
            </div>
            <div className="mt-3 p-3 bg-blue-50 rounded-lg text-sm">
              <strong>File Limits:</strong> Maximum 3 files, 25MB per file
            </div>
          </div>
        </section>

        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">3. Select Pharmacophore Features</h3>
          <p className="text-slate-700 mb-2">
            If you uploaded an SDF file for pharmacophore extraction, the system will automatically detect 
            pharmacophore features. You can then interactively select which features to use:
          </p>
          <ul className="list-disc list-inside space-y-2 text-slate-700">
            <li>Click on spheres in the viewer to select/deselect features</li>
            <li>You must select at least one feature before submitting</li>
          </ul>
          <div className="mt-3 p-3 bg-amber-50 rounded-lg text-sm text-amber-800">
            <strong>Note:</strong> If you upload XYZ files directly, pharmacophore selection is not needed as 
            all features in the file will be used.
          </div>
        </section>

        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">4. Configure Job Settings (Optional)</h3>
          <ul className="list-disc list-inside space-y-2 text-slate-700">
            <li>
              <strong>Custom Job ID</strong>: Optionally specify a custom identifier for your job. 
              If not provided, a unique ID will be automatically generated.
            </li>
          </ul>
        </section>

        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">5. Submit Job</h3>
          <p className="text-slate-700 mb-2">
            Click &quot;🚀 Run Sampling&quot; to submit your job.
          </p>
          <ul className="list-disc list-inside space-y-2 text-slate-700">
            <li>For 10 samples, protein conditioned tasks take ~2-3 minutes to complete, unconditional tasks take &lt;1 minute.</li>
          </ul>

        </section>

        <section className="bg-slate-50/70 rounded-2xl p-6 shadow-sm">
          <h3 className="text-xl font-semibold text-slate-900 mb-3">6. View Results</h3>
          <ul className="list-disc list-inside space-y-2 text-slate-700">
            <li><strong>Job List</strong>: Monitor all your jobs and their status in the Jobs tab</li>
            <li><strong>3D Molecular Viewer</strong>: Interactively view generated molecules</li>
            <li><strong>Poseview Diagram</strong>: For protein-conditioned modes, view 2D interaction diagrams showing how the molecule interacts with the protein binding site. Note: The diagram may not generate if PoseView cannot detect any interactions or if the molecule is not properly positioned in the binding site</li>
            <li><strong>Navigation</strong>: Use previous/next buttons or sample selector to browse through generated molecules</li>
            <li><strong>Metrics Table</strong>: View computed properties for each molecule; click on any row in the metrics table to jump directly to that molecule in the viewer</li>
            <li><strong>Downloads</strong>: Download individual molecule files (SDF format) or all outputs as a ZIP archive</li>
          </ul>
        </section>
      </div>
    </div>
  );
}

