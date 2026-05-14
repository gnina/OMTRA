# Pharmacophore Inputs

OMTRA supports pharmacophore-conditioned generation tasks, where pharmacophore constraints guide the model to produce molecules matching a desired binding hypothesis. This document covers the supported pharmacophore file formats and how to prepare pharmacophore inputs.

## Supported File Formats

OMTRA accepts pharmacophore constraints in three formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| **JSON** | `.json` | Pharmit-compatible format with full control over feature types and coordinates |
| **XYZ** | `.xyz` | Simpler format using element symbols as feature type proxies |
| **SDF** | `.sdf` | Ligand file — pharmacophore features are automatically extracted |

## Generating Pharmacophore Files

### From a Ligand SDF (Recommended)

The easiest way to create a pharmacophore file is to extract features from an existing ligand using the built-in `mol2pharm` command:

```bash
# Basic usage
omtra mol2pharm ligand.sdf -o pharmacophore.json --pretty

# Verbose output showing extracted features
omtra mol2pharm ligand.sdf -o pharmacophore.json --pretty --verbose
```

Additional options:

```bash
# Create with all features disabled by default (for selective enabling)
omtra mol2pharm ligand.sdf -o pharm.json --all-disabled

# Process only the first molecule in a multi-molecule SDF
omtra mol2pharm multi.sdf -o pharm.json --first-only
```

### Other Methods

- **Pharmit Web Interface**: Visit [http://pharmit.csb.pitt.edu/](http://pharmit.csb.pitt.edu/), upload a ligand, and export features as JSON.
- **Pharmit CLI**: `pharmit pharma -in ligand.sdf -out pharmacophore.json`
- **OMTRA Web Application**: Upload an SDF file to the web interface, which extracts and visualizes pharmacophore features for interactive selection.
- **Manual Creation**: Write JSON or XYZ files directly using the formats described below.

-----

## JSON Format

The JSON format follows the structure used by the [Pharmit](http://pharmit.csb.pitt.edu/) pharmacophore search engine and provides the most control.

### Structure

```json
{
  "points": [
    {
      "name": "Aromatic",
      "x": 10.5,
      "y": 20.3,
      "z": 15.2,
      "enabled": true
    },
    {
      "name": "HydrogenAcceptor",
      "x": 8.2,
      "y": 18.7,
      "z": 14.1,
      "enabled": true
    }
  ]
}
```

### Fields

- **`points`** (array, required): List of pharmacophore feature definitions.
  - **`name`** (string, required): Feature type (see [Supported Feature Types](#supported-feature-types)).
  - **`x`**, **`y`**, **`z`** (float, required): 3D coordinates in Angstroms.
  - **`enabled`** (boolean, optional): Whether this feature is active. Defaults to `true`. Set to `false` to exclude a feature without removing it from the file.

### Complete Example

```json
{
  "points": [
    {
      "name": "Aromatic",
      "x": 12.456,
      "y": 8.234,
      "z": 15.789,
      "enabled": true
    },
    {
      "name": "HydrogenDonor",
      "x": 10.123,
      "y": 11.456,
      "z": 14.234,
      "enabled": true
    },
    {
      "name": "HydrogenAcceptor",
      "x": 14.567,
      "y": 9.890,
      "z": 13.456,
      "enabled": true
    },
    {
      "name": "Hydrophobic",
      "x": 11.234,
      "y": 7.890,
      "z": 17.123,
      "enabled": true
    },
    {
      "name": "PositiveIon",
      "x": 13.890,
      "y": 12.345,
      "z": 16.789,
      "enabled": false
    }
  ]
}
```

In this example, the `PositiveIon` feature is disabled and will be ignored during generation.

-----

## XYZ Format

A simpler format that uses element symbols as proxies for feature types:

```
7
Pharmacophore features
P 12.456 8.234 15.789
S 10.123 11.456 14.234
F 14.567 9.890 13.456
C 11.234 7.890 17.123
N 13.890 12.345 16.789
O 9.123 10.456 12.890
Cl 15.678 13.234 18.456
```

**Format:**
- Line 1: Number of pharmacophore points
- Line 2: Comment line (ignored)
- Lines 3+: `ELEMENT X Y Z`

**Element-to-feature mapping:**

| Element | Feature Type |
|---------|-------------|
| `P` | Aromatic |
| `S` | HydrogenDonor |
| `F` | HydrogenAcceptor |
| `N` | PositiveIon |
| `O` | NegativeIon |
| `C` | Hydrophobic |
| `Cl` | Halogen |

-----

## Supported Feature Types

OMTRA recognizes seven pharmacophore feature types:

| Feature Type | Description |
|--------------|-------------|
| `Aromatic` | Aromatic ring center (6-membered or 5-membered rings) |
| `HydrogenDonor` | Hydrogen bond donor (e.g., NH, OH groups) |
| `HydrogenAcceptor` | Hydrogen bond acceptor (e.g., C=O, N, O atoms) |
| `PositiveIon` | Positively charged or ionizable group |
| `NegativeIon` | Negatively charged or ionizable group (e.g., carboxylate) |
| `Hydrophobic` | Hydrophobic/lipophilic region |
| `Halogen` | Halogen bond donor (F, Cl, Br, I) |

Features with unrecognized `name` values will be treated as `UNK` (unknown) type.

-----

## SDF Format

You can pass an SDF ligand file directly as the `--pharmacophore_file` argument. OMTRA will automatically extract pharmacophore features from the ligand's 3D structure. This is equivalent to running `omtra mol2pharm` and using the resulting JSON, but skips the intermediate file.

```bash
omtra --task denovo_ligand_from_pharmacophore_condensed \
  --pharmacophore_file reference_ligand.sdf \
  --n_samples 100
```
