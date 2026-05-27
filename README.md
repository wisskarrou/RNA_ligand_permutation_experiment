# RNA–Ligand Permutation Experiment

[![Paper](https://img.shields.io/badge/arXiv-2512.15645-b31b1b.svg)](https://arxiv.org/abs/2512.15645)

This repository contains the code for the **permutation experiment** introduced in the review:

> **Machine learning for RNA-targeting drug design**  
> Wissam Karroucha, Carlos Oliver, Véronique Stoven, Vincent Mallet  
> *arXiv preprint arXiv:2512.15645*, 2025

The experiment evaluates whether RNA–ligand binding models genuinely exploit **both** the RNA target and the ligand, or whether they rely on only one of the two inputs—a critical diagnostic for specificity in virtual screening.

---

## Table of Contents

- [Motivation](#motivation)
- [Experimental Design](#experimental-design)
  - [RNA permutation (target-swap)](#rna-permutation-target-swap)
  - [Ligand permutation (ligand-swap)](#ligand-permutation-ligand-swap)
  - [Derangement algorithm](#derangement-algorithm)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Running the Experiment](#running-the-experiment)
- [Key Results](#key-results)
- [Citation](#citation)

---

## Motivation

A standard concern in machine learning benchmarks for binding prediction is that a model may learn *spurious correlations* rather than the biophysical complementarity between an RNA target and a small molecule. For instance, if the decoy set only contains molecules that are chemically very different from true binders, a model can achieve high apparent performance by learning to recognise RNA-binding molecules *in general*, regardless of the specific RNA target.

The permutation experiment provides a model-agnostic stress test: if a model truly exploits the identity of the RNA, its performance should drop significantly when RNA–ligand pairs are randomly reshuffled.

---

## Experimental Design

All models are evaluated in three conditions:

| Condition | RNA | Ligand | Expected behaviour |
|-----------|-----|--------|--------------------|
| `none` (baseline) | original | original | highest performance |
| `target-swap` | **permuted** | original | drop ↔ RNA specificity |
| `ligand-swap` | original | **permuted** | drop ↔ ligand specificity |

### RNA permutation (target-swap)

Each RNA in the test set is replaced by a *different* RNA drawn from the same test set. The permutation is a **derangement**: no RNA is mapped to itself. The labels (binding affinities or binary interactions) are kept with the original ligands, deliberately breaking the RNA–ligand correspondence.

```python
# Conceptual sketch — see each model's ablation_utils.py for the full implementation
rna_indices   = list(range(n_rnas))
shuffled      = guaranteed_derangement(rna_indices, seed=seed)

for i, sample in enumerate(test_set):
    sample.rna = test_set[shuffled[i]].rna   # swap RNA
    # sample.ligand and sample.label are unchanged
```

### Ligand permutation (ligand-swap)

Symmetrically, each ligand is replaced by a different ligand from the test set while the RNA target is kept fixed.

```python
mol_indices  = list(range(n_mols))
shuffled     = guaranteed_derangement(mol_indices, seed=seed)

for i, sample in enumerate(test_set):
    sample.ligand = test_set[shuffled[i]].ligand   # swap ligand
    # sample.rna and sample.label are unchanged
```

Both swaps are repeated over **multiple random seeds** (typically 3) to obtain stable estimates.

### Derangement algorithm

To guarantee that no element is mapped to itself (which would leave some pairs unperturbed), we use a *shuffle-and-fix* derangement:

```python
def guaranteed_derangement(items, seed=0):
    """
    Returns a permutation of `items` where no element stays at its
    original position (derangement).
    """
    random.seed(seed)
    shuffled = items[:]

    for _ in range(100):
        random.shuffle(shuffled)
        fixed = [i for i in range(len(items)) if shuffled[i] == items[i]]

        if not fixed:
            return shuffled

        # Resolve fixed points by swapping pairs
        for k in range(0, len(fixed) - 1, 2):
            shuffled[fixed[k]], shuffled[fixed[k+1]] = \
                shuffled[fixed[k+1]], shuffled[fixed[k]]

        if len(fixed) % 2 != 0:       # odd leftover: swap with a non-fixed element
            i = fixed[-1]
            j = next(k for k in range(len(items)) if k not in fixed)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        if all(shuffled[i] != items[i] for i in range(len(items))):
            return shuffled

    return shuffled   # fallback — should never be reached
```

---

## Repository Structure

```
RNA_ligand_permutation_experiment/
│
├── DeepRSMA/                   # Affinity prediction (graph neural network)
│   ├── ablations.py            #   ← permutation experiment entry point
│   ├── ablation_utils.py       #   derangement + dataset helpers
│   └── ...
│
├── GerNA-Bind/                 # Binding specificity (geometric deep learning)
│   ├── ablations.py
│   └── utils/ablation_utils.py
│
├── RNAsmol/                    # Sequence-based binding prediction
│   └── rnasmol/
│       └── ablations.py
│
├── RSAPred/                    # Feature-based linear model
│   └── Model_evaluation/
│       └── ablations.py
│
├── rnamigos2/                  # Virtual screening (graph + docking)
│   ├── ablations.py
│   └── ablation_utils.py
│
└── boltz/                      # Structure prediction baseline
    └── scripts/
        └── eval/
```

Each model sub-directory is a lightly modified copy of the original authors' codebase. The only additions are:

- **`ablations.py`** — main script that runs the three conditions (`none`, `target-swap`, `ligand-swap`) and exports results as a LaTeX table or CSV.
- **`ablation_utils.py`** (where applicable) — the `guaranteed_derangement`, `target_swap`, and `ligand_swap` helper functions.

---

## Models

| Model | Reference |
|-------|-----------|
| **RNAmigos 2** | [Carvajal-Patiño et al., 2025](https://www.nature.com/articles/s41467-025-57852-0) |
| **DeepRSMA** | [Huang et al., 2024](https://doi.org/10.1093/bioinformatics/btae678) |
| **GerNA-Bind** | [Xia et al., 2025](https://www.nature.com/articles/s42256-025-01154-z) |
| **RNAsmol** | [Ma et al., 2025](https://www.nature.com/articles/s43588-025-00820-x) |
| **RSAPred** | [Krishnan et al., 2024](https://doi.org/10.1093/bib/bbae002) |
| **Boltz-2** | [Passaro et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/) |

---

## Running the Experiment

All scripts assume that the original model weights and data are present (see each model's own README). 

### DeepRSMA

```bash
cd DeepRSMA
python ablations.py
# Results written to ablation_results.tex
```

### GerNA-Bind

```bash
cd GerNA-Bind
python ablations.py
# Results written to auroc_results.tex and inference_results.csv
```

### RNAsmol

```bash
cd RNAsmol/rnasmol
python ablations.py
# Results written to stdout and a LaTeX table
```

### RSAPred

```bash
cd RSAPred/Model_evaluation
# Pre-generate swapped datasets first (see README_swapping.md):
#   python swap_v2.py <dataset_raw> <output_file>
python ablations.py
# Results written to inference_results.csv
```

### RNAmigos 2

```bash
cd rnamigos2
python ablations.py
# Per-seed AuROC / Rognan-score printed to stdout; CSVs in outputs/pockets/
```

### Boltz

```bash
cd boltz
# Follow the structure prediction pipeline in scripts/process/ and scripts/train/,
# then evaluate with:
python scripts/eval/run_evals.py
python scripts/eval/aggregate_evals.py
```

---

## Key Results

Performance significantly decreases for most models when RNA targets are swapped, indicating that these models genuinely exploit the identity of the RNA target:

- **RNAmigos 2**, **DeepRSMA**, and **GerNA-Bind** show a pronounced drop under `target-swap`, confirming that structural reasoning about the RNA is key to their specificity.
- **RSAPred** shows limited specificity under `target-swap`, likely due to its reliance on expert-crafted features that do not fully capture geometric RNA–ligand complementarity.
- **RNAsmol** shows limited specificity under `ligand-swap` (molecular perturbations setup): its negatives are constructed by replacing the true ligand with molecules sharing similar MACCS fingerprints, allowing the model to memorise active compounds without learning target-specific interactions.

The `ligand-swap` experiment produces an even stronger performance drop across all models, showing that no model relies on RNA features alone—consistent with the higher expressiveness of ligand encodings and the balanced active/inactive ratio per RNA target in the test set.

> These findings underscore the necessity of carefully designed datasets and decoy sets. When decoys are chemically very different from RNA binders, a model can learn to identify *RNA-binding molecules in general* rather than to predict binding to a *specific* RNA target. We recommend reporting permutation-experiment results alongside standard benchmarks in future work.

---

## Citation

If you use this code or build on the permutation experiment design, please cite:

```bibtex
@article{karroucha2025machine,
  title   = {Machine learning for RNA-targeting drug design},
  author  = {Karroucha, Wissam and Oliver, Carlos and Stoven, Veronique and Mallet, Vincent},
  journal = {arXiv preprint arXiv:2512.15645},
  year    = {2025}
}
```

Please also cite the original model papers listed in the [Models](#models) table.
