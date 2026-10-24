# ergo-keymap-optimizer

A corpus-driven **Simulated Annealing & Genetic Order-Crossover keymap optimizer** for split columnar mechanical keyboards (**34-key Ferris Sweep**, **36-key Corne / Chocofi**, and **42-key Corne**) tailored to systems programming workloads (Python, Rust, C++) and prose.

Models physical switch coordinates, columnar vertical stagger, per-finger strength multipliers, and thumb-fan geometry to minimize **Same-Finger Bigrams (SFB)**, **Same-Finger Skipgrams (SFS)**, **Lateral Stretch Bigrams (LSB)**, and **row-jump scissors** while maximizing **inward finger rolls** and balanced hand alternation. Outputs ready-to-compile **QMK (`keymap.c`)** and **ZMK (`.keymap`)** firmware layer definitions.

## Key Capabilities

- **Physical Split Columnar Geometry (`KeyboardGeometry`)**: Built-in coordinate models for `corne_36` (`3x5+3`), `ferris_34` (`3x5+2`), and `corne_42` (`3x6+3`), including column stagger offsets (`0.35u` pinky stagger, middle/ring offsets, inner-index lateral reach) and thumb-arc coordinates.
- **Source Code N-Gram Profiler (`CorpusProfile`)**: Extracts normalized monogram, bigram, 1-skipgram (`a_b` from `axb`), and trigram distributions from custom text or the bundled multi-language systems code corpus.
- **Biomechanical Cost Model (`LayoutScorer`)**:
  - **Same-Finger Bigrams (SFB) & Skipgrams (SFS)**: Penalizes consecutive or 1-skip keystrokes on the same finger weighted by Euclidean switch-cap distance and finger weakness multipliers (`PINKY=1.55x`, `RING=1.25x`).
  - **Lateral Stretch Bigrams (LSB) & Row-Jump Scissors**: Detects adjacent-finger stretches exceeding `1.35u` horizontally (e.g. middle-to-inner-index column) and 2-row vertical jumps on the same hand.
  - **Roll Directionality**: Rewards outer-to-inner finger progressions (`PINKY -> RING -> MIDDLE -> INDEX`) as smooth inward rolls and penalizes awkward outward rolls and hand load imbalance.
- **Seeded Simulated Annealing & OX1 Crossover (`SimulatedAnnealingOptimizer`)**: Combines 2-key swap and 3-key cycle perturbations under a cooling schedule with Order-1 Genetic Crossover (`genetic_crossover`), respecting pinned thumb/modifier keys.
- **Firmware Code Generation**: Exports optimized layouts directly to QMK `LAYOUT_split_3x5_3` / `LAYOUT_split_3x5_2` C macros and ZMK devicetree `.keymap` bindings.

## Quickstart

```python
from ergo_keymap import (
    CorpusProfile,
    KeymapLayout,
    LayoutScorer,
    SimulatedAnnealingOptimizer,
    export_qmk_keymap,
    export_zmk_keymap,
)

# 1. Load the systems programming corpus (Python / Rust / C++)
corpus = CorpusProfile.code_corpus()
scorer = LayoutScorer()

# 2. Compare baseline QWERTY vs. Colemak-DH on 36-key Corne geometry
qwerty = KeymapLayout.qwerty_36()
colemak = KeymapLayout.colemak_dh_36()
print("QWERTY SFB rate:", f"{scorer.evaluate(qwerty, corpus).sfb_rate:.2%}")
print("Colemak-DH SFB rate:", f"{scorer.evaluate(colemak, corpus).sfb_rate:.2%}")

# 3. Run seeded Simulated Annealing starting from QWERTY (keeping thumb cluster pinned)
optimizer = SimulatedAnnealingOptimizer(initial_temp=0.40, cooling_rate=0.99, seed=42)
result = optimizer.optimize(qwerty, corpus, iterations=500)
print(f"Optimized score: {result.initial_breakdown.total_score:.3f} -> {result.best_breakdown.total_score:.3f}")

# 4. Export to QMK and ZMK firmware snippets
print(export_qmk_keymap(result.best_layout, layer_name="_BASE"))
print(export_zmk_keymap(result.best_layout, layer_name="default_layer"))
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Author

**Abdullah Azmi** — [abdullahazmi.com](https://abdullahazmi.com) · [@aazminu](https://github.com/aazminu)


## CLI Diagnostics & Invariant Verification

Run the built-in deterministic diagnostic probe and invariant self-check directly via the module CLI:

```bash
PYTHONPATH=src python -m ergo_keymap --self-test --json
```

Run the complete regression test suite:

```bash
PYTHONPATH=src pytest -q
```
