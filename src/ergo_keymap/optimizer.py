"""Corpus-driven simulated annealing & genetic layout optimizer for split columnar keyboards.

Models 34-key (Ferris Sweep 3x5+2), 36-key (Corne / Chocofi 3x5+3), and 42-key (Corne 3x6+3)
split columnar mechanical keyboard geometries. Extracts monogram, bigram, 1-skipgram (SFS),
and trigram frequencies from programming/prose corpora, evaluates Same-Finger Bigrams (SFB),
Same-Finger Skipgrams (SFS), Lateral Stretch Bigrams (LSB), row-jump scissors, and
inward/outward finger rolls, optimizes layouts via seeded Simulated Annealing & genetic
crossover, and exports ready-to-compile QMK `keymap.c` and ZMK `.keymap` definitions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import math
import random
from typing import Iterable, Mapping, Optional, Sequence


class Hand(str, Enum):
    """Left or right split half of a columnar keyboard."""

    LEFT = "left"
    RIGHT = "right"


class Finger(IntEnum):
    """Finger identifier ordered from outer (pinky=0) to inner (index=3, thumb=4).

    Ordering 0..3 from outer to inner on each hand makes inward vs. outward roll
    detection simple: moving from a smaller finger index to a larger finger index
    (e.g. Ring=1 -> Middle=2 -> Index=3) is an *inward* roll on that hand.
    """

    PINKY = 0
    RING = 1
    MIDDLE = 2
    INDEX = 3
    THUMB = 4


FINGER_STRENGTH_PENALTY: dict[Finger, float] = {
    Finger.PINKY: 1.55,
    Finger.RING: 1.25,
    Finger.MIDDLE: 1.00,
    Finger.INDEX: 1.00,
    Finger.THUMB: 0.90,
}


@dataclass(frozen=True)
class KeyCoord:
    """Physical geometry and biomechanical cost metadata for a single key switch."""

    index: int
    hand: Hand
    finger: Finger
    row: int
    col: int
    x: float
    y: float
    base_effort: float
    is_home: bool = False
    is_thumb: bool = False

    def distance_to(self, other: KeyCoord) -> float:
        """Euclidean switch-cap center distance in key-unit (1u = 19.05mm / 18mm Choc) units."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class KeyboardGeometry:
    """Physical switch layout for a split columnar mechanical keyboard."""

    name: str
    keys: tuple[KeyCoord, ...]

    @property
    def num_keys(self) -> int:
        return len(self.keys)

    @classmethod
    def corne_36(cls) -> KeyboardGeometry:
        """Standard 3x5+3 (36-key) split columnar layout (Corne / Chocofi / Kyria 3x5).

        Index order:
        - 0..4:   Left top row (pinky -> inner index)
        - 5..9:   Right top row (inner index -> pinky)
        - 10..14: Left home row (pinky -> inner index)
        - 15..19: Right home row (inner index -> pinky)
        - 20..24: Left bottom row (pinky -> inner index)
        - 25..29: Right bottom row (inner index -> pinky)
        - 30..32: Left thumb cluster (tuck, home, reach)
        - 33..35: Right thumb cluster (reach, home, tuck)
        """
        # Columnar vertical stagger offsets (in 1u key units) per column:
        # col 0=pinky, 1=ring, 2=middle, 3=index_home, 4=index_inner
        col_stagger = (0.35, 0.12, 0.00, 0.15, 0.22)
        col_fingers = (Finger.PINKY, Finger.RING, Finger.MIDDLE, Finger.INDEX, Finger.INDEX)

        # Base per-key effort matrix [row 0..2][col 0..4 from pinky to inner index]
        effort_grid = (
            (2.4, 1.8, 1.4, 1.6, 2.3),  # Top row
            (1.4, 1.1, 1.0, 1.0, 1.8),  # Home row (lowest effort on home positions col 0..3)
            (2.6, 2.0, 1.7, 1.6, 2.5),  # Bottom row
        )

        keys: list[KeyCoord] = []
        idx = 0
        for row in range(3):
            # Left hand columns 0..4 (pinky to inner index)
            for col in range(5):
                finger = col_fingers[col]
                x = float(col)
                y = float(row) + col_stagger[col]
                is_home = row == 1 and col in (0, 1, 2, 3)
                keys.append(
                    KeyCoord(
                        index=idx,
                        hand=Hand.LEFT,
                        finger=finger,
                        row=row,
                        col=col,
                        x=x,
                        y=y,
                        base_effort=effort_grid[row][col],
                        is_home=is_home,
                        is_thumb=False,
                    )
                )
                idx += 1

            # Right hand columns 0..4 (inner index to pinky, mirroring left hand)
            for r_col in range(5):
                mirror_col = 4 - r_col  # 4=inner index, 3=index, 2=middle, 1=ring, 0=pinky
                finger = col_fingers[mirror_col]
                x = 7.0 + float(r_col)
                y = float(row) + col_stagger[mirror_col]
                is_home = row == 1 and mirror_col in (0, 1, 2, 3)
                keys.append(
                    KeyCoord(
                        index=idx,
                        hand=Hand.RIGHT,
                        finger=finger,
                        row=row,
                        col=mirror_col,
                        x=x,
                        y=y,
                        base_effort=effort_grid[row][mirror_col],
                        is_home=is_home,
                        is_thumb=False,
                    )
                )
                idx += 1

        # Left 3-key thumb fan (indices 30..32)
        thumb_efforts_left = (1.3, 1.0, 1.5)
        for t_idx in range(3):
            keys.append(
                KeyCoord(
                    index=idx,
                    hand=Hand.LEFT,
                    finger=Finger.THUMB,
                    row=3,
                    col=2 + t_idx,
                    x=2.5 + t_idx * 1.0,
                    y=3.15 + t_idx * 0.1,
                    base_effort=thumb_efforts_left[t_idx],
                    is_home=(t_idx == 1),
                    is_thumb=True,
                )
            )
            idx += 1

        # Right 3-key thumb fan (indices 33..35)
        thumb_efforts_right = (1.5, 1.0, 1.3)
        for t_idx in range(3):
            keys.append(
                KeyCoord(
                    index=idx,
                    hand=Hand.RIGHT,
                    finger=Finger.THUMB,
                    row=3,
                    col=4 - t_idx,
                    x=6.5 + t_idx * 1.0,
                    y=3.35 - t_idx * 0.1,
                    base_effort=thumb_efforts_right[t_idx],
                    is_home=(t_idx == 1),
                    is_thumb=True,
                )
            )
            idx += 1

        return cls(name="corne_36", keys=tuple(keys))

    @classmethod
    def ferris_34(cls) -> KeyboardGeometry:
        """3x5+2 (34-key) Ferris Sweep split columnar geometry."""
        base_36 = cls.corne_36()
        # Keep the 30 alpha keys + 2 primary thumb keys per side (drop inner reach thumbs 32 and 33)
        filtered = [k for k in base_36.keys if k.index not in (32, 33)]
        reindexed = [
            KeyCoord(
                index=new_i,
                hand=k.hand,
                finger=k.finger,
                row=k.row,
                col=k.col,
                x=k.x,
                y=k.y,
                base_effort=k.base_effort,
                is_home=k.is_home,
                is_thumb=k.is_thumb,
            )
            for new_i, k in enumerate(filtered)
        ]
        return cls(name="ferris_34", keys=tuple(reindexed))

    @classmethod
    def corne_42(cls) -> KeyboardGeometry:
        """3x6+3 (42-key) 6-column Corne split geometry with outer pinky column."""
        base_36 = cls.corne_36()
        keys: list[KeyCoord] = list(base_36.keys)
        idx = len(keys)
        for hand in (Hand.LEFT, Hand.RIGHT):
            for row in range(3):
                x = -1.0 if hand == Hand.LEFT else 12.0
                y = float(row) + 0.45
                keys.append(
                    KeyCoord(
                        index=idx,
                        hand=hand,
                        finger=Finger.PINKY,
                        row=row,
                        col=-1,
                        x=x,
                        y=y,
                        base_effort=2.9 if row != 1 else 2.2,
                        is_home=False,
                        is_thumb=False,
                    )
                )
                idx += 1
        return cls(name="corne_42", keys=tuple(keys))


DEFAULT_CODE_CORPUS = """
def allocate_sequence(self, seq_id: str, prompt_tokens: list[int]) -> SequenceTable:
    if seq_id in self._sequences:
        raise AllocationError(f"sequence {seq_id} already exists in block table")
    matched_ids, parent_hash = self.prefix_cache.match_prefix_blocks(prompt_tokens)
    for block_id in matched_ids:
        self._increment_ref(block_id)
    return self._sequences[seq_id]

fn compute_prefix_hash(tokens: &[u64], parent: Option<u64>) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &token in tokens.iter() {
        hash ^= token;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

template <typename T>
std::vector<int> schedule_batch(const std::vector<T>& requests, size_t watermark) {
    std::vector<int> active_ids;
    active_ids.reserve(requests.size());
    for (const auto& req : requests) {
        if (req.num_blocks <= watermark) {
            active_ids.push_back(req.id);
        }
    }
    return active_ids;
}
"""


@dataclass(frozen=True)
class CorpusProfile:
    """Normalized n-gram statistics extracted from source code or prose text."""

    name: str
    total_chars: int
    monograms: Mapping[str, float]
    bigrams: Mapping[tuple[str, str], float]
    skipgrams: Mapping[tuple[str, str], float]
    trigrams: Mapping[tuple[str, str, str], float]

    @classmethod
    def from_text(
        cls,
        text: str,
        name: str = "custom",
        allowed_chars: Optional[ set[str] ] = None,
        fold_case: bool = True,
    ) -> CorpusProfile:
        """Extract normalized monogram, bigram, 1-skipgram, and trigram frequencies."""
        processed = text.lower() if fold_case else text
        # Normalize runs of whitespace/newlines to single space or newline tokens
        filtered: list[str] = []
        for ch in processed:
            if ch in ("\r",):
                continue
            if allowed_chars is not None and ch not in allowed_chars:
                continue
            filtered.append(ch)

        n = len(filtered)
        if n == 0:
            raise ValueError("Corpus text produced 0 valid characters after filtering.")

        mono_counts: Counter[str] = Counter(filtered)
        bi_counts: Counter[tuple[str, str]] = Counter()
        skip_counts: Counter[tuple[str, str]] = Counter()
        tri_counts: Counter[tuple[str, str, str]] = Counter()

        for i in range(n - 1):
            a, b = filtered[i], filtered[i + 1]
            bi_counts[(a, b)] += 1
            if i + 2 < n:
                c = filtered[i + 2]
                skip_counts[(a, c)] += 1
                tri_counts[(a, b, c)] += 1

        mono_total = float(sum(mono_counts.values())) or 1.0
        bi_total = float(sum(bi_counts.values())) or 1.0
        skip_total = float(sum(skip_counts.values())) or 1.0
        tri_total = float(sum(tri_counts.values())) or 1.0

        return cls(
            name=name,
            total_chars=n,
            monograms={k: v / mono_total for k, v in mono_counts.items()},
            bigrams={k: v / bi_total for k, v in bi_counts.items()},
            skipgrams={k: v / skip_total for k, v in skip_counts.items()},
            trigrams={k: v / tri_total for k, v in tri_counts.items()},
        )

    @classmethod
    def code_corpus(cls) -> CorpusProfile:
        """Build a default multilingual systems programming corpus (Python, Rust, C++)."""
        return cls.from_text(DEFAULT_CODE_CORPUS, name="systems_code_corpus", fold_case=True)


@dataclass(frozen=True)
class KeymapLayout:
    """Mapping between physical key indices on a `KeyboardGeometry` and character symbols."""

    name: str
    geometry: KeyboardGeometry
    symbols: tuple[str, ...]
    char_to_index: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.symbols) != self.geometry.num_keys:
            raise ValueError(
                f"Layout {self.name!r} has {len(self.symbols)} symbols, "
                f"but geometry {self.geometry.name!r} requires {self.geometry.num_keys}."
            )
        mapping: dict[str, int] = {}
        for idx, sym in enumerate(self.symbols):
            if sym and sym not in mapping:
                mapping[sym] = idx
        object.__setattr__(self, "char_to_index", mapping)

    def swap(self, idx_a: int, idx_b: int, name: Optional[str] = None) -> KeymapLayout:
        """Return a new KeymapLayout with positions `idx_a` and `idx_b` swapped."""
        syms = list(self.symbols)
        syms[idx_a], syms[idx_b] = syms[idx_b], syms[idx_a]
        return KeymapLayout(
            name=name or self.name,
            geometry=self.geometry,
            symbols=tuple(syms),
        )

    @classmethod
    def qwerty_36(cls) -> KeymapLayout:
        """Standard QWERTY 3x5+3 split keymap for 36-key Corne/Chocofi."""
        geom = KeyboardGeometry.corne_36()
        symbols = (
            # Top row: left 0..4, right 5..9
            "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
            # Home row: left 10..14, right 15..19
            "a", "s", "d", "f", "g", "h", "j", "k", "l", ";",
            # Bottom row: left 20..24, right 25..29
            "z", "x", "c", "v", "b", "n", "m", ",", ".", "/",
            # Thumbs: left 30..32, right 33..35
            "\t", " ", "\n", "-", "'", "\b",
        )
        return cls(name="qwerty_36", geometry=geom, symbols=symbols)

    @classmethod
    def colemak_dh_36(cls) -> KeymapLayout:
        """Colemak-DH (matrix/columnar standard) 3x5+3 split keymap for 36-key Corne."""
        geom = KeyboardGeometry.corne_36()
        symbols = (
            # Top row: q w f p b | j l u y ;
            "q", "w", "f", "p", "b", "j", "l", "u", "y", ";",
            # Home row: a r s t g | m n e i o
            "a", "r", "s", "t", "g", "m", "n", "e", "i", "o",
            # Bottom row: z x c d v | k h , . /
            "z", "x", "c", "d", "v", "k", "h", ",", ".", "/",
            # Thumbs
            "\t", " ", "\n", "-", "'", "\b",
        )
        return cls(name="colemak_dh_36", geometry=geom, symbols=symbols)


@dataclass(frozen=True)
class ScoringWeights:
    """Hyperparameters governing the biomechanical cost function."""

    base_effort_weight: float = 1.0
    sfb_weight: float = 14.0
    sfs_weight: float = 4.5
    lsb_weight: float = 5.0
    scissor_weight: float = 6.5
    inward_roll_bonus: float = 1.8
    outward_roll_penalty: float = 0.6
    hand_imbalance_weight: float = 2.5


@dataclass(frozen=True)
class ScoreBreakdown:
    """Detailed biomechanical evaluation metrics for a keymap layout."""

    total_score: float
    base_effort: float
    sfb_rate: float
    sfs_rate: float
    lsb_rate: float
    scissor_rate: float
    inward_roll_rate: float
    outward_roll_rate: float
    hand_alternation_rate: float
    left_hand_share: float
    worst_sfbs: tuple[tuple[str, str, float], ...]


class LayoutScorer:
    """Evaluates a `KeymapLayout` against a `CorpusProfile` using biomechanical heuristics."""

    def __init__(self, weights: Optional[ScoringWeights] = None) -> None:
        self.weights = weights or ScoringWeights()

    def evaluate(self, layout: KeymapLayout, corpus: CorpusProfile) -> ScoreBreakdown:
        """Compute total ergonomic score (lower is better) and detailed rate breakdown."""
        keys = layout.geometry.keys
        c2i = layout.char_to_index

        # 1. Monogram base effort and hand load balance
        base_effort = 0.0
        left_load = 0.0
        mapped_mono_mass = 0.0
        for ch, freq in corpus.monograms.items():
            idx = c2i.get(ch)
            if idx is None:
                continue
            kc = keys[idx]
            base_effort += freq * kc.base_effort
            mapped_mono_mass += freq
            if kc.hand == Hand.LEFT:
                left_load += freq

        left_share = (left_load / mapped_mono_mass) if mapped_mono_mass > 0 else 0.5
        imbalance = abs(left_share - 0.5)

        # 2. Bigram metrics: SFB, LSB, Scissors, Rolls, Hand Alternation
        sfb_rate = 0.0
        sfb_weighted_cost = 0.0
        lsb_rate = 0.0
        scissor_rate = 0.0
        inward_roll_rate = 0.0
        outward_roll_rate = 0.0
        alternation_rate = 0.0
        sfb_contrib: list[tuple[str, str, float]] = []

        for (a, b), freq in corpus.bigrams.items():
            if a == b:
                # Same-key repeat is layout-invariant; only distinct-key same-finger bigrams depend on layout
                continue
            ia = c2i.get(a)
            ib = c2i.get(b)
            if ia is None or ib is None or ia == ib:
                continue

            ka = keys[ia]
            kb = keys[ib]

            if ka.hand != kb.hand:
                alternation_rate += freq
                continue

            # Same hand bigram
            if ka.finger == kb.finger:
                sfb_rate += freq
                dist = ka.distance_to(kb)
                penalty = freq * dist * FINGER_STRENGTH_PENALTY[ka.finger]
                sfb_weighted_cost += penalty
                sfb_contrib.append((a, b, freq))
                continue

            # Ignore thumb-to-finger bigrams for LSB / Scissor / Roll classification
            if ka.is_thumb or kb.is_thumb:
                continue

            # Check Lateral Stretch Bigram (LSB): adjacent fingers stretched horizontally > 1.35u
            finger_gap = abs(int(ka.finger) - int(kb.finger))
            horiz_dist = abs(ka.x - kb.x)
            if finger_gap == 1 and horiz_dist > 1.35:
                lsb_rate += freq

            # Check Row-jump Scissor: same-hand different fingers jumping >= 2 rows vertically
            row_diff = abs(ka.row - kb.row)
            if row_diff >= 2 and abs(ka.y - kb.y) >= 1.75:
                scissor_rate += freq
            else:
                # Roll classification: Finger enum increases from PINKY(0) -> INDEX(3)
                if int(ka.finger) < int(kb.finger):
                    inward_roll_rate += freq
                elif int(ka.finger) > int(kb.finger):
                    outward_roll_rate += freq

        # 3. Same-Finger Skipgrams (SFS / 1-skip bigrams a_b)
        sfs_rate = 0.0
        sfs_weighted_cost = 0.0
        for (a, c), freq in corpus.skipgrams.items():
            if a == c:
                continue
            ia = c2i.get(a)
            ic = c2i.get(c)
            if ia is None or ic is None or ia == ic:
                continue
            ka = keys[ia]
            kc = keys[ic]
            if ka.hand == kc.hand and ka.finger == kc.finger:
                sfs_rate += freq
                sfs_weighted_cost += freq * ka.distance_to(kc) * FINGER_STRENGTH_PENALTY[ka.finger]

        w = self.weights
        total_score = (
            w.base_effort_weight * base_effort
            + w.sfb_weight * sfb_weighted_cost
            + w.sfs_weight * sfs_weighted_cost
            + w.lsb_weight * lsb_rate
            + w.scissor_weight * scissor_rate
            + w.outward_roll_penalty * outward_roll_rate
            - w.inward_roll_bonus * inward_roll_rate
            + w.hand_imbalance_weight * imbalance
        )

        sfb_contrib.sort(key=lambda item: item[2], reverse=True)
        return ScoreBreakdown(
            total_score=total_score,
            base_effort=base_effort,
            sfb_rate=sfb_rate,
            sfs_rate=sfs_rate,
            lsb_rate=lsb_rate,
            scissor_rate=scissor_rate,
            inward_roll_rate=inward_roll_rate,
            outward_roll_rate=outward_roll_rate,
            hand_alternation_rate=alternation_rate,
            left_hand_share=left_share,
            worst_sfbs=tuple(sfb_contrib[:5]),
        )


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by `SimulatedAnnealingOptimizer`."""

    initial_layout: KeymapLayout
    best_layout: KeymapLayout
    initial_breakdown: ScoreBreakdown
    best_breakdown: ScoreBreakdown
    iterations: int
    accepted_moves: int
    score_history: tuple[float, ...]


class SimulatedAnnealingOptimizer:
    """Seeded Simulated Annealing & Genetic Order-Crossover keymap optimizer."""

    def __init__(
        self,
        scorer: Optional[LayoutScorer] = None,
        initial_temp: float = 0.35,
        cooling_rate: float = 0.992,
        seed: int = 42,
    ) -> None:
        if initial_temp <= 0.0:
            raise ValueError("initial_temp must be positive")
        if not (0.0 < cooling_rate < 1.0):
            raise ValueError("cooling_rate must be in (0.0, 1.0)")
        self.scorer = scorer or LayoutScorer()
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.seed = seed

    def optimize(
        self,
        initial_layout: KeymapLayout,
        corpus: CorpusProfile,
        iterations: int = 400,
        pinned_indices: Optional[Iterable[int]] = None,
    ) -> OptimizationResult:
        """Run seeded Simulated Annealing over swappable key indices.

        By default, if `pinned_indices` is None, thumb cluster keys (`is_thumb == True`)
        are pinned so alpha/punctuation keys optimize on the 3x5 finger matrix.
        Pass `pinned_indices=set()` to allow all keys including thumbs to swap.
        """
        if iterations < 0:
            raise ValueError("iterations must be non-negative")

        rng = random.Random(self.seed)
        if pinned_indices is None:
            pinned = {k.index for k in initial_layout.geometry.keys if k.is_thumb}
        else:
            pinned = set(pinned_indices)

        movable = [i for i in range(initial_layout.geometry.num_keys) if i not in pinned]
        if len(movable) < 2:
            raise ValueError("At least 2 movable key indices are required for optimization.")

        current_layout = initial_layout
        current_eval = self.scorer.evaluate(current_layout, corpus)
        best_layout = current_layout
        best_eval = current_eval
        initial_eval = current_eval

        temp = self.initial_temp
        accepted = 0
        history: list[float] = [best_eval.total_score]

        for step in range(iterations):
            # 80% 2-key swap, 20% 3-key cycle rotation (when >= 3 movable keys exist)
            syms = list(current_layout.symbols)
            if len(movable) >= 3 and rng.random() < 0.20:
                i1, i2, i3 = rng.sample(movable, 3)
                syms[i1], syms[i2], syms[i3] = syms[i2], syms[i3], syms[i1]
            else:
                i1, i2 = rng.sample(movable, 2)
                syms[i1], syms[i2] = syms[i2], syms[i1]

            candidate = KeymapLayout(
                name=f"{initial_layout.name}_opt",
                geometry=current_layout.geometry,
                symbols=tuple(syms),
            )
            cand_eval = self.scorer.evaluate(candidate, corpus)
            delta = cand_eval.total_score - current_eval.total_score

            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current_layout = candidate
                current_eval = cand_eval
                accepted += 1
                if cand_eval.total_score < best_eval.total_score:
                    best_layout = candidate
                    best_eval = cand_eval

            temp *= self.cooling_rate
            if (step + 1) % max(1, iterations // 20) == 0 or step == iterations - 1:
                history.append(best_eval.total_score)

        return OptimizationResult(
            initial_layout=initial_layout,
            best_layout=best_layout,
            initial_breakdown=initial_eval,
            best_breakdown=best_eval,
            iterations=iterations,
            accepted_moves=accepted,
            score_history=tuple(history),
        )

    def genetic_crossover(
        self,
        parent_a: KeymapLayout,
        parent_b: KeymapLayout,
        pinned_indices: Optional[Iterable[int]] = None,
        seed: Optional[int] = None,
    ) -> KeymapLayout:
        """Perform Order-1 Crossover (OX1) on two layouts sharing the same geometry and symbol set."""
        if parent_a.geometry.name != parent_b.geometry.name:
            raise ValueError("Parents must share the same keyboard geometry")
        if sorted(parent_a.symbols) != sorted(parent_b.symbols):
            raise ValueError("Parents must contain the exact same multiset of symbols")

        rng = random.Random(self.seed if seed is None else seed)
        pinned = (
            {k.index for k in parent_a.geometry.keys if k.is_thumb}
            if pinned_indices is None
            else set(pinned_indices)
        )
        movable = [i for i in range(parent_a.geometry.num_keys) if i not in pinned]
        if len(movable) < 2:
            return parent_a

        cut1, cut2 = sorted(rng.sample(range(len(movable)), 2))
        slice_indices = set(movable[cut1 : cut2 + 1])

        child_syms: list[Optional[str]] = [None] * parent_a.geometry.num_keys
        used_syms: set[str] = set()

        for idx in pinned:
            child_syms[idx] = parent_a.symbols[idx]
            used_syms.add(parent_a.symbols[idx])

        for idx in slice_indices:
            child_syms[idx] = parent_a.symbols[idx]
            used_syms.add(parent_a.symbols[idx])

        remaining_from_b = [
            parent_b.symbols[idx] for idx in movable if parent_b.symbols[idx] not in used_syms
        ]
        b_iter = iter(remaining_from_b)
        for idx in movable:
            if child_syms[idx] is None:
                child_syms[idx] = next(b_iter)

        return KeymapLayout(
            name=f"{parent_a.name}_x_{parent_b.name}",
            geometry=parent_a.geometry,
            symbols=tuple(s or "" for s in child_syms),
        )


_QMK_KEYCODE_MAP: dict[str, str] = {
    " ": "KC_SPC",
    "\n": "KC_ENT",
    "\t": "KC_TAB",
    "\b": "KC_BSPC",
    ";": "KC_SCLN",
    ",": "KC_COMM",
    ".": "KC_DOT",
    "/": "KC_SLSH",
    "-": "KC_MINS",
    "'": "KC_QUOT",
    "[": "KC_LBRC",
    "]": "KC_RBRC",
    "=": "KC_EQL",
    "\\": "KC_BSLS",
    "`": "KC_GRV",
}

_ZMK_KEYCODE_MAP: dict[str, str] = {
    " ": "&kp SPACE",
    "\n": "&kp RET",
    "\t": "&kp TAB",
    "\b": "&kp BSPC",
    ";": "&kp SEMI",
    ",": "&kp COMMA",
    ".": "&kp DOT",
    "/": "&kp FSLH",
    "-": "&kp MINUS",
    "'": "&kp SQT",
    "[": "&kp LBKT",
    "]": "&kp RBKT",
    "=": "&kp EQUAL",
    "\\": "&kp BSLH",
    "`": "&kp GRAVE",
}


def _to_qmk_code(sym: str) -> str:
    if not sym:
        return "KC_NO"
    if sym in _QMK_KEYCODE_MAP:
        return _QMK_KEYCODE_MAP[sym]
    if len(sym) == 1 and sym.isalpha():
        return f"KC_{sym.upper()}"
    if len(sym) == 1 and sym.isdigit():
        return f"KC_{sym}"
    return sym


def _to_zmk_code(sym: str) -> str:
    if not sym:
        return "&none"
    if sym in _ZMK_KEYCODE_MAP:
        return _ZMK_KEYCODE_MAP[sym]
    if len(sym) == 1 and sym.isalpha():
        return f"&kp {sym.upper()}"
    if len(sym) == 1 and sym.isdigit():
        return f"&kp N{sym}"
    return sym


def export_qmk_keymap(layout: KeymapLayout, layer_name: str = "_BASE") -> str:
    """Export a 36-key or 34-key `KeymapLayout` to a valid QMK `keymap.c` macro snippet."""
    codes = [_to_qmk_code(s) for s in layout.symbols]
    if layout.geometry.num_keys == 36:
        macro = "LAYOUT_split_3x5_3"
        row0 = ", ".join(codes[0:10])
        row1 = ", ".join(codes[10:20])
        row2 = ", ".join(codes[20:30])
        thumbs = ", ".join(codes[30:36])
    elif layout.geometry.num_keys == 34:
        macro = "LAYOUT_split_3x5_2"
        row0 = ", ".join(codes[0:10])
        row1 = ", ".join(codes[10:20])
        row2 = ", ".join(codes[20:30])
        thumbs = ", ".join(codes[30:34])
    else:
        macro = f"LAYOUT_{layout.geometry.name}"
        row0 = ", ".join(codes[:10])
        row1 = ", ".join(codes[10:20])
        row2 = ", ".join(codes[20:30])
        thumbs = ", ".join(codes[30:])

    return (
        f"// Generated by ergo-keymap-optimizer ({layout.name})\n"
        f"[{layer_name}] = {macro}(\n"
        f"    {row0},\n"
        f"    {row1},\n"
        f"    {row2},\n"
        f"    {thumbs}\n"
        f")"
    )


def export_zmk_keymap(layout: KeymapLayout, layer_name: str = "default_layer") -> str:
    """Export a `KeymapLayout` to a valid ZMK devicetree `.keymap` layer node."""
    codes = [_to_zmk_code(s) for s in layout.symbols]
    row0 = "  ".join(codes[0:10])
    row1 = "  ".join(codes[10:20])
    row2 = "  ".join(codes[20:30])
    thumbs = "  ".join(codes[30:])
    return (
        f"/* Generated by ergo-keymap-optimizer ({layout.name}) */\n"
        f"{layer_name} {{\n"
        f"    bindings = <\n"
        f"        {row0}\n"
        f"        {row1}\n"
        f"        {row2}\n"
        f"        {thumbs}\n"
        f"    >;\n"
        f"}};"
    )
