"""Comprehensive pytest test suite for ergo-keymap-optimizer."""

from __future__ import annotations

import pytest

from ergo_keymap import (
    CorpusProfile,
    Finger,
    Hand,
    KeyboardGeometry,
    KeymapLayout,
    LayoutScorer,
    ScoringWeights,
    SimulatedAnnealingOptimizer,
    export_qmk_keymap,
    export_zmk_keymap,
)


def test_corne_36_geometry_structure_and_hands() -> None:
    geom = KeyboardGeometry.corne_36()
    assert geom.name == "corne_36"
    assert geom.num_keys == 36

    left_keys = [k for k in geom.keys if k.hand == Hand.LEFT]
    right_keys = [k for k in geom.keys if k.hand == Hand.RIGHT]
    thumb_keys = [k for k in geom.keys if k.is_thumb]
    home_keys = [k for k in geom.keys if k.is_home]

    assert len(left_keys) == 18
    assert len(right_keys) == 18
    assert len(thumb_keys) == 6
    assert len(home_keys) == 10  # 8 finger home keys + 2 primary thumb home keys


def test_ferris_34_and_corne_42_geometries() -> None:
    g34 = KeyboardGeometry.ferris_34()
    g42 = KeyboardGeometry.corne_42()

    assert g34.num_keys == 34
    assert sum(1 for k in g34.keys if k.is_thumb) == 4
    assert [k.index for k in g34.keys] == list(range(34))

    assert g42.num_keys == 42
    assert sum(1 for k in g42.keys if k.finger == Finger.PINKY) == 12


def test_corpus_profile_ngram_extraction_and_normalization() -> None:
    corpus = CorpusProfile.from_text("ababa", name="toy")
    assert corpus.total_chars == 5
    assert pytest.approx(sum(corpus.monograms.values()), rel=1e-6) == 1.0
    assert pytest.approx(sum(corpus.bigrams.values()), rel=1e-6) == 1.0
    assert pytest.approx(sum(corpus.skipgrams.values()), rel=1e-6) == 1.0
    assert pytest.approx(sum(corpus.trigrams.values()), rel=1e-6) == 1.0

    assert pytest.approx(corpus.monograms["a"]) == 3.0 / 5.0
    assert pytest.approx(corpus.monograms["b"]) == 2.0 / 5.0
    assert pytest.approx(corpus.bigrams[("a", "b")]) == 0.5
    assert pytest.approx(corpus.bigrams[("b", "a")]) == 0.5
    # In "ababa", 1-skipgrams are (a,a), (b,b), (a,a)
    assert pytest.approx(corpus.skipgrams[("a", "a")]) == 2.0 / 3.0
    assert pytest.approx(corpus.skipgrams[("b", "b")]) == 1.0 / 3.0


def test_default_code_corpus_contains_systems_code_tokens() -> None:
    corpus = CorpusProfile.code_corpus()
    assert corpus.total_chars > 400
    for ch in ("e", "t", "s", "r", "i", "n", "_"):
        assert ch in corpus.monograms


def test_qwerty_vs_colemak_dh_ergonomic_evaluation() -> None:
    scorer = LayoutScorer()
    corpus = CorpusProfile.code_corpus()
    qwerty = KeymapLayout.qwerty_36()
    colemak = KeymapLayout.colemak_dh_36()

    q_eval = scorer.evaluate(qwerty, corpus)
    c_eval = scorer.evaluate(colemak, corpus)

    # Colemak-DH places high-frequency letters on the home row and drastically cuts SFBs vs QWERTY
    assert c_eval.base_effort < q_eval.base_effort
    assert c_eval.sfb_rate < q_eval.sfb_rate
    assert c_eval.total_score < q_eval.total_score


def test_same_finger_bigram_sfb_detection_exact() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # On QWERTY, 'e' (left middle top row, index 2) and 'd' (left middle home row, index 12) share Finger.MIDDLE
    sfb_corpus = CorpusProfile.from_text("ed" * 50, name="ed_sfb")
    breakdown = scorer.evaluate(qwerty, sfb_corpus)

    assert breakdown.sfb_rate > 0.95
    assert breakdown.worst_sfbs[0][0] in ("e", "d")
    assert breakdown.worst_sfbs[0][1] in ("e", "d")


def test_same_finger_skipgram_sfs_detection_exact() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # 'e' (left middle) -> 'j' (right index) -> 'd' (left middle) -> ' ' (thumb) produces 0 SFBs but high SFS on 'e'_'d'
    sfs_corpus = CorpusProfile.from_text("ejd " * 40, name="ejd_sfs")
    breakdown = scorer.evaluate(qwerty, sfs_corpus)

    assert breakdown.sfb_rate == 0.0
    assert breakdown.sfs_rate > 0.20


def test_lateral_stretch_bigram_lsb_detection() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # On QWERTY, 'd' is left middle (col 2, x=2.0) and 'g' is left inner index (col 4, x=4.0), horizontal gap = 2.0u > 1.35u
    lsb_corpus = CorpusProfile.from_text("dg" * 40, name="dg_lsb")
    breakdown = scorer.evaluate(qwerty, lsb_corpus)

    assert breakdown.lsb_rate > 0.95
    assert breakdown.sfb_rate == 0.0


def test_row_jump_scissor_detection() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # On QWERTY, 'q' is left pinky top row (row 0, y=0.35) and 'x' is left ring bottom row (row 2, y=2.12)
    scissor_corpus = CorpusProfile.from_text("qx" * 40, name="qx_scissor")
    breakdown = scorer.evaluate(qwerty, scissor_corpus)

    assert breakdown.scissor_rate > 0.95


def test_inward_vs_outward_roll_classification() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # 'a' (left pinky=0) -> 's' (left ring=1) -> 'd' (left middle=2) -> 'f' (left index=3) is purely inward rolls
    inward_corpus = CorpusProfile.from_text("asdf " * 40, name="inward")
    # 'f' (left index=3) -> 'd' (left middle=2) -> 's' (left ring=1) -> 'a' (left pinky=0) is purely outward rolls
    outward_corpus = CorpusProfile.from_text("fdsa " * 40, name="outward")

    in_eval = scorer.evaluate(qwerty, inward_corpus)
    out_eval = scorer.evaluate(qwerty, outward_corpus)

    assert in_eval.inward_roll_rate > out_eval.inward_roll_rate
    assert out_eval.outward_roll_rate > in_eval.outward_roll_rate
    assert in_eval.total_score < out_eval.total_score


def test_hand_alternation_and_left_hand_share() -> None:
    scorer = LayoutScorer()
    qwerty = KeymapLayout.qwerty_36()
    # 'f' (left index) and 'j' (right index) alternate hands on every bigram
    alt_corpus = CorpusProfile.from_text("fj" * 50, name="alternating")
    breakdown = scorer.evaluate(qwerty, alt_corpus)

    assert pytest.approx(breakdown.hand_alternation_rate, rel=1e-2) == 1.0
    assert pytest.approx(breakdown.left_hand_share, rel=1e-2) == 0.5


def test_simulated_annealing_improves_qwerty_on_code_corpus() -> None:
    corpus = CorpusProfile.code_corpus()
    qwerty = KeymapLayout.qwerty_36()
    optimizer = SimulatedAnnealingOptimizer(
        initial_temp=0.40,
        cooling_rate=0.985,
        seed=1337,
    )

    result = optimizer.optimize(qwerty, corpus, iterations=350)
    assert result.best_breakdown.total_score < result.initial_breakdown.total_score
    assert result.accepted_moves > 0
    # Thumb keys must remain untouched by default
    for thumb_idx in range(30, 36):
        assert result.best_layout.symbols[thumb_idx] == qwerty.symbols[thumb_idx]
    # Symbol multiset must be preserved without loss or duplication
    assert sorted(result.best_layout.symbols) == sorted(qwerty.symbols)


def test_simulated_annealing_is_deterministic_with_fixed_seed() -> None:
    corpus = CorpusProfile.code_corpus()
    qwerty = KeymapLayout.qwerty_36()

    opt1 = SimulatedAnnealingOptimizer(seed=2026).optimize(qwerty, corpus, iterations=120)
    opt2 = SimulatedAnnealingOptimizer(seed=2026).optimize(qwerty, corpus, iterations=120)

    assert opt1.best_layout.symbols == opt2.best_layout.symbols
    assert opt1.best_breakdown.total_score == opt2.best_breakdown.total_score


def test_genetic_order_crossover_preserves_symbol_bijection_and_pins() -> None:
    qwerty = KeymapLayout.qwerty_36()
    colemak = KeymapLayout.colemak_dh_36()
    optimizer = SimulatedAnnealingOptimizer(seed=99)

    child = optimizer.genetic_crossover(qwerty, colemak)
    assert child.geometry.num_keys == 36
    assert sorted(child.symbols) == sorted(qwerty.symbols)
    # Pinned thumb keys preserved
    for thumb_idx in range(30, 36):
        assert child.symbols[thumb_idx] == qwerty.symbols[thumb_idx]


def test_export_qmk_and_zmk_keymap_formats() -> None:
    colemak = KeymapLayout.colemak_dh_36()
    qmk_code = export_qmk_keymap(colemak, layer_name="_COLEMAK_DH")
    zmk_code = export_zmk_keymap(colemak, layer_name="colemak_dh_layer")

    assert "[_COLEMAK_DH] = LAYOUT_split_3x5_3(" in qmk_code
    assert "KC_Q, KC_W, KC_F, KC_P, KC_B" in qmk_code
    assert "KC_TAB, KC_SPC, KC_ENT, KC_MINS, KC_QUOT, KC_BSPC" in qmk_code

    assert "colemak_dh_layer {" in zmk_code
    assert "bindings = <" in zmk_code
    assert "&kp Q  &kp W  &kp F  &kp P  &kp B" in zmk_code
    assert "&kp TAB  &kp SPACE  &kp RET  &kp MINUS  &kp SQT  &kp BSPC" in zmk_code


def test_validation_and_error_handling() -> None:
    geom = KeyboardGeometry.corne_36()
    with pytest.raises(ValueError):
        KeymapLayout(name="bad_len", geometry=geom, symbols=("a", "b"))

    with pytest.raises(ValueError):
        CorpusProfile.from_text("", name="empty")

    with pytest.raises(ValueError):
        SimulatedAnnealingOptimizer(initial_temp=0.0)

    with pytest.raises(ValueError):
        SimulatedAnnealingOptimizer(cooling_rate=1.2)


def test_layout_swap_immutability_and_char_index_map() -> None:
    qwerty = KeymapLayout.qwerty_36()
    idx_q = qwerty.char_to_index["q"]
    idx_p = qwerty.char_to_index["p"]

    swapped = qwerty.swap(idx_q, idx_p, name="qp_swapped")
    assert swapped.name == "qp_swapped"
    assert swapped.char_to_index["q"] == idx_p
    assert swapped.char_to_index["p"] == idx_q
    assert qwerty.char_to_index["q"] == idx_q


def test_export_ferris_34_qmk_macro() -> None:
    geom34 = KeyboardGeometry.ferris_34()
    syms34 = KeymapLayout.qwerty_36().symbols[:30] + ("\t", " ", "\n", "\b")
    layout34 = KeymapLayout(name="ferris_qwerty", geometry=geom34, symbols=syms34)
    qmk = export_qmk_keymap(layout34, layer_name="_SWEEP")
    assert "LAYOUT_split_3x5_2(" in qmk
    assert "KC_TAB, KC_SPC, KC_ENT, KC_BSPC" in qmk
