"""Reference grading, the perturbation, and the miner-side generators."""

from __future__ import annotations

import pytest

from relearn_eval.generators import disjoint, generate
from relearn_eval.grading import grade_choice, grade_reference, grade_rubric, normalize
from relearn_eval.perturb import PERTURBATION_VERSION, perturb_prompt


def test_normalisation_ignores_case_accents_and_punctuation():
    assert normalize("  Voilà, the ANSWER! ") == "voila the answer"


def test_a_reference_answer_matches_on_a_word_boundary():
    assert grade_reference("The answer is 144.", "144") == 1.0
    assert grade_reference("I think it is 1440", "144") == 0.0
    assert grade_reference("carbon dioxide, mostly", "CO2", ("carbon dioxide",)) == 1.0
    assert grade_reference("", "144") == 0.0


def test_a_choice_is_read_from_the_letter():
    assert grade_choice("C", "C") == 1.0
    assert grade_choice("The answer is C.", "C") == 1.0
    assert grade_choice("B", "C") == 0.0
    assert grade_choice("I cannot say", "C") == 0.0


def test_a_rubric_scores_coverage_and_order():
    rubric = ("read the logs", "roll back", "verify")
    assert grade_rubric("read the logs, roll back, then verify", rubric) == 1.0
    assert grade_rubric("verify, roll back, read the logs", rubric) == pytest.approx(1 / 3)
    assert grade_rubric("read the logs then verify", rubric) == pytest.approx(2 / 3)
    assert grade_rubric("no plan", rubric) == 0.0
    assert grade_rubric("anything", ()) == 0.0


def test_the_perturbation_is_deterministic_and_preserves_the_question():
    prompt = "Explain   why  the sky is blue."
    once = perturb_prompt(prompt)
    assert once == perturb_prompt(prompt)
    assert "Explain why the sky is blue." in once
    assert once != prompt
    assert PERTURBATION_VERSION == 1


def test_the_generators_stay_disjoint_and_deterministic():
    train, evaluation = disjoint("seed-1", 16)
    assert len(train) == len(evaluation) == 16
    assert not set(train) & set(evaluation)
    assert generate("train", "seed-1-train", 4) == train[:4]
    with pytest.raises(ValueError, match="kind must be"):
        generate("holdout", "seed", 1)
