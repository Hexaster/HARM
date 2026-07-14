import pytest

from har.knowledge import all_diseases, differentials, key_points, standardized_note

EXPECTED_DISEASES = {
    "stress incontinence",
    "urge incontinence",
    "overflow incontinence",
    "overactive bladder",
    "lower urinary tract syndrome",
    "carcinoma of prostate",
    "hyperplasia of prostate",
    "contracture of bladder neck",
    "urethral stricture",
    "neurogenic bladder",
    "carcinoma of bladder",
    "vesical calculus",
    "urinary tract infection",
    "cystitis",
    "vaginitis",
    "urethritis",
    "prostatitis",
}


def test_all_17_diseases_load():
    assert len(EXPECTED_DISEASES) == 17
    assert set(all_diseases()) == EXPECTED_DISEASES


def test_every_differential_target_is_a_known_disease():
    for disease in EXPECTED_DISEASES:
        for target in differentials(disease):
            assert target in EXPECTED_DISEASES, f"{disease} -> {target} is not a known disease"


def test_key_points_present_for_every_disease():
    for disease in EXPECTED_DISEASES:
        assert len(key_points(disease)) >= 3


def test_standardized_note_present_for_every_disease():
    for disease in EXPECTED_DISEASES:
        note = standardized_note(disease)
        for field in (
            "medical_history",
            "physical_examination",
            "auxiliary_examination",
            "clinical_features",
            "initial_diagnosis",
            "diagnostic_basis",
            "disease_list",
            "differential_process",
            "final_diagnosis",
        ):
            assert field in note


def test_standardized_note_anchor_case_from_paper():
    note = standardized_note("stress incontinence")
    assert note["reviewed"] is True
    assert "Mirabegron" in note["medical_history"]
    assert note["final_diagnosis"] == "stress incontinence, overactive bladder"
    assert note["disease_list"] == ["urge incontinence", "overflow incontinence", "lower urinary tract syndrome"]


def test_fuzzy_disease_name_matching():
    assert differentials("Stress Incontinence") == differentials("stress incontinence")
    assert key_points("  overactive   bladder ") == key_points("overactive bladder")


def test_unknown_disease_raises():
    with pytest.raises(ValueError):
        key_points("completely unrelated made-up disease")
