from har.llm import parse_json
from har.nodes.coordinator import CAReflection
from har.nodes.dda import DDAReflection, DifferentialDiagnosisProcess
from har.nodes.pda import PDAReflection


def test_response_keys_are_case_insensitive():
    result = parse_json(
        DDAReflection,
        '{"FLAG": true, "Final_Diagnosis": "stress incontinence"}',
    )

    assert result.flag is True
    assert result.final_diagnosis == "stress incontinence"


def test_text_fields_accept_lists():
    dda = parse_json(
        DDAReflection,
        '{"flag": false, "diff_error": ["urge", "overflow"]}',
    )
    pda = parse_json(
        PDAReflection,
        '{"flag": false, "diagnosis_error": ["missing history", "weak basis"]}',
    )
    coordinator = parse_json(
        CAReflection,
        '{"flag": false, "ICA_error": ["history", "examination"]}',
    )
    process = parse_json(
        DifferentialDiagnosisProcess,
        '{"differential_diagnosis_process": ["exclude urge", "exclude overflow"]}',
    )

    assert dda.diff_error == "urge, overflow"
    assert pda.diagnosis_error == "missing history, weak basis"
    assert coordinator.ica_error == "history, examination"
    assert process.differential_diagnosis_process == "exclude urge, exclude overflow"


def test_parser_ignores_markdown_and_trailing_explanation():
    result = parse_json(
        DDAReflection,
        '```json\n{"flag": false, "diff_error": "retry"}\n``` '
        'Explanation with {non-JSON braces}.',
    )

    assert result.diff_error == "retry"
