"""Shared helpers for HAR agent nodes."""
import re


def split_labeled_sections(text: str, fields: dict[str, str]) -> dict[str, str]:
    """Split free text into sections marked by the paper's template headers.

    `fields` maps an output key to a regex pattern (matched case-insensitively)
    for that section's header. The ICA extraction and PDA diagnosis prompts
    don't mandate JSON output, so nodes split on the same header vocabulary
    the paper's own clinical-note template uses (e.g. "Medical history:",
    "Initial diagnosis:") instead of parsing free-form prose.
    """
    markers = []
    for field, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            markers.append((match.start(), match.end(), field))
    markers.sort()

    sections = {field: "" for field in fields}
    for i, (_start, end, field) in enumerate(markers):
        stop = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        sections[field] = text[end:stop].strip(" \n:-")

    return sections
