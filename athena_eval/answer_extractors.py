"""Utility functions to extract answers from model outputs for each task."""

from __future__ import annotations

import re
from typing import Callable, Dict

# Strip common prefixes like "Answer:", "Final Answer:", "Prediction:", etc.
_PREFIX_RE = re.compile(
    r'^\s*(?:final\s+answer|answer|prediction|output|result)\s*[:\-–—]?\s*',
    re.IGNORECASE,
)

def _strip_prefix(s: str) -> str:
    return _PREFIX_RE.sub("", s).strip()

def _extract_from_lines(text: str, pattern: str, transform=lambda x: x) -> str:
    """Search *text* from bottom to top and return the last regex match.

    If a line containing the word ``Answer`` is encountered and does not match
    the pattern, the preceding and following lines are also checked to handle
    outputs of the form::

        Answer:
        <actual answer>
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        raw = lines[i]
        line = _strip_prefix(raw)

        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return transform(match.group(1))

        # If the *raw* line looked like an "answer" label, also check neighbors
        if re.search(r'\banswer\b', raw, re.IGNORECASE):
            if i + 1 < len(lines):
                nxt = _strip_prefix(lines[i + 1])
                match = re.search(pattern, nxt, re.IGNORECASE)
                if match:
                    return transform(match.group(1))
            if i > 0:
                prv = _strip_prefix(lines[i - 1])
                match = re.search(pattern, prv, re.IGNORECASE)
                if match:
                    return transform(match.group(1))
    return ""

def _clean_freeform(s: str) -> str:
    # Trim and remove surrounding quotes if present
    s = s.strip()
    s = s.strip('"\'')

    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def extract_rcm_answer(text: str) -> str:
    """Extract CWE identifier from an RCM model response."""
    return _extract_from_lines(text, r"(CWE-\d+)", lambda s: s.upper())

def extract_vsp_answer(text: str) -> str:
    """Extract CVSS vector string from a CVSS model response."""
    return _extract_from_lines(text, r"(CVSS:3\.1/[^\s]+)", lambda s: s.strip())

def extract_taa_answer(text: str) -> str:
    """Extract threat actor name from a TAA model response."""
    # Free-form name: after prefix stripping, grab the line and clean quotes/spacing
    return _extract_from_lines(text, r"(.+)", _clean_freeform)


def extract_rms_answer(text: str) -> str:
    """Extract mitigation technique IDs from an RMS model response.

    Returns a comma separated string of ``M####`` identifiers. If multiple IDs
    are present, they are returned in the order found.
    """
    line = _extract_from_lines(text, r"(.+)", _clean_freeform).upper()
    ids = re.findall(r"M\d{4}", line)
    return ", ".join(ids)


def extract_ate_answer(text: str) -> str:
    """Extract the technique identifier from an ATE model response.

    Only the first ``T####`` (optionally with ``.xxx`` subtechnique) is
    returned, normalised to the top-level technique ID.
    """
    tid = _extract_from_lines(text, r"(T\d{4}(?:\.\d{3})?)", lambda s: s.upper())
    return tid.split(".")[0]

_EXTRACTORS: Dict[str, Callable[[str], str]] = {
    "RCM": extract_rcm_answer,
    "VSP": extract_vsp_answer,
    "TAA": extract_taa_answer,
    "RMS": extract_rms_answer,
    "ATE": extract_ate_answer,
    "MCQ": lambda text: _extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper()),
    # Alias MCQ3k to the standard MCQ extractor
    "MCQ3K": lambda text: _extract_from_lines(text, r"\b([A-E])\b", lambda s: s.upper()),
}

def extract_answer(task: str, text: str) -> str:
    """Return the parsed answer for *task* from *text*.

    Parameters
    ----------
    task:
        Name of the task (e.g. ``"RCM"``).
    text:
        Model output text.
    """
    func = _EXTRACTORS.get(task.upper())
    return func(text) if func else ""
