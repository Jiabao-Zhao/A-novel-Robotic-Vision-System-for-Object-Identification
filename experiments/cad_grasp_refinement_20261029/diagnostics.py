"""Read independently evaluated categories without rerunning full validation."""
from .geometry import CATEGORIES


def evaluate_candidate(validator, candidate, validation=None):
    """Use the validation result's complete cached category diagnostics."""
    result = validator.validate(candidate) if validation is None else validation
    values = result.get('diagnostics')
    if not isinstance(values, dict) or any(key not in values for key in CATEGORIES):
        raise ValueError('Independent diagnostics require the exhaustive refinement validator result')
    return values


def summarize(records):
    """Count passes/fails/unavailability separately; categories may overlap."""
    counts = {key: {status: 0 for status in ['pass', 'fail', 'unavailable', 'not_evaluated']}
              for key in CATEGORIES}
    for record in records:
        result = record.get('validity', record)
        for key, values in result['diagnostics'].items():
            counts[key][values['status']] += 1
    return counts
