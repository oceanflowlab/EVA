"""Native exact-matching evaluation utilities; no remote judge in this release."""
DEBUG_MESSAGE = 'This release uses local exact_matching evaluation only.'


def build_judge(**kwargs):
    raise ValueError('Only exact_matching is supported; no API key or remote judge is needed.')

