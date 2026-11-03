"""Label-blind final-answer extraction. Never use reference answers here."""
import json
import re


POLICY_VERSION = 'compatible_v1'


def question_options(question):
    """Read labeled options, keeping multiline option text for exact text matching."""
    question = str(question or '')
    if '\nOptions:\n' in question:
        question = question.rsplit('\nOptions:\n', 1)[1]
    matches = list(re.finditer(r'^\s*([A-Z])[.)]\s+([^\n]*)', question, re.MULTILINE))
    return {m[1]: question[m.start(2):matches[i + 1].start() if i + 1 < len(matches)
                           else len(question)].strip()
            for i, m in enumerate(matches)}


def _views(text):
    # Do not use unicode_escape: it corrupts LaTeX and non-ASCII answers.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    yield text
    for _ in range(2):
        unescaped = text.replace('\\"', '"')
        if unescaped == text:
            break
        text = unescaped
        yield text


def _clean(value):
    text = str(value).strip().strip('`*_ \t\r\n')
    for _ in range(3):
        if len(text) > 1 and (text[0], text[-1]) in (('"', '"'), ("'", "'"), ('(', ')'), ('[', ']')):
            text = text[1:-1].strip()
        else:
            break
    return text


def _canonical(value, task, options):
    text = _clean(value)
    if task == 'open':
        return text if text and text.lower() not in ('null', 'none', 'string') else None
    text = _clean(text.rstrip('.!'))
    boxed = re.fullmatch(r'\$?\\(?:boxed|fbox)\{([^{}]+)\}\$?', text)
    if boxed:
        text = _clean(boxed[1])
    if task == 'binary':
        aliases = {'yes': 'Yes', 'true': 'Yes', 'yeah': 'Yes', 'yep': 'Yes',
                   'no': 'No', 'false': 'No', 'nope': 'No'}
        if text.lower().rstrip('.!') in aliases:
            return aliases[text.lower().rstrip('.!')]
        words = set(re.findall(r'\b(?:yes|no)\b', text.lower()))
        if len(words) == 1:
            return words.pop().capitalize()
        return None
    match = re.fullmatch(r'(?:(?:the\s+)?(?:correct\s+)?(?:option|choice)\s*)?'
                         r'[\[(]?([A-Z])[\])]?(?:\s*[.:])?', text, re.IGNORECASE)
    if match and match[1].upper() in options:
        return match[1].upper()
    # Also accept a letter followed by its option text, but not "A or B".
    match = re.fullmatch(r'[\[(]?([A-Z])[\])]?[.:\s]+(.+)', text, re.DOTALL)
    if match and match[1] in options and options[match[1]]:
        if _clean(match[2]).casefold().rstrip('.') == _clean(options[match[1]]).casefold().rstrip('.'):
            return match[1]
    if options:
        found = [key for key, value in options.items() if value and
                 _clean(value).casefold().rstrip('.') == text.casefold().rstrip('.')]
        if len(found) == 1:
            return found[0]
    return None


def _declarations(text):
    """Prefer explicit final declarations over generic answer mentions."""
    final, generic, boxed = [], [], []
    decoder = json.JSONDecoder()
    for view in _views(text):
        # Decode only the answer value, not the full response (which may contain LaTeX).
        for match in re.finditer(r'["\'](final[_ ]?answer|answer)["\']\s*:\s*', view, re.IGNORECASE):
            tail = view[match.end():]
            try:
                value, _ = decoder.raw_decode(tail)
            except ValueError:
                quoted = re.match(r"'([^'\n]*)'", tail)
                if not quoted:
                    continue
                value = quoted[1]
            if isinstance(value, (str, bool, int, float)):
                (final if match[1].lower() != 'answer' else generic).append(value)
        for match in re.finditer(r'<(final_answer|answer)>\s*(.*?)\s*</\1>', view,
                                 re.IGNORECASE | re.DOTALL):
            (final if match[1].lower() == 'final_answer' else generic).append(match[2])
        for line in view.splitlines():
            line = line.strip().strip('`*_ ')
            match = re.match(r'^(?:(?:therefore|thus|hence|so)[,:]?\s+)?'
                             r'(?:(?:the\s+)?(?:correct\s+)?)'
                             r'(final(?:\s+answer)?|answer|(?:correct|closest|selected|best)\s+option|choice)'
                             r'\s*[*_]*(?:is\b\s*[:=]?\s*|[:=]\s*)[*_]*\s*(.+)$', line, re.IGNORECASE)
            if match:
                value = match[2].strip().strip('*_ ')
                (final if match[1].lower().startswith('final') else generic).append(value)
        boxed.extend(re.findall(r'\\(?:boxed|fbox)\{([^{}]+)\}', view))
    return final, generic, boxed


def extract_answer(answer, task='choice', choices=(), fallback=True, choice_matcher=None):
    """Return an answer plus provenance; ambiguity is never called successful parsing.

    ``choices`` can be labels or a label-to-text mapping. Open responses never
    receive an artificial A: MMMU's evaluator uses A internally for ground truth.
    """
    if task not in ('choice', 'binary', 'open'):
        raise ValueError(f'Unknown answer task: {task}')
    options = dict(choices) if isinstance(choices, dict) else dict.fromkeys(choices, '')
    text = '' if answer is None else (json.dumps(answer) if isinstance(answer, (dict, list)) else str(answer))
    result = {'policy': POLICY_VERSION, 'task': task, 'raw_answer': text,
              'answer': text.strip(), 'method': 'unresolved', 'defaulted': False,
              'ambiguous': False}
    final, generic, boxed = _declarations(text)
    for method, candidates in (('explicit_final', final), ('explicit_answer', generic), ('boxed', boxed)):
        # Copied schema placeholders and a heading with no value are not answers.
        candidates = [value for value in candidates if _clean(value).upper() not in
                      (':', '=', 'ANSWER', 'YOUR ANSWER', 'YOUR_ANSWER', 'OPTION LETTER', 'STRING')]
        if not candidates:
            continue
        normalized = [_canonical(value, task, options) for value in candidates]
        valid = set(value for value in normalized if value is not None)
        if len(valid) == 1 and all(value is not None for value in normalized):
            result.update(answer=valid.pop(), method=method)
            return result
        # A malformed/conflicting explicit final answer must not fall through to
        # an incidental option in the explanation or an earlier generic answer.
        result['ambiguous'] = len(valid) > 1
        break
    else:
        views = list(_views(text))
        canonical = _canonical(views[-1], task, options)
        if canonical is not None:
            result.update(answer=canonical, method='canonical')
            return result
        lines = [line.strip() for line in views[-1].splitlines() if line.strip() and
                 not line.strip().startswith('```')]
        # Only a standalone final letter is safe here, never arbitrary letters
        # inside reasoning, option lists, or a truncated sentence.
        if task == 'choice' and len(lines) > 1:
            canonical = _canonical(lines[-1], task, options)
            if canonical is not None:
                result.update(answer=canonical, method='final_line')
                return result
        if task == 'choice' and choice_matcher is not None:
            # Preserve the benchmark's pre-existing option/text matching, but
            # never use it to override an explicit ambiguous final declaration.
            matched = choice_matcher(text, dict(options))
            if matched in options:
                result.update(answer=matched, method='native_choice_match')
                return result
    if task == 'open':
        result['method'] = 'open_raw'
    elif fallback and (task == 'binary' or 'A' in options):
        result.update(answer='No' if task == 'binary' else 'A',
                      method='default', defaulted=True)
    return result
