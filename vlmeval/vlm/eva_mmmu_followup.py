"""MMMU-only EVA followups, without extra model passes or label-dependent rules."""
import copy
import json
import re

from .eva_benchmarks import normalize_choice
from .eva_ceiling import Qwen2VLCeilingToolAgent


MMMU_FOLLOWUP_PROFILES = {'r_plain': {'base': 'reasoned', 'mode': 'plain_final', 'words': 96, 'parse_final': True}}


def explicit_final_answer(text, choices):
    """Extract explicit answer fields only; conflicting declarations stay unresolved."""
    value = str(text).strip()
    candidates = []
    json_text = '\n'.join(value.splitlines()[1:-1]) if value.startswith('```') and value.endswith('```') else value
    try:
        payload = json.loads(json_text)
        if isinstance(payload, dict) and isinstance(payload.get('final_answer'), str):
            candidates.append(payload['final_answer'])
    except (ValueError, TypeError):
        pass
    candidates.extend(re.findall(r'<final_answer>\s*(.*?)\s*</final_answer>', value,
                                  flags=re.IGNORECASE | re.DOTALL))
    candidates.extend(re.findall(r'^\s*(?:FINAL(?:_ANSWER)?|FINAL ANSWER|ANSWER)\s*[:=]\s*(.+?)\s*$',
                                 value, flags=re.IGNORECASE | re.MULTILINE))
    normalized = []
    for candidate in candidates:
        candidate = candidate.strip().strip('"\'').strip()
        normalized.append(normalize_choice(candidate, choices) if choices else candidate)
    if not normalized or len(set(normalized)) != 1:
        return None
    answer = normalized[0]
    if not answer or choices and answer not in choices:
        return None
    return answer


class Qwen2VLMMMUFollowupToolAgent(Qwen2VLCeilingToolAgent):
    def __init__(self, benchmark_family='MMMU', benchmark_variant='r_plain', **kwargs):
        if benchmark_family != 'MMMU' or benchmark_variant not in MMMU_FOLLOWUP_PROFILES:
            raise ValueError('Unknown MMMU followup variant or non-MMMU dataset family')
        options = copy.deepcopy(MMMU_FOLLOWUP_PROFILES[benchmark_variant])
        super().__init__(benchmark_family='MMMU', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._followup_options = options
        self._benchmark_options.update({k: v for k, v in options.items() if k != 'base'})
        self._tuning_options.update({k: v for k, v in options.items() if k in (
            'direct_min_pixels', 'router', 'dedup', 'round_call_limit', 'drawing_gate')})

    def generate_inner(self, message, dataset=None):
        self._followup_extractions = []
        return super().generate_inner(message, dataset)

    def _build_direct_answer_prompt(self, question):
        mode = self._followup_options.get('mode')
        if not mode or not getattr(self, '_direct_answer_active', False):
            return super()._build_direct_answer_prompt(question)
        common = ('Solve the question using the supplied images and relevant subject knowledge. '
                  'Read the necessary labels and numbers, then compute or infer the requested quantity. '
                  'Use all relevant Source Images. Do not infer a physical measurement from a schematic scale. ')
        if getattr(self, '_benchmark_choices', ()):
            common += ('Select the best-supported offered option. Do not answer "none of the options" '
                       'unless that is explicitly an offered option. ')
        else:
            common += 'This is an open question: provide the computed value or concise term, with needed units. '
        words = self._followup_options.get('words', 96)
        if mode == 'plain_final':
            output = (f'Give a compact solution of at most {words} words. No JSON is needed. '
                      'End with exactly one final-answer tag: <final_answer>YOUR ANSWER</final_answer>. '
                      'Put only the answer, not an explanation, inside that tag. ')
        elif mode == 'answer_first':
            output = (f'Return only a JSON object, putting final_answer FIRST and then a reasoning '
                      f'field of at most {words} words: {{"final_answer": string, "reasoning": string}}. ')
        elif mode == 'structured_calc':
            output = (f'In at most {words} words, list only the needed givens and relation, perform '
                      'the calculation or inference, and check units/signs against the options. '
                      'Do not output a general scene description. Return only JSON: '
                      '{"givens": string, "calculation": string, "final_answer": string}. ')
        elif mode == 'exam':
            output = ('Answer as a subject-matter exam question. Work out the solution silently. '
                      'Output only FINAL: followed by the answer, with no explanation or JSON. ')
        else:
            raise ValueError(f'Unknown MMMU output mode: {mode}')
        return (common + output + self._answer_rule() + self._focus_instruction()
                + self._contrast_instruction() + '\n\nQuestion:\n' + question)

    def _normalize_final_answer(self, answer, dataset):
        if answer is not None and self._followup_options.get('parse_final'):
            choices = getattr(self, '_benchmark_choices', ())
            extracted = explicit_final_answer(answer, choices)
            if extracted is not None:
                self._followup_extractions.append({'raw_answer': str(answer), 'extracted_answer': extracted,
                                                   'parser': 'explicit final field/tag/line, unique declaration'})
                answer = extracted
        return super()._normalize_final_answer(answer, dataset)

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(mmmu_followup=True, mmmu_followup_options=copy.deepcopy(self._followup_options),
                             mmmu_explicit_answer_extractions=copy.deepcopy(self._followup_extractions))
        return super()._finalize_intermediate_records(extra_records, final_answer)
