"""Opt-in, benchmark-specific Qwen EVA followups; existing aliases stay unchanged."""
import copy
import json

from .eva_benchmarks import BENCHMARK_PROFILES, Qwen2VLBenchmarkToolAgent


CEILING_PROFILES = {'reasoned': {'reasoning': True}, 'option_compare': {'reasoning': True, 'contrast': True}}
CEILING_FAMILIES = ('MMMU', 'MMStar', 'POPE')


class Qwen2VLCeilingToolAgent(Qwen2VLBenchmarkToolAgent):
    def __init__(self, benchmark_family='MMMU', benchmark_variant='control', **kwargs):
        if benchmark_family not in CEILING_FAMILIES or benchmark_variant not in CEILING_PROFILES:
            raise ValueError('Unknown EVA ceiling family or variant')
        anchor = 'task_focus' if benchmark_family == 'MMMU' else 'compact'
        super().__init__(benchmark_family=benchmark_family, benchmark_variant=anchor, **kwargs)
        self.benchmark_variant = benchmark_variant
        self._ceiling_anchor = anchor
        self._benchmark_options = copy.deepcopy(BENCHMARK_PROFILES[benchmark_family][anchor])
        changes = copy.deepcopy(CEILING_PROFILES[benchmark_variant])
        self._benchmark_options.update(changes)
        self._tuning_options.update({k: v for k, v in changes.items() if k in (
            'direct_min_pixels', 'visual_verifier', 'evidence_trust', 'failure_hint',
            'dedup', 'round_call_limit', 'sparse_gate', 'router', 'drawing_gate')})

    def generate_inner(self, message, dataset=None):
        self._ceiling_reviews = []
        return super().generate_inner(message, dataset)

    def _contrast_instruction(self):
        if not self._benchmark_options.get('contrast'):
            return ''
        return {
            'MMMU': (' Separate the observed labels and numbers from the quantities to infer. '
                     'Apply relevant subject knowledge and perform the necessary calculation. '
                     'Solve the requested relationship, check units, signs and scale, then compare '
                     'the strongest competing options. Use all relevant Source Images; do not '
                     'assume a schematic is drawn to scale.'),
            'MMStar': (' Identify the exact referenced instances, count each once, and preserve '
                       'the subject/object and comparison direction. Compare the strongest '
                       'competing options against visible details rather than scene stereotypes.'),
            'POPE': (' Inspect foreground and background, small objects and partially occluded '
                     'instances for the requested object. Distinguish that object from similar '
                     'or associated objects. An identifiable partial instance can establish '
                     'presence; a scene association cannot. An empty detector result alone '
                     'cannot establish absence. Use the same visual standard for Yes and No.'),
        }[self.benchmark_family]

    def _build_direct_answer_prompt(self, question):
        if (not self._benchmark_options.get('reasoning')
                or not getattr(self, '_direct_answer_active', False)):
            return super()._build_direct_answer_prompt(question)
        words = self._benchmark_options.get(
            'reasoning_words', {'MMMU': 96, 'MMStar': 64, 'POPE': 32}[self.benchmark_family])
        knowledge = (' Apply relevant subject knowledge to the diagram and perform the '
                     'necessary calculation; visual resemblance alone is not a solution.'
                     if self.benchmark_family == 'MMMU' else '')
        return ('Use the image and the supplied question to solve the exact task. '
                'Separate visible evidence from inference. Give a brief, task-relevant '
                f'reasoning of at most {words} words, not a general scene description. '
                'Do not invent unreadable text, measurements or objects. '
                'Return only valid JSON with schema: {"reasoning": string, "final_answer": string}. '
                + self._answer_rule() + self._focus_instruction() + knowledge + self._contrast_instruction()
                + '\n\nQuestion:\n' + question)

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        prompt = super()._build_plan_prompt(question, include_final_answer, extra_context)
        if self._benchmark_options.get('targeted_router'):
            prompt += {
                'MMMU': ('\nFor an unresolved diagram label or table entry, use OCR; for a chart '
                         'value use GetBarInfo. Tools do not solve symbolic equations or validate '
                         'a calculation. Solve readable mathematics directly. Do not draw guessed measurements.'),
                'MMStar': ('\nLocalize the exact requested object with GroundingDINO only if '
                           'localization can resolve the question; use OCR only for necessary '
                           'visible text. Do not replace visual counting with raw box counts.'),
                'POPE': ('\nIf the requested object remains unresolved, GroundingDINO may '
                         'localize it: use a short object noun phrase, not the yes/no question. '
                         'OCR and chart extraction do not normally resolve object existence. '
                         'When presence or absence is visually clear, use no tools. '
                         'Verify detections against the image; missed detections are inconclusive.'),
            }[self.benchmark_family]
        return prompt

    def _fallback_direct_answer(self, image_path, question, dataset=None):
        candidate = super()._fallback_direct_answer(image_path, question, dataset)
        if not self._benchmark_options.get('direct_review') or self._tool_call_metrics:
            return candidate
        prompt = ('Independently inspect the image and the exact question before checking the candidate. '
                  'Correct it only for a concrete visual or reasoning error; do not switch merely '
                  'to sound cautious. Do not assume the candidate is correct. '
                  'Return only JSON: {"check": "brief evidence or calculation", "final_answer": string}. '
                  + self._answer_rule() + self._focus_instruction()
                  + '\n\nQuestion:\n' + question + '\n\nCandidate:\n'
                  + json.dumps(candidate, ensure_ascii=False))
        previous = self._direct_answer_active
        self._direct_answer_active = True
        try:
            raw = self._run_vision_prompt(image_path, prompt, dataset)
        finally:
            self._direct_answer_active = previous
        payload = self._extract_json(raw)
        revised = self._normalize_final_answer(payload.get('final_answer'), dataset) if isinstance(payload, dict) else None
        allowed = ('Yes', 'No') if self.benchmark_family == 'POPE' else self._benchmark_choices
        valid = isinstance(revised, str) and bool(revised.strip()) and (not allowed or revised in allowed)
        self._ceiling_reviews.append({'candidate': candidate, 'prompt': prompt, 'raw_response': raw,
                                      'accepted': valid, 'revised_answer': revised})
        self._model_call_metrics[-1]['stage'] = 'direct_self_check'
        return revised if valid else candidate

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(ceiling_suite=True, ceiling_anchor=self._ceiling_anchor,
                             ceiling_direct_reviews=copy.deepcopy(self._ceiling_reviews),
                             ceiling_extra_model_pass_enabled=bool(self._benchmark_options.get('direct_review')),
                             ceiling_extra_model_calls=len(self._ceiling_reviews))
        return super()._finalize_intermediate_records(extra_records, final_answer)
