"""POPE-only EVA followups with unchanged model passes and tool-round limits."""
import copy

from .eva_ceiling import Qwen2VLCeilingToolAgent
from .eva_mmmu_followup import explicit_final_answer


POPE_FOLLOWUP_PROFILES = {'c_native1280': {'base': 'option_compare', 'direct_min_pixels': 1003520}}


def explicit_binary_answer(text):
    """Accept unique explicit Yes/No declarations, never guess from reasoning."""
    answer = explicit_final_answer(text, ())
    if answer is not None:
        value = answer.strip().lower().rstrip('.')
        if value in ('yes', 'no'):
            return value.capitalize()
    return None


class Qwen2VLPOPEFollowupToolAgent(Qwen2VLCeilingToolAgent):
    def __init__(self, benchmark_family='POPE', benchmark_variant='c_native1280', **kwargs):
        if benchmark_family != 'POPE' or benchmark_variant not in POPE_FOLLOWUP_PROFILES:
            raise ValueError('Unknown POPE followup variant or non-POPE dataset family')
        options = copy.deepcopy(POPE_FOLLOWUP_PROFILES[benchmark_variant])
        super().__init__(benchmark_family='POPE', benchmark_variant=options['base'], **kwargs)
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
        common = ('Decide whether the exact object category requested in the question is visibly '
                  'present in the image. Use the same visual standard for Yes and No. '
                  'Do not infer an object from scene associations or substitute a related object. ')
        words = self._followup_options.get('words', 32)
        if mode == 'answer_first':
            instruction = (self._contrast_instruction()
                           + f' Return only JSON with final_answer FIRST, then reasoning of at most {words} words: '
                           '{"final_answer": string, "reasoning": string}. ')
        elif mode == 'partial_instance':
            instruction = ('A small or partially occluded instance can establish presence if its '
                           'visible parts identify the requested category; the whole object need not '
                           'be visible. Do not require a prominent, unobstructed or textbook instance. '
                           'A vague patch or merely likely companion object is not sufficient. '
                           'Answer No when no identifiable instance is visible, not merely because '
                           'the object is small or incomplete. '
                           f'Give decisive visible evidence in at most {words} words. '
                           'Return JSON: {"reasoning": string, "final_answer": string}. ')
        elif mode == 'object_scan':
            instruction = ('Inspect foreground, middle distance, background and image edges for '
                           'the requested category, including small or partially occluded instances. '
                           'Distinguish that category from visually similar objects. '
                           'Answer Yes if an identifiable instance is visible, and No if none is '
                           'identifiable. Do not turn a missed detection into proof of absence. '
                           f'Summarize only decisive evidence in at most {words} words. '
                           'Return JSON: {"evidence": string, "final_answer": string}. ')
        else:
            raise ValueError(f'Unknown POPE output mode: {mode}')
        return common + instruction + self._answer_rule() + '\n\nQuestion:\n' + question

    def _normalize_final_answer(self, answer, dataset):
        if answer is not None and self._followup_options.get('parse_final'):
            extracted = explicit_binary_answer(answer)
            if extracted is not None:
                self._followup_extractions.append({'raw_answer': str(answer), 'extracted_answer': extracted,
                                                   'parser': 'unique explicit final Yes/No declaration'})
                answer = extracted
        return super()._normalize_final_answer(answer, dataset)

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(pope_followup=True, pope_followup_options=copy.deepcopy(self._followup_options),
                             pope_explicit_answer_extractions=copy.deepcopy(self._followup_extractions))
        return super()._finalize_intermediate_records(extra_records, final_answer)
