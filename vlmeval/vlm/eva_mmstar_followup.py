"""MMStar EVA followups with unchanged model-pass and tool-round structure."""
import copy

from .eva_benchmarks import Qwen2VLBenchmarkToolAgent
from .eva_ceiling import Qwen2VLCeilingToolAgent
from .eva_mmmu_followup import explicit_final_answer


MMSTAR_FOLLOWUP_PROFILES = {'v2_answer_first': {'origin': 'ceiling', 'base': 'reasoned', 'direct_min_pixels': 1003520, 'mode': 'answer_first', 'words': 64, 'parse_final': True}}


class Qwen2VLMMStarFollowupToolAgent(Qwen2VLCeilingToolAgent):
    def __init__(self, benchmark_family='MMStar', benchmark_variant='v2_answer_first', **kwargs):
        if benchmark_family != 'MMStar' or benchmark_variant not in MMSTAR_FOLLOWUP_PROFILES:
            raise ValueError('Unknown MMStar followup variant or non-MMStar dataset family')
        options = copy.deepcopy(MMSTAR_FOLLOWUP_PROFILES[benchmark_variant])
        if options['origin'] == 'benchmark':
            # Preserve the original baseline, rather than the ceiling compact anchor.
            Qwen2VLBenchmarkToolAgent.__init__(self, benchmark_family='MMStar',
                                             benchmark_variant=options['base'], **kwargs)
            self._ceiling_anchor = options['base']
        else:
            super().__init__(benchmark_family='MMStar', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._followup_options = options
        self._benchmark_options.update({k: v for k, v in options.items() if k not in ('origin', 'base')})
        self._tuning_options.update({k: v for k, v in options.items() if k in (
            'direct_min_pixels', 'router', 'dedup', 'round_call_limit', 'drawing_gate')})

    def generate_inner(self, message, dataset=None):
        self._followup_extractions = []
        return super().generate_inner(message, dataset)

    def _build_direct_answer_prompt(self, question):
        mode = self._followup_options.get('mode')
        if not mode or not getattr(self, '_direct_answer_active', False):
            return super()._build_direct_answer_prompt(question)
        common = ('Answer the exact multiple-choice question using the image and relevant knowledge. '
                  'Compare the supplied options and select the best-supported one. '
                  'Do not invent unreadable text or objects, and do not replace the requested '
                  'instance with a similar or associated one. Do not abstain with "none of the options" '
                  'unless that is explicitly an offered option. ')
        words = self._followup_options.get('words', 48)
        if mode == 'exam':
            output = ('Treat this as an exam question. Determine the answer silently. '
                      'Output only FINAL: followed by the option letter; no explanation or JSON. ')
        elif mode == 'plain_final':
            output = (f'Explain only the decisive evidence or calculation in at most {words} words. '
                      'No JSON is needed. End with exactly one tag: '
                      '<final_answer>OPTION LETTER</final_answer>. ')
        elif mode == 'answer_first':
            output = (f'Return only JSON with final_answer FIRST, then reasoning of at most {words} words: '
                      '{"final_answer": string, "reasoning": string}. ')
        elif mode == 'entity_check':
            output = ('Bind each named entity to the correct visible instance. For counts, count '
                      'distinct requested instances once. For relations and comparisons, preserve '
                      'the subject/object order, direction and negation in the question. '
                      f'Give only decisive evidence in at most {words} words. '
                      'Return JSON: {"evidence": string, "final_answer": string}. ')
        elif mode == 'task_focus':
            output = ('Identify what resolves this question: for perception inspect the required '
                      'attribute or fine detail; for counting define what counts as one instance; '
                      'for spatial relations preserve the reference objects and direction. '
                      'For mathematical or scientific questions, read the given labels/values '
                      'and apply the relevant rule; do not estimate lengths from schematic scale. '
                      f'Give a compact solution in at most {words} words. '
                      'Return JSON: {"reasoning": string, "final_answer": string}. ')
        elif mode == 'visual_fact':
            output = ('First state one decisive question-relevant visible fact in at most 12 words, '
                      'not a general scene description. Then select the option supported by it. '
                      'Return JSON: {"observation": string, "final_answer": string}. ')
        elif mode == 'visual_anchor':
            output = ('State one decisive question-relevant visible fact in at most 12 words. '
                      'Then connect that fact to the best option; for numerical or scientific '
                      'questions use the given labels, values and relevant rule rather than '
                      'the apparent scale of a schematic. Preserve units and comparison direction. '
                      f'Keep reasoning within {words} words. '
                      'Return JSON: {"observation": string, "reasoning": string, "final_answer": string}. ')
        else:
            raise ValueError(f'Unknown MMStar output mode: {mode}')
        return common + output + self._answer_rule() + '\n\nQuestion:\n' + question

    def _normalize_final_answer(self, answer, dataset):
        if answer is not None and self._followup_options.get('parse_final'):
            extracted = explicit_final_answer(answer, getattr(self, '_benchmark_choices', ()))
            if extracted is not None:
                self._followup_extractions.append({'raw_answer': str(answer), 'extracted_answer': extracted,
                                                   'parser': 'explicit final field/tag/line, unique declaration'})
                answer = extracted
        return super()._normalize_final_answer(answer, dataset)

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(mmstar_followup=True, mmstar_followup_options=copy.deepcopy(self._followup_options),
                             mmstar_explicit_answer_extractions=copy.deepcopy(self._followup_extractions))
        return super()._finalize_intermediate_records(extra_records, final_answer)
