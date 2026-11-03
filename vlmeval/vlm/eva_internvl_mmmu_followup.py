"""MMMU-only InternVL3 EVA variants, preserving the shared two-round pipeline."""
import copy

from .eva_internvl import InternVL3EVAToolAgent
from .eva_mmmu_followup import explicit_final_answer


INTERNVL_MMMU_FOLLOWUP_PROFILES = {'baseline_control': {'base': 'baseline'}}


class InternVL3MMMUFollowupToolAgent(InternVL3EVAToolAgent):
    def __init__(self, benchmark_family='MMMU', benchmark_variant='baseline_control', **kwargs):
        if benchmark_family != 'MMMU' or benchmark_variant not in INTERNVL_MMMU_FOLLOWUP_PROFILES:
            raise ValueError('Unknown InternVL MMMU followup or non-MMMU dataset family')
        options = copy.deepcopy(INTERNVL_MMMU_FOLLOWUP_PROFILES[benchmark_variant])
        super().__init__(benchmark_family='MMMU', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._followup_options = options
        self._benchmark_options.update({k: v for k, v in options.items() if k != 'base'})

    def generate_inner(self, message, dataset=None):
        self._followup_extractions = []
        return super().generate_inner(message, dataset)

    def _build_direct_answer_prompt(self, question):
        mode = self._followup_options.get('mode')
        if not mode or not getattr(self, '_direct_answer_active', False):
            return super()._build_direct_answer_prompt(question)
        words = self._followup_options['words']
        common = ('Solve the exact question using all supplied images and relevant subject knowledge. '
                  'Read necessary labels, numbers and units; distinguish given values from quantities '
                  'to infer. Apply the relevant relation or calculation rather than visual resemblance. '
                  'Do not treat a schematic as drawn to scale or invent unreadable measurements. ')
        if getattr(self, '_benchmark_choices', ()):
            common += ('Select one of the supplied options; do not answer "none of the options" '
                       'unless it is explicitly offered. ')
        else:
            common += 'No options are supplied: give the computed value or concise term, with units if needed. '
        if self._followup_options.get('contrast'):
            common += ('Check signs, units and comparison direction. Compare the strongest competing '
                       'options against the decisive evidence or calculation, not a scene stereotype. ')
        if mode == 'plain_final':
            output = (f'Give a compact solution of at most {words} words. No JSON is needed. '
                      'End with exactly one tag containing only the answer: '
                      '<final_answer>ANSWER</final_answer>. ')
        elif mode == 'answer_first':
            output = (f'Return only JSON with final_answer FIRST, then reasoning of at most {words} words: '
                      '{"final_answer": string, "reasoning": string}. ')
        elif mode == 'structured_calc':
            output = (f'Within {words} words total, list only needed givens, the governing relation '
                      'and the decisive calculation or inference. Check units and signs. '
                      'Return only JSON: {"givens": string, "calculation": string, "final_answer": string}. ')
        elif mode == 'reasoned':
            output = (f'Give task-relevant reasoning of at most {words} words, not a general image description. '
                      'Return only JSON: {"reasoning": string, "final_answer": string}. ')
        else:
            raise ValueError(f'Unknown InternVL MMMU output mode: {mode}')
        return common + output + self._answer_rule() + '\n\nQuestion:\n' + question

    def _normalize_final_answer(self, answer, dataset):
        if answer is not None and self._followup_options.get('parse_final'):
            extracted = explicit_final_answer(answer, getattr(self, '_benchmark_choices', ()))
            if extracted is not None:
                self._followup_extractions.append({'raw_answer': str(answer), 'extracted_answer': extracted,
                                                   'parser': 'unique explicit final field/tag/line'})
                answer = extracted
        return super()._normalize_final_answer(answer, dataset)

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(internvl_mmmu_followup=True,
                             internvl_mmmu_followup_options=copy.deepcopy(self._followup_options),
                             internvl_mmmu_answer_extractions=copy.deepcopy(self._followup_extractions))
        return super()._finalize_intermediate_records(extra_records, final_answer)
