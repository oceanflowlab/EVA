"""MMStar-specific InternVL EVA probes with the same two-round agent structure."""
import copy

from .eva_internvl import InternVL3EVAToolAgent


INTERNVL_MMSTAR_FOLLOWUP_PROFILES = {'m_exam': {'base': 'baseline', 'mode': 'exam'}}


class InternVL3MMStarFollowupToolAgent(InternVL3EVAToolAgent):
    def __init__(self, benchmark_family='MMStar', benchmark_variant='m_exam', **kwargs):
        if benchmark_family != 'MMStar' or benchmark_variant not in INTERNVL_MMSTAR_FOLLOWUP_PROFILES:
            raise ValueError('Unknown InternVL MMStar followup or non-MMStar dataset family')
        options = copy.deepcopy(INTERNVL_MMSTAR_FOLLOWUP_PROFILES[benchmark_variant])
        super().__init__(benchmark_family='MMStar', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._followup_options = options
        self._benchmark_options.update({key: value for key, value in options.items() if key != 'base'})
        self._tuning_options.update({key: value for key, value in options.items() if key in (
            'compression', 'detection_limit', 'dedup', 'failure_hint', 'router', 'round_call_limit')})

    def _build_direct_answer_prompt(self, question):
        mode = self._followup_options.get('mode')
        if not mode or not getattr(self, '_direct_answer_active', False):
            return super()._build_direct_answer_prompt(question)
        common = ('Answer the exact multiple-choice question using all supplied images and relevant knowledge. '
                  'Compare the supplied options and select the best-supported one. '
                  'Keep the named object, attribute, count, negation and comparison direction exact. '
                  'Do not invent unreadable text or replace the requested instance with an associated one. '
                  'Do not answer "none of the options" unless that is explicitly offered. ')
        words = self._followup_options.get('words', 64)
        if mode == 'answer_first':
            output = (f'Return only JSON with final_answer FIRST, followed by reasoning of at most {words} words: '
                      '{"final_answer": string, "reasoning": string}. ')
        elif mode == 'reasoned':
            output = (f'Give only the decisive evidence or calculation in at most {words} words. '
                      'Return only JSON: {"reasoning": string, "final_answer": string}. ')
        elif mode == 'plain_final':
            output = (f'Explain only the decisive evidence or calculation in at most {words} words. '
                      'No JSON is needed. End with exactly one tag containing only the option letter: '
                      '<final_answer>OPTION LETTER</final_answer>. ')
        elif mode == 'exam':
            output = ('Treat this as an exam question. Work out the answer silently. '
                      'Output only FINAL: followed by the option letter; no explanation or JSON. ')
        elif mode == 'task_focus':
            output = ('Inspect the requested fine detail for perception. For counting, count each '
                      'distinct requested instance once. For relations, preserve subject/object order '
                      'and the reference frame. For numerical or scientific tasks, read the given '
                      'labels and values, apply the relevant rule and check units and signs; '
                      'do not infer measurements from schematic scale. '
                      f'Give a compact solution within {words} words. '
                      'Return only JSON: {"reasoning": string, "final_answer": string}. ')
        elif mode == 'visual_anchor':
            output = ('First state one decisive question-relevant visible fact in at most 12 words, '
                      'not a general scene description. Connect it to the best option; when needed, '
                      'use the given numbers, units and governing rule rather than schematic scale. '
                      f'Keep reasoning within {words} words. Return only JSON: '
                      '{"observation": string, "reasoning": string, "final_answer": string}. ')
        else:
            raise ValueError(f'Unknown InternVL MMStar output mode: {mode}')
        return common + output + self._answer_rule() + '\n\nQuestion:\n' + question

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(internvl_mmstar_followup=True,
                             internvl_mmstar_followup_options=copy.deepcopy(self._followup_options))
        return super()._finalize_intermediate_records(extra_records, final_answer)
