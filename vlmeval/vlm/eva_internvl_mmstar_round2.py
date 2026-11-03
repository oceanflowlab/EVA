"""MMStar followups that also vary routing, tool verification and routed vision."""
import copy

from .eva_internvl_mmstar_followup import InternVL3MMStarFollowupToolAgent


INTERNVL_MMSTAR_ROUND2_PROFILES = {'s2_exam_sparse': {'anchor': 'm_exam', 'sparse_gate': 'strict', 'dedup': True, 'round_call_limit': 1, 'failure_hint': True, 'evidence_trust': True}}


KNOWLEDGE_INSTRUCTION = (
    'Use relevant subject knowledge together with the visible givens. For science or technology, '
    'identify the governing mechanism, relation or circuit before selecting an option. '
    'For mathematical questions, compute from labels and given values; check units, signs, '
    'negation and comparison direction. A diagram need not be drawn to scale. '
    'A tool failure or missing detection supplies no evidence for any option. '
    'State only decisive reasoning, within 96 words, and select one offered option.'
)


class InternVL3MMStarRound2ToolAgent(InternVL3MMStarFollowupToolAgent):
    def __init__(self, benchmark_family='MMStar', benchmark_variant='s2_exam_sparse', **kwargs):
        if benchmark_family != 'MMStar' or benchmark_variant not in INTERNVL_MMSTAR_ROUND2_PROFILES:
            raise ValueError('Unknown InternVL MMStar round-two profile or non-MMStar family')
        options = copy.deepcopy(INTERNVL_MMSTAR_ROUND2_PROFILES[benchmark_variant])
        if 'all_max_tiles' in options:
            kwargs['base_max_tiles'] = options['all_max_tiles']
        super().__init__(benchmark_family='MMStar', benchmark_variant=options['anchor'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._round2_options = options
        if 'words' in options:
            self._followup_options['words'] = options['words']
            self._benchmark_options['words'] = options['words']
        self._tuning_options.update({key: value for key, value in options.items() if key in (
            'sparse_gate', 'dedup', 'round_call_limit', 'failure_hint', 'evidence_trust',
            'visual_verifier', 'compression', 'detection_limit', 'drawing_gate')})
        if options.get('native_order'):
            self._evidence_options['native_order'] = True

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        prompt = super()._build_plan_prompt(question, include_final_answer, extra_context)
        if self._round2_options.get('capability_gate'):
            prompt += (
                '\nRequest a tool only for a concrete fact that this tool can actually establish. '
                'GroundingDINO localizes described object categories; it does not determine fine '
                'attributes, species, intentions, scientific mechanisms or exact counts reliably. '
                'OCR reads visible text but cannot infer missing labels. Use GetBarInfo only for '
                'a real chart/table. If the unresolved part is knowledge or logical inference '
                'rather than an obtainable visual fact, use tool_calls=[] and answer visually. '
                'Do not call a detector merely to verify an already visible object.'
            )
        return prompt

    def _build_direct_answer_prompt(self, question):
        if self._round2_options.get('native_short') and getattr(self, '_direct_answer_active', False):
            return (question + '\n\nAnswer the question using the image and relevant knowledge. '
                    "Answer with only the correct option's letter. " + self._answer_rule())
        prompt = super()._build_direct_answer_prompt(question)
        if self._round2_options.get('knowledge'):
            prompt += '\n' + KNOWLEDGE_INSTRUCTION
        return prompt

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        prompt = super()._build_evidence_synthesis_prompt(question, plan, tool_calls, tool_results)
        if self._round2_options.get('knowledge'):
            prompt += '\n' + KNOWLEDGE_INSTRUCTION
        return prompt

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls,
                                                 tool_results, evidence_payload):
        prompt = super()._build_evidence_consistency_check_prompt(
            question, plan, tool_calls, tool_results, evidence_payload)
        if self._round2_options.get('knowledge'):
            prompt += '\n' + KNOWLEDGE_INSTRUCTION
        return prompt

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(internvl_mmstar_round2=True,
                             internvl_mmstar_round2_options=copy.deepcopy(self._round2_options))
        return super()._finalize_intermediate_records(extra_records, final_answer)
