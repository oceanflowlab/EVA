"""Independent binary visual reasoning probes with unchanged EVA orchestration."""
import copy

from .eva_internvl import InternVL3EVAToolAgent


INTERNVL_HALLUSION_FOLLOWUP_PROFILES = {'h_symmetric12': {'base': 'observe12', 'symmetric': True}}

RELATION_INSTRUCTION = (
    'Ground the exact claim in the displayed image: identify the named objects, the requested '
    'attribute or relation, subject/object order and the stated reference frame. '
    'Check negation and comparison direction before deciding Yes or No. '
    'For equality, length, alignment or color, compare the requested features rather than '
    'the surrounding pattern or a remembered illusion. A schematic is not necessarily to scale. '
    'An annotation is not a measurement and missing detections do not prove absence.'
)
SYMMETRIC_INSTRUCTION = (
    'Apply the same evidential standard to Yes and No. Internally identify what visible fact '
    'would support the exact claim and what would contradict it, then judge the depicted case. '
    'Do not assume a question is misleading or false merely because the image is unusual. '
    'Do not assume a claim is true merely because it is plausible in a typical scene. '
    'Tool failure and uncertainty alone are not evidence for either answer.'
)


class InternVL3HallusionFollowupToolAgent(InternVL3EVAToolAgent):
    def __init__(self, benchmark_family='HallusionBench', benchmark_variant='h_symmetric12', **kwargs):
        if benchmark_family != 'HallusionBench' or benchmark_variant not in INTERNVL_HALLUSION_FOLLOWUP_PROFILES:
            raise ValueError('Unknown InternVL HallusionBench followup or non-HallusionBench family')
        options = copy.deepcopy(INTERNVL_HALLUSION_FOLLOWUP_PROFILES[benchmark_variant])
        super().__init__(benchmark_family='HallusionBench', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._hallusion_options = options
        self._benchmark_options.update({k: v for k, v in options.items() if k != 'base'})
        self._tuning_options.update({k: v for k, v in options.items() if k in (
            'sparse_gate', 'dedup', 'round_call_limit', 'failure_hint', 'evidence_trust',
            'visual_verifier', 'compression', 'detection_limit')})

    def _claim_instruction(self):
        if self._hallusion_options.get('relation'):
            return '\n' + RELATION_INSTRUCTION
        if self._hallusion_options.get('symmetric'):
            return '\n' + SYMMETRIC_INSTRUCTION
        return ''

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        return (super()._build_plan_prompt(question, include_final_answer, extra_context)
                + self._claim_instruction())

    def _build_direct_answer_prompt(self, question):
        return super()._build_direct_answer_prompt(question) + self._claim_instruction()

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        return (super()._build_evidence_synthesis_prompt(question, plan, tool_calls, tool_results)
                + self._claim_instruction())

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls,
                                                 tool_results, evidence_payload):
        return (super()._build_evidence_consistency_check_prompt(
            question, plan, tool_calls, tool_results, evidence_payload) + self._claim_instruction())

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(internvl_hallusion_followup=True,
                             internvl_hallusion_followup_options=copy.deepcopy(self._hallusion_options))
        return super()._finalize_intermediate_records(extra_records, final_answer)
