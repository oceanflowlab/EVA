"""POPE-only explicit tool-need gates; keep the existing EVA model passes."""
import copy
import json

from .eva_internvl_pope_round2 import InternVL3POPERound2ToolAgent
from .multi_agent import TOOL_ARG_SPECS


INTERNVL_POPE_ROUND3_PROFILES = {'p3_gate13': {'anchor': 'p2_strict_plain', 'all_max_tiles': 13}}

VISUAL_GATE = (
    'Answer the exact question from the original image first. Then decide whether an external '
    'tool can reveal a necessary visual detail you still cannot resolve. '
    'Use need_tools=false and tool_calls=[] when direct inspection is sufficient, for either '
    'Yes or No. Recognizing an ordinary visible object does not require a detector. '
    'Set need_tools=true only for a specific missing detail that could change the answer, '
    'and describe that detail briefly in plan. A desire to verify your answer is not a missing detail. '
    'When detection is useful, query only the requested category as a short noun phrase.'
)

PARTS_SCAN = (
    'Inspect the foreground, background and image edges for identifiable visible parts of '
    'the requested category, including small and partially occluded instances. '
    'Use the depicted features to identify the object rather than its usual scene companions.'
)


class InternVL3POPERound3ToolAgent(InternVL3POPERound2ToolAgent):
    def __init__(self, benchmark_family='POPE', benchmark_variant='p3_gate13', **kwargs):
        if benchmark_family != 'POPE' or benchmark_variant not in INTERNVL_POPE_ROUND3_PROFILES:
            raise ValueError('Unknown InternVL POPE round-three profile or non-POPE family')
        options = copy.deepcopy(INTERNVL_POPE_ROUND3_PROFILES[benchmark_variant])
        if 'all_max_tiles' in options:
            kwargs['base_max_tiles'] = options['all_max_tiles']
        super().__init__(benchmark_family='POPE', benchmark_variant=options['anchor'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._pope_round3_options = options
        self._benchmark_options.update({k: v for k, v in options.items() if k != 'anchor'})
        self._pope_gate_records = []
        self._pope_gate_pending, self._pope_gate_current = False, None

    def generate_inner(self, message, dataset=None):
        self._pope_gate_records = []
        self._pope_gate_pending, self._pope_gate_current = False, None
        return super().generate_inner(message, dataset)

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        specs = copy.deepcopy({tool: TOOL_ARG_SPECS.get(tool, {}) for tool in self.tools})
        specs['DrawLine']['format']['coordinates'] = (
            'Nonempty JSON array of integer absolute pixel positions: x for vertical, y for horizontal.')
        descriptions = {
            'ocr': 'Read relevant visible text.',
            'GroundingDINO': 'Suggest locations of a requested object category.',
            'DrawLine': 'Annotate a horizontal or vertical alignment.',
            'DrawShape': 'Annotate an identifiable spatial comparison.',
            'GetBarInfo': 'Read structured values from an actual chart.',
        }
        prompt = (VISUAL_GATE + '\nReturn only valid JSON: '
                  '{"final_answer": string, "plan": [string], "need_tools": boolean, '
                  '"tool_calls": [{"name": string, "args": object}]}. '
                  'final_answer must be exactly Yes or No. Keep plan within 16 words. '
                  'Use at most one tool call this round. Image is injected by the system.\n'
                  + 'Available tools: ' + ', '.join(self.tools) + '\n'
                  + 'Tool descriptions: ' + json.dumps(descriptions) + '\n'
                  + 'Tool argument specs: ' + json.dumps(specs))
        if self._pope_round3_options.get('localize_gate'):
            prompt += ('\nUse GroundingDINO only when the missing detail is where a plausible '
                       'instance is located. If identity is already visually clear, answer directly. '
                       'If the difficulty is distinguishing fine visual features rather than '
                       'localizing an instance, inspect those features in the image.')
        if self._pope_round3_options.get('parts_scan'):
            prompt += '\n' + PARTS_SCAN
        if self._pope_round2_options.get('visual_candidate'):
            prompt += ('\nPut the current image-based answer in final_answer before choosing tools. '
                       'Keep that visual judgement available for later evidence comparison.')
            self._pope_capture_plan = True
        if extra_context:
            prompt += ('\nPrevious evidence and tool history:\n' + extra_context
                       + '\nThis is the only replanning round. Stop with need_tools=false '
                       'once the image and observations resolve the question. Request a call '
                       'only for a different remaining necessary visual detail.')
        self._pope_gate_pending, self._pope_gate_current = True, None
        return prompt + '\n\nQuestion:\n' + question

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        capture = self._pope_gate_pending
        self._pope_gate_pending = False
        paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
        originals = list(getattr(self, '_benchmark_images', []))
        paths = originals + [p for p in paths if p and p not in originals]
        raw = super()._run_vision_prompt(paths, prompt, dataset)
        if capture:
            payload = self._extract_json(raw)
            value = payload.get('need_tools') if isinstance(payload, dict) else None
            # Missing/invalid decisions keep the ordinary parsed-call policy, not a guessed gate.
            record = {'need_tools': value if isinstance(value, bool) else None,
                      'decision_valid': isinstance(value, bool),
                      'visual_answer': payload.get('final_answer') if isinstance(payload, dict) else None,
                      'plan': copy.deepcopy(payload.get('plan', [])) if isinstance(payload, dict) else []}
            self._pope_gate_records.append(record)
            self._pope_gate_current = record
        return raw

    def _normalize_tool_calls(self, tool_calls):
        calls = super()._normalize_tool_calls(tool_calls)
        record = self._pope_gate_current
        if record is None:
            return calls
        record['proposed_calls'] = copy.deepcopy(tool_calls)
        selected = [] if record['need_tools'] is False else calls
        record['calls_before_gate'] = copy.deepcopy(calls)
        record['selected_calls'] = copy.deepcopy(selected)
        record['suppressed_calls'] = len(calls) - len(selected)
        self._pope_gate_current = None
        return selected

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        if self._pope_round3_options.get('brief_evidence'):
            return (question + '\n\nInspect the original image and use the tool observations '
                    'to locate relevant details. Choose Yes or No from the image. '
                    'Return only valid JSON: {"final_answer": string}.\n\n'
                    + json.dumps(self._evidence_payload(question, plan, tool_calls, tool_results),
                                 ensure_ascii=False))
        prompt = super()._build_evidence_synthesis_prompt(question, plan, tool_calls, tool_results)
        if self._pope_round3_options.get('parts_scan'):
            prompt += '\n' + PARTS_SCAN
        return prompt

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls,
                                                 tool_results, evidence_payload):
        if self._pope_round3_options.get('brief_evidence'):
            payload = self._evidence_payload(question, plan, tool_calls, tool_results)
            payload['candidate'] = evidence_payload
            return (question + '\n\nCheck the candidate against the original image. Keep it '
                    'unless a concrete visible feature supports the opposite answer. '
                    'Return only valid JSON: {"final_answer": string}.\n\n'
                    + json.dumps(payload, ensure_ascii=False))
        prompt = super()._build_evidence_consistency_check_prompt(
            question, plan, tool_calls, tool_results, evidence_payload)
        if self._pope_round3_options.get('parts_scan'):
            prompt += '\n' + PARTS_SCAN
        return prompt

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(internvl_pope_round3=True,
                             internvl_pope_round3_options=copy.deepcopy(self._pope_round3_options),
                             pope_source_view_policy='originals_plus_current_annotation',
                             pope_tool_need_gates=copy.deepcopy(self._pope_gate_records))
        return super()._finalize_intermediate_records(extra_records, final_answer)
