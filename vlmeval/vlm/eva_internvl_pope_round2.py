"""POPE-only ablations of detector reliance, with an explicit direct control."""
import copy
import json
import os
import time

from .eva_internvl import InternVL3EVAToolAgent
from .eva_tuning import compact_detections
from .multi_agent import TOOL_ARG_SPECS, TOOL_DESCRIPTIONS


INTERNVL_POPE_ROUND2_PROFILES = {'p2_strict_plain': {'base': 'baseline', 'router': 'strict'}}

PRESENCE_RULE = (
    'Answer whether an identifiable instance of the requested category is visible in the '
    'original image. Small, partly occluded and edge-truncated instances count when their '
    'visible parts identify the category. Use the original image to decide Yes or No; '
    'detector boxes are locations to inspect. A confidence value ranks detector matches, '
    'not the probability that this question should be answered Yes. '
    'Choose the answer best supported by the image.'
)

ROUTING_RULE = (
    'Inspect the original image before choosing any tool. If you can identify the requested '
    'object, or the scene clearly resolves its absence, return tool_calls=[]. '
    'Call a tool only for a specific unresolved visual detail that it can reveal. '
    'For GroundingDINO use only the requested category as a short noun phrase; '
    'use each returned box to inspect that location in the original image. '
    'OCR is for relevant visible text, GetBarInfo for an actual chart, and drawing tools '
    'for an identifiable spatial comparison. Prefer no tool over a redundant check.'
)


class InternVL3POPERound2ToolAgent(InternVL3EVAToolAgent):
    def __init__(self, benchmark_family='POPE', benchmark_variant='p2_strict_plain', **kwargs):
        if benchmark_family != 'POPE' or benchmark_variant not in INTERNVL_POPE_ROUND2_PROFILES:
            raise ValueError('Unknown InternVL POPE round-two profile or non-POPE family')
        options = copy.deepcopy(INTERNVL_POPE_ROUND2_PROFILES[benchmark_variant])
        if 'all_max_tiles' in options:
            kwargs['base_max_tiles'] = options['all_max_tiles']
        super().__init__(benchmark_family='POPE', benchmark_variant=options['base'], **kwargs)
        self.benchmark_variant = benchmark_variant
        self._pope_round2_options = options
        self._pope_plan_candidates, self._pope_capture_plan = [], False
        self._benchmark_options.update({k: v for k, v in options.items() if k != 'base'})
        if options.get('router'):
            self._tuning_options.update(dedup=True, round_call_limit=1)

    def generate_inner(self, message, dataset=None):
        self._pope_plan_candidates = []
        self._pope_capture_plan = False
        if not self._pope_round2_options.get('direct_only'):
            return super().generate_inner(message, dataset)

        # A genuine single-pass control, without a planner or tool-dependent fallback.
        self._validate_benchmark_dataset(dataset)
        if any(item.get('type') not in ('image', 'text') for item in message):
            raise ValueError('InternVL POPE direct control accepts only text and images')
        self._benchmark_images = list(dict.fromkeys(
            os.path.abspath(item['value']) for item in message if item.get('type') == 'image'))
        if len(self._benchmark_images) != 1:
            raise ValueError('POPE direct control requires exactly one source image')
        _, question = self._extract_image_and_question(message)
        self._benchmark_choices, self._benchmark_tool_targets = (), []
        self._tuning_question = question
        self._raw_tool_observations, self._tool_selection_records = [], []
        self._model_call_metrics, self._tool_call_metrics = [], []
        self._line_coordinate_adaptations, self._direct_answer_records = [], []
        self._sample_start = time.perf_counter()
        self._direct_answer_active = True
        try:
            raw = self._run_vision_prompt(self._benchmark_images, question, dataset)
        finally:
            self._direct_answer_active = False
        record = {'method': 'InternVL3Direct', 'question': question,
                  'original_image_path': self._benchmark_images[0],
                  'tool_rounds': [], 'tool_results': [], 'native_direct_raw': raw}
        answer = self._finalize_answer(raw, dataset, record)
        if self.save_intermediate:
            self._finalize_intermediate_records(record, answer)
            return {'prediction': answer, 'extra_records': record}
        return answer

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        options = self._pope_round2_options
        if not options.get('router'):
            return super()._build_plan_prompt(question, include_final_answer, extra_context)
        schema = ('{"final_answer": string, "plan": [string], '
                  '"tool_calls": [{"name": string, "args": object}]}')
        # Build the tool schema structurally; shared descriptors remain untouched.
        specs = copy.deepcopy({tool: TOOL_ARG_SPECS.get(tool, {}) for tool in self.tools})
        specs['DrawLine']['format']['coordinates'] = (
            'Nonempty JSON array of integer absolute pixel positions: x for vertical, y for horizontal.')
        descriptions = {tool: TOOL_DESCRIPTIONS.get(tool, '') for tool in self.tools}
        descriptions['GroundingDINO'] = 'Propose object locations matching a short category query.'
        prompt = (ROUTING_RULE + '\n' + PRESENCE_RULE
                  + '\nReturn only valid JSON: ' + schema + '. '
                  'Keep the plan within 20 words. Use at most one tool call this round. '
                  'Do not include image in args; it is injected by the system.\n'
                  + 'Available tools: ' + ', '.join(self.tools) + '\n'
                  + 'Tool descriptions: ' + json.dumps(descriptions) + '\n'
                  + 'Tool argument specs: ' + json.dumps(specs))
        if options['router'] == 'strict':
            prompt += ('\nUse tools only if the unresolved detail is necessary to distinguish the '
                       'two answers and the tool is suited to reveal it. Ordinary object recognition '
                       'should be answered directly. A desire to double-check is not a reason to call.')
        if options.get('visual_candidate'):
            prompt += ('\nPut your current image-based answer in final_answer before selecting tools. '
                       'Name the visible feature supporting it; choose a tool only for a remaining '
                       'question that could change that answer.')
        if extra_context:
            prompt += ('\nPrevious evidence and tool history:\n' + extra_context
                       + '\nStop with tool_calls=[] when the image resolves the question. '
                       'Request another call only for a different remaining necessary detail. '
                       'There is at most one replanning round.')
        if options.get('visual_candidate'):
            self._pope_capture_plan = True
        return prompt + '\n\nQuestion:\n' + question

    def _build_direct_answer_prompt(self, question):
        if self._pope_round2_options.get('direct_fact'):
            return (PRESENCE_RULE + '\nReturn only valid JSON: '
                    '{"observation": "one decisive visible fact, at most 12 words", '
                    '"final_answer": string}. ' + self._answer_rule()
                    + '\n\nQuestion:\n' + question)
        return question

    def _evidence_payload(self, question, plan, tool_calls, tool_results):
        payload = {'question': question, 'plan': plan, 'tool_calls': tool_calls,
                   'tool_results': self._strip_verbose_fields(tool_results, drop_edited_image=True)}
        if self._pope_round2_options.get('visual_candidate') and self._pope_plan_candidates:
            payload['initial_visual_candidate'] = self._pope_plan_candidates[0]
        return payload

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        return (PRESENCE_RULE + '\nInspect the original image again at the relevant locations '
                'and answer the exact question. Explain the decisive visible feature, '
                'not just the detector output. Return only valid JSON: '
                '{"final_answer": string, "explanation": string}. '
                'Keep the explanation within 24 words. ' + self._answer_rule()
                + '\n\n' + json.dumps(self._evidence_payload(
                    question, plan, tool_calls, tool_results), ensure_ascii=False))

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls,
                                                 tool_results, evidence_payload):
        payload = self._evidence_payload(question, plan, tool_calls, tool_results)
        payload['candidate'] = evidence_payload
        return (PRESENCE_RULE + '\nCheck the candidate against the original image. '
                'Keep it unless an identifiable visible feature supports the opposite answer. '
                'Judge the same requested category and image regions for either answer. '
                'Return only valid JSON: {"final_answer": string, "explanation": string}. '
                'Keep the explanation within 24 words. ' + self._answer_rule()
                + '\n\n' + json.dumps(payload, ensure_ascii=False))

    def _compact_tool_result_for_agent(self, tool_name, tool_res):
        view = self._pope_round2_options.get('detector_view')
        if (tool_name != 'GroundingDINO' or not view or not isinstance(tool_res, dict)
                or tool_res.get('status') != 'success'):
            return super()._compact_tool_result_for_agent(tool_name, tool_res)
        result = compact_detections(tool_name, tool_res, self._tuning_question, limit=8)
        if 'rows' in result:
            result['rows'] = ([[row[0], row[2]] for row in result['rows']] if view == 'label_boxes'
                              else [[row[2]] for row in result['rows']])
            result['columns'] = (['label', 'bbox_xyxy_absolute_pixels'] if view == 'label_boxes'
                                 else ['bbox_xyxy_absolute_pixels'])
            result.pop('note', None)
        return result

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        capture = self._pope_capture_plan
        self._pope_capture_plan = False
        raw = super()._run_vision_prompt(image_path, prompt, dataset)
        if capture:
            payload = self._extract_json(raw)
            if isinstance(payload, dict):
                answer = str(payload.get('final_answer', '')).strip().lower()
                if answer in ('yes', 'no'):
                    self._pope_plan_candidates.append({'answer': answer.title(),
                                                       'plan': copy.deepcopy(payload.get('plan', []))})
        return raw

    def _finalize_intermediate_records(self, extra_records, final_answer):
        direct = bool(self._pope_round2_options.get('direct_only'))
        extra_records.update(internvl_pope_round2=True,
                             evaluation_role='direct_control' if direct else 'eva',
                             internvl_pope_round2_options=copy.deepcopy(self._pope_round2_options),
                             pope_plan_candidates=copy.deepcopy(self._pope_plan_candidates))
        return super()._finalize_intermediate_records(extra_records, final_answer)
