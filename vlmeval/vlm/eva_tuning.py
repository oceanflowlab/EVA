"""Opt-in EVA tuning profiles; the original model registration stays unchanged."""
import copy
import base64
import csv
import io
import json
import math
import re
import time
from PIL import Image

from .eva_evidence import Qwen2VLEvidenceToolAgent, normalize_drawing_args


TUNING_PROFILES = {'sparse_simple_line_coords': {'simple_line_coords': True, 'sparse_gate': 'balanced', 'sparse_replan': True, 'concise_plan': True}, 'avg_mid1152_obs12': {'simple_line_coords': True, 'sparse_gate': 'balanced', 'sparse_replan': True, 'concise_plan': True, 'direct_prompt': 'observe12', 'direct_min_pixels': 903168}}

SPARSE_GATE_INSTRUCTION = (
    '\nTool activation rule: inspect the image and exact question first. '
    'Use tool_calls=[] when direct visual evidence is sufficient. '
    'Call a tool only for a specific unresolved fact that could change the answer and that '
    'this tool can resolve. Vague uncertainty or a desire to double-check is not sufficient. '
    'Prefer one targeted call; add another only for a different necessary fact. '
    'Drawing annotations are not measurements. Do not invent coordinates or repeat templates. '
    'Keep the plan brief and name the unresolved fact.'
)

SPARSE_REPLAN_INSTRUCTION = (
    '\nReplanning stop rule: stop with tool_calls=[] once the image and evidence resolve the question, '
    'or no available tool can resolve the remaining uncertainty. '
    'Request another call only for a still-missing, answer-relevant observation. '
    'Do not repeat the same arguments; retry failures only with corrected arguments. '
    'Empty detections alone justify neither absence nor a retry. '
    'There is at most one replanning round. Finish brief, complete JSON.'
)

LITERAL_INSTRUCTION = (
    '\nEvaluate the exact claim in the question, including comparison direction, '
    'negation, counts, and spatial relationships. Use the depicted scene, not an assumed '
    'real-world fact or a remembered illusion. Do not replace the claim with an easier one. '
    'For chart comparisons check the named categories and axis labels. '
    'Missing OCR text or detector boxes is not proof that an object is absent. '
    'Drawn lines and shapes are annotations, not measurements of equality or length. '
    'Return the required answer format without introducing an automatic Yes or No bias.'
)


def adapt_line_coordinates(args, width, height):
    """Translate pixel-position arrays to the unchanged DrawLine worker contract."""
    args = dict(args)
    mode = str(args.get('line_type', '')).lower()
    if mode not in ('horizontal', 'vertical'):
        raise ValueError('invalid_line_type')
    axis, limit = ('y', height) if mode == 'horizontal' else ('x', width)
    positions = args.get('coordinates')
    if isinstance(positions, str):
        text = positions.strip()
        if '=' in text:
            legacy = normalize_drawing_args('DrawLine', args, width, height)
            positions = [float(pair.split('=', 1)[1]) for pair in legacy['coordinates'].split(',')]
        elif text.startswith('['):
            positions = json.loads(text)
        else:
            try:
                positions = [float(value.strip()) for value in next(csv.reader([text]))]
            except csv.Error as error:
                raise ValueError('invalid_coordinate_format') from error
    if not isinstance(positions, list) or not positions:
        raise ValueError('coordinates_must_be_a_nonempty_array')
    for value in positions:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or isinstance(value, float) and (not math.isfinite(value) or not value.is_integer())):
            raise ValueError('coordinates_must_be_integer_pixels')
        if not 0 <= value < limit:
            raise ValueError('coordinates_out_of_bounds')
    args['line_type'] = mode
    args['coordinates'] = ', '.join(f'{axis}{index}={int(value)}' for index, value in enumerate(positions, 1))
    return args


def compact_detections(tool_name, response, question='', limit=None):
    """Keep labels, confidence and pixel boxes; explicitly identify omitted evidence."""
    detections = response.get('detections', [])
    if not isinstance(detections, list):
        return {'tool_response_from': tool_name, 'error': 'invalid_detection_format'}
    indexed = [(index, item) for index, item in enumerate(detections) if isinstance(item, dict)]
    if limit is not None:
        terms = set(re.findall(r'\w+', question.lower()))
        terms -= {'the', 'a', 'an', 'is', 'are', 'in', 'on', 'of', 'to', 'and', 'or',
                  'does', 'do', 'it', 'this', 'that', 'there', 'image', 'picture'}
        ranked = sorted(indexed, key=lambda pair: (
            -len(terms & set(re.findall(r'\w+', str(pair[1].get('label', '')).lower()))), pair[0]))
        indexed = sorted(ranked[:limit], key=lambda pair: pair[0])
    rows = []
    for _, item in indexed:
        box = item.get('pixel_bbox') or item.get('bbox') or {}
        if isinstance(box, dict):
            coords = [box.get(key) for key in ('x_min', 'y_min', 'x_max', 'y_max')]
        else:
            coords = box
        rows.append([item.get('label', ''), item.get('confidence'), coords])
    result = {
        'tool_response_from': tool_name,
        'image_dimensions_pixels': response.get('image_dimensions_pixels', {}),
        'columns': ['label', 'confidence', 'bbox_xyxy_absolute_pixels'],
        'rows': rows,
    }
    if not rows:
        result['note'] = 'No detections returned; this is not proof of absence.'
    if len(indexed) < len(detections):
        result['omitted_detections'] = len(detections) - len(indexed)
        result['note'] = 'Partial evidence; omitted detections must not be treated as absent.'
    return result


class Qwen2VLTuningToolAgent(Qwen2VLEvidenceToolAgent):
    def __init__(self, tuning_profile='avg_mid1152_obs12', **kwargs):
        if tuning_profile not in TUNING_PROFILES:
            raise ValueError(f'Unknown tuning profile: {tuning_profile}')
        self.tuning_profile = tuning_profile
        self._tuning_options = copy.deepcopy(TUNING_PROFILES[tuning_profile])
        kwargs.update(self._tuning_options.get('parameters', {}))
        if int(kwargs.get('max_tool_rounds', 2)) != 2:
            raise ValueError('EVA tuning fixes max_tool_rounds=2: one plan and at most one replan')
        kwargs['max_tool_rounds'] = 2
        super().__init__(evidence_profile=self._tuning_options.get('evidence_profile', 'baseline'), **kwargs)

    def generate_inner(self, message, dataset=None):
        _, self._tuning_question = self._extract_image_and_question(message)
        self._raw_tool_observations = []
        self._model_call_metrics = []
        self._tool_call_metrics = []
        self._line_coordinate_adaptations = []
        self._tool_selection_records = []
        self._direct_answer_active = False
        self._direct_answer_records = []
        self._sample_start = time.perf_counter()
        return super().generate_inner(message, dataset=dataset)

    def _normalize_tool_calls(self, tool_calls):
        calls = super()._normalize_tool_calls(tool_calls)
        if not self._tuning_options.get('dedup'):
            return calls
        selected, seen = [], set()
        for call in calls:
            args = call.get('args')
            if isinstance(args, dict):
                args = {key: value for key, value in args.items() if key != 'image'}
            key = (self._normalize_tool_name(str(call.get('name', ''))),
                   json.dumps(args, sort_keys=True, ensure_ascii=False))
            if key not in seen:
                selected.append(call)
                seen.add(key)
        unique_count = len(selected)
        limit = self._tuning_options.get('round_call_limit')
        if limit is not None:
            selected = selected[:limit]
        if not hasattr(self, '_tool_selection_records'):
            self._tool_selection_records = []
        self._tool_selection_records.append({'proposed_calls': copy.deepcopy(calls),
                                             'duplicate_calls_removed': len(calls) - unique_count,
                                             'budget_calls_removed': unique_count - len(selected),
                                             'selected_calls': copy.deepcopy(selected)})
        return selected

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        prompt = super()._build_plan_prompt(question, include_final_answer, extra_context)
        if self._tuning_options.get('simple_line_coords'):
            # Replace the structured spec locally, without mutating shared tool definitions.
            lines = prompt.splitlines()
            for index, line in enumerate(lines):
                prefix = 'Tool argument specs: '
                if line.startswith(prefix):
                    specs = json.loads(line[len(prefix):])
                    if 'DrawLine' in specs:
                        specs['DrawLine']['format']['coordinates'] = (
                            'Nonempty JSON array of integer absolute pixel positions; '
                            'vertical: x positions; horizontal: y positions. No x/y labels or strings.')
                    lines[index] = prefix + json.dumps(specs, ensure_ascii=False)
            prompt = '\n'.join(lines)
        if self._tuning_options.get('concise_plan'):
            prompt += (
                '\nStart the response with { and end with }. No prose outside JSON and no code fences. '
                'Keep the plan to brief observations, not a long explanation. '
                'Include each proposed tool call once; never repeat a template to fill the response. '
                'Use concrete arguments, not placeholders. Finish all JSON arrays and objects.'
            )
        if self._tuning_options.get('router'):
            prompt += (
                '\nChoose evidence for the specific unresolved claim: OCR for visible text, '
                'GetBarInfo for a chart/table value comparison, GroundingDINO for object localization. '
                'Use drawing tools only when a visual annotation can help resolve the actual question. '
                'Do not use chart extraction on a non-chart image or use annotations as measurements. '
                'Prefer a small non-redundant set of calls. If the image already resolves the claim, '
                'use no tools. In replanning, request new evidence only for what is still unresolved.'
            )
        if self._tuning_options.get('sparse_gate'):
            prompt += SPARSE_GATE_INSTRUCTION
            if self._tuning_options['sparse_gate'] == 'strict':
                prompt += (
                    '\nUse a stricter gate: tools are warranted only when the necessary fact cannot '
                    'be established by direct inspection and the selected tool is likely to resolve it. '
                    'Do not request speculative measurements or use tools just to confirm a clear observation.'
                )
        if self._tuning_options.get('sparse_replan') and extra_context:
            prompt += SPARSE_REPLAN_INSTRUCTION
        limit = self._tuning_options.get('round_call_limit')
        if limit is not None:
            prompt += f'\nRequest at most {limit} tool calls this round, ordered by relevance to the unresolved claim.'
        if self._tuning_options.get('drawing_gate'):
            prompt += (
                '\nUse DrawLine or DrawShape only for an answer-relevant alignment, intersection, '
                'or geometric relationship with positions identifiable in this image. '
                'Do not guess coordinates or draw merely to confirm a clear observation. '
                'Annotations cannot establish physical lengths or equality.'
            )
        return prompt

    def _fallback_direct_answer(self, image_path, question, dataset=None):
        previous = getattr(self, '_direct_answer_active', False)
        # Do not change a rare answer-recovery call after the tool path.
        self._direct_answer_active = not getattr(self, '_tool_call_metrics', [])
        try:
            return super()._fallback_direct_answer(image_path, question, dataset=dataset)
        finally:
            self._direct_answer_active = previous

    def _build_direct_answer_prompt(self, question):
        mode = self._tuning_options.get('direct_prompt')
        if mode and getattr(self, '_direct_answer_active', False):
            brief_instructions = {
                'minimal': 'Answer the question from the image.',
                'claim': 'Judge the exact claim in the question using the image.',
                'inspect': 'Inspect the relevant visual detail before answering.',
            }
            if mode in brief_instructions:
                return (brief_instructions[mode] + ' Return only valid JSON with schema: '
                        '{"final_answer":"Yes or No"}. final_answer must be exactly Yes or No.'
                        '\n\nQuestion:\n' + question)
            schema = ('{"final_answer":"Yes or No"}' if mode == 'short' else
                      '{"observation":"one short image-grounded observation", "final_answer":"Yes or No"}')
            if mode == 'observe12':
                schema = ('{"observation":"one concrete visible fact, at most 12 words", '
                          '"final_answer":"Yes or No"}')
            words = self._tuning_options.get('direct_observation_words')
            if words is not None:
                description = ('one concrete visible fact' if mode == 'observe12'
                               else 'one short image-grounded observation')
                schema = json.dumps({'observation': f'{description}, at most {words} words',
                                     'final_answer': 'Yes or No'}, separators=(', ', ':'))
            grounding = ''
            if self._tuning_options.get('direct_relation'):
                grounding += (
                    'Compare the named objects or values in the order stated in the question '
                    'before answering. '
                )
            if self._tuning_options.get('direct_fact_first'):
                grounding += (
                    'Identify the question-relevant visible fact before choosing Yes or No. '
                    'Do not output the observation; output only the final answer in the required JSON. '
                )
            if mode == 'fact':
                schema = ('{"observation":"one brief question-relevant visible fact", '
                          '"final_answer":"Yes or No"}')
                grounding = (
                    'First state one visible fact needed to answer the question, then decide '
                    'whether it supports the claim. Describe the visible relation, not the answer. '
                )
            balance = self._tuning_options.get('direct_balance')
            if balance == 'uncertainty':
                grounding += (
                    'Check the depicted relationship before choosing either answer. '
                    'Do not treat uncertainty or missing tool detections as evidence that the claim is false. '
                )
            elif balance == 'symmetric':
                grounding += (
                    'Check what the image supports and what it contradicts. '
                    'Apply the same visual standard to Yes and No; do not prefer either answer by default. '
                )
            return (
                'Inspect the image and answer the exact question. '
                'Use visible details, not a remembered description of this image. '
                + grounding + 'Return only valid JSON with schema: ' + schema + '. '
                'final_answer must be exactly Yes or No.\n\nQuestion:\n' + question
            )
        prompt = super()._build_direct_answer_prompt(question)
        if self._tuning_options.get('literal'):
            prompt += LITERAL_INSTRUCTION
        if self._tuning_options.get('evidence_trust'):
            prompt += self._evidence_trust_instruction()
        return prompt

    @staticmethod
    def _evidence_trust_instruction():
        return (
            '\nTreat tool outputs as fallible observations, not ground truth. OCR may misread text; '
            'an empty detector output does not prove absence. Drawing annotations do not prove '
            'the chosen positions are correct. Failed or irrelevant tools provide no evidence. '
            'Resolve conflicts using clearly visible details in the image; do not change an answer '
            'solely because a tool was called or returned no result.'
        )

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        prompt = super()._build_evidence_synthesis_prompt(question, plan, tool_calls, tool_results)
        if self._tuning_options.get('literal'):
            prompt += LITERAL_INSTRUCTION
        if self._tuning_options.get('evidence_trust'):
            prompt += self._evidence_trust_instruction()
        return prompt

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls,
                                                 tool_results, evidence_payload):
        if self._tuning_options.get('visual_verifier'):
            payload = {'question': question, 'candidate': evidence_payload,
                       'tool_results': self._strip_verbose_fields(tool_results, drop_edited_image=True)}
            prompt = (
                'Check the candidate against the image and available tool evidence. '
                'Keep the candidate unless you can identify a concrete contradictory observation. '
                'Lack of a tool measurement alone is not a reason to reverse a visually supported answer. '
                'A detector or OCR can miss objects or text; an annotation cannot establish a measurement. '
                'Do not change an answer merely to sound more cautious. '
                'Output valid JSON: {"explanation": string, "final_answer": string}.\n'
                + json.dumps(payload, ensure_ascii=False)
            )
        else:
            prompt = super()._build_evidence_consistency_check_prompt(
                question, plan, tool_calls, tool_results, evidence_payload)
        if self._tuning_options.get('literal'):
            prompt += LITERAL_INSTRUCTION
        if self._tuning_options.get('evidence_trust'):
            prompt += self._evidence_trust_instruction()
        return prompt

    def _postprocess_tool_result(self, tool_name, tool_res):
        if isinstance(tool_res, dict):
            # Full text/detections stay in the local trace, not in the model prompt.
            raw = {key: copy.deepcopy(value) for key, value in tool_res.items() if key != 'edited_image'}
            self._raw_tool_observations.append({'tool_name': tool_name, 'response': raw})
        return super()._postprocess_tool_result(tool_name, tool_res)

    def _compact_tool_result_for_agent(self, tool_name, tool_res):
        if self._tuning_options.get('failure_hint') and isinstance(tool_res, dict) and tool_res.get('status') != 'success':
            return {'tool_response_from': tool_name, 'error': 'tool_failed'}
        compression = self._tuning_options.get('compression')
        if not compression or tool_name not in ('ocr', 'GroundingDINO'):
            return super()._compact_tool_result_for_agent(tool_name, tool_res)
        if not isinstance(tool_res, dict) or tool_res.get('status') != 'success':
            return {'tool_response_from': tool_name, 'error': 'tool_failed'}
        limit = self._tuning_options['detection_limit'] if compression == 'budget' else None
        return compact_detections(tool_name, tool_res, self._tuning_question, limit)

    def _call_tool(self, image_b64, call):
        start = time.perf_counter()
        try:
            if (self._tuning_options.get('simple_line_coords')
                    and self._normalize_tool_name(str(call.get('name', ''))) == 'DrawLine'):
                record = {'requested_args': copy.deepcopy(call.get('args')), 'tool': 'DrawLine'}
                self._line_coordinate_adaptations.append(record)
                try:
                    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as image:
                        width, height = image.size
                    record.update(width=width, height=height)
                    args = adapt_line_coordinates(call.get('args') or {}, width, height)
                    record['worker_args'] = copy.deepcopy(args)
                    args['image'] = image_b64
                    call = {'name': 'DrawLine', 'args': args}
                except (ValueError, TypeError, OSError) as error:
                    record['error'] = str(error)
                    return {'tool_response_from': 'DrawLine', 'error': 'invalid_line_coordinates'}
            result = super()._call_tool(image_b64, call)
            if self._tuning_options.get('failure_hint') and isinstance(result, dict) and result.get('status') == 'failed':
                return {'tool_response_from': result.get('tool_response_from', call.get('name')),
                        'error': 'tool_failed'}
            return result
        finally:
            self._tool_call_metrics.append({'tool': call.get('name'),
                                            'latency_seconds': time.perf_counter() - start})

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        start = time.perf_counter()
        if self._tuning_options.get('simple_line_coords') and 'Tool argument specs:' in prompt:
            paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
            if paths and paths[0]:
                with Image.open(paths[0]) as image:
                    width, height = image.size
                prompt += (
                    f'\nCurrent image: width={width}, height={height} pixels. '
                    f'DrawLine vertical coordinates are x positions: 0 <= x < {width}; '
                    f'horizontal coordinates are y positions: 0 <= y < {height}. '
                    'Use integer pixel positions, not normalized values or line endpoints. '
                    f'Format example only: {{"line_type":"vertical","coordinates":[{width // 2}]}}. '
                    'Choose positions from the actual image, not from this example.'
                )
        direct = getattr(self, '_direct_answer_active', False)
        original_min = self.min_pixels
        original_max = self.max_pixels
        effective_min = self._tuning_options.get('direct_min_pixels') if direct else None
        if effective_min is not None:
            if original_max is not None and effective_min > original_max:
                raise ValueError('Direct-answer min_pixels exceeds max_pixels')
            self.min_pixels = effective_min
        try:
            response = super()._run_vision_prompt(image_path, prompt, dataset=dataset)
            if direct and (effective_min is not None or self._tuning_options.get('direct_prompt')):
                if not hasattr(self, '_direct_answer_records'):
                    self._direct_answer_records = []
                self._direct_answer_records.append({
                    'prompt': prompt, 'raw_response': response,
                    'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels,
                    'prompt_mode': self._tuning_options.get('direct_prompt', 'original'),
                })
        finally:
            self.min_pixels, self.max_pixels = original_min, original_max
        tokenizer = getattr(getattr(self, 'processor', None), 'tokenizer', None)
        # Decoded-text re-tokenization is an estimate, not the raw generated token IDs.
        count = len(tokenizer.encode(response, add_special_tokens=False)) if tokenizer else None
        self._model_call_metrics.append({'latency_seconds': time.perf_counter() - start,
                                        'generated_tokens_estimate': count,
                                        'stage': 'direct_answer' if direct else 'agent',
                                        'min_pixels': effective_min if effective_min is not None else original_min,
                                        'max_pixels': original_max})
        return response

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records['tuning_profile'] = self.tuning_profile
        extra_records['tuning_options'] = copy.deepcopy(self._tuning_options)
        extra_records['raw_tool_observations'] = self._raw_tool_observations
        extra_records['tool_selection_records'] = getattr(self, '_tool_selection_records', [])
        if self._tuning_options.get('direct_prompt') or self._tuning_options.get('direct_min_pixels'):
            extra_records['direct_answer_records'] = getattr(self, '_direct_answer_records', [])
        if self._tuning_options.get('simple_line_coords'):
            extra_records['line_coordinate_adaptations'] = self._line_coordinate_adaptations
        extra_records['efficiency'] = {
            'model_calls': len(self._model_call_metrics),
            'tool_calls': len(self._tool_call_metrics),
            'model_call_metrics': self._model_call_metrics,
            'tool_call_metrics': self._tool_call_metrics,
            'latency_seconds': time.perf_counter() - self._sample_start,
            'token_count_method': 'decoded-text re-tokenization; excludes special tokens',
        }
        extra_records['resolved_parameters'] = {
            'generation': copy.deepcopy(self.generate_kwargs), 'max_tool_rounds': self.max_tool_rounds,
            'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels,
        }
        return super()._finalize_intermediate_records(extra_records, final_answer)
