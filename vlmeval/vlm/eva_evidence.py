"""Controlled evidence-presentation variants of EVA's existing policy."""
import copy
import json
import base64
import io
import math
import re
from PIL import Image

from .multi_agent import Qwen2VLToolAgentMultiStep


EVIDENCE_PROFILES = {'baseline': {}}


def normalize_drawing_args(name, args, width, height):
    args = dict(args)
    if name == 'DrawLine':
        mode = str(args.get('line_type', '')).lower()
        if mode not in ('horizontal', 'vertical'):
            raise ValueError('invalid_coordinates')
        axis, limit = ('y', height) if mode == 'horizontal' else ('x', width)
        pairs = re.findall(r'\b([xy]\d*)\s*=\s*(-?\d+(?:\.\d+)?)', str(args.get('coordinates', '')))
        pairs = [(key, value) for key, value in pairs if key.startswith(axis)]
        if not pairs or any(not 0 <= float(value) < limit for _, value in pairs):
            raise ValueError('invalid_coordinates')
        args.update(line_type=mode, coordinates=', '.join(f'{key}={value}' for key, value in pairs))
    elif name == 'DrawShape':
        boxes = args.get('bboxes')
        if isinstance(boxes, str):
            boxes = json.loads(boxes)
        if not isinstance(boxes, list) or not boxes:
            raise ValueError('invalid_coordinates')
        for box in boxes:
            if not isinstance(box, dict) or box.get('shape') not in ('rectangle', 'ellipse', 'circle'):
                raise ValueError('invalid_coordinates')
            coords = box.get('coords')
            if (not isinstance(coords, list) or len(coords) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coords)):
                raise ValueError('invalid_coordinates')
            x1, y1, x2, y2 = coords
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError('invalid_coordinates')
        args['bboxes'] = boxes
    return args


class Qwen2VLEvidenceToolAgent(Qwen2VLToolAgentMultiStep):
    def __init__(self, evidence_profile='baseline', **kwargs):
        if evidence_profile not in EVIDENCE_PROFILES:
            raise ValueError(f'Unknown evidence profile: {evidence_profile}')
        self.evidence_profile = evidence_profile
        self._evidence_options = dict(EVIDENCE_PROFILES[evidence_profile])
        super().__init__(**kwargs)

    def generate_inner(self, message, dataset=None):
        self._seen_tool_calls = set()
        self._original_evidence_image, _ = self._extract_image_and_question(message)
        self._grounded_evidence_image = None
        return super().generate_inner(message, dataset=dataset)

    def _postprocess_tool_result(self, tool_name, tool_res):
        if tool_name == 'GetBarInfo' and self._evidence_options.get('native_order'):
            return tool_res
        return super()._postprocess_tool_result(tool_name, tool_res)

    def _compact_tool_result_for_agent(self, tool_name, tool_res):
        if self._evidence_options.get('robust') and isinstance(tool_res, dict) and tool_res.get('status') != 'success':
            return {'tool_response_from': tool_name, 'error': 'tool_failed'}
        if not self._evidence_options.get('readable') or tool_name not in {'ocr', 'GroundingDINO'}:
            return super()._compact_tool_result_for_agent(tool_name, tool_res)
        if not isinstance(tool_res, dict):
            return super()._compact_tool_result_for_agent(tool_name, tool_res)
        dimensions = tool_res.get('image_dimensions_pixels', {})
        lines = ['Bounding boxes use absolute pixels: [x_min, y_min, x_max, y_max].']
        for index, detection in enumerate(tool_res.get('detections', []), start=1):
            box = detection.get('pixel_bbox') or detection.get('bbox') or {}
            coordinates = [box.get(key) for key in ('x_min', 'y_min', 'x_max', 'y_max')]
            label = json.dumps(detection.get('label', ''), ensure_ascii=False)
            lines.append(f'{index}. label={label}; confidence={detection.get("confidence")}; bbox={coordinates}')
        if not tool_res.get('detections'):
            lines.append('No detections returned; this alone does not prove absence.')
        return {
            'tool_response_from': tool_name,
            'status': tool_res.get('status', 'failed'),
            'image_dimensions_pixels': dimensions,
            'evidence_text': '\n'.join(lines),
            **({'message': tool_res['message']} if tool_res.get('status') != 'success' and 'message' in tool_res else {}),
        }

    def _call_tool(self, image_b64, call):
        if self._evidence_options.get('robust'):
            name = self._normalize_tool_name(str(call.get('name', '')))
            if name not in self.tools:
                return {'tool_response_from': name, 'error': 'unsupported_tool'}
            try:
                with Image.open(io.BytesIO(base64.b64decode(image_b64))) as image:
                    width, height = image.size
                args = normalize_drawing_args(name, call.get('args') or {}, width, height)
                args.pop('image', None)
                key = (image_b64, name, json.dumps(args, sort_keys=True))
                if key in self._seen_tool_calls:
                    return {'tool_response_from': name, 'error': 'duplicate_call'}
                self._seen_tool_calls.add(key)
                call = {'name': name, 'args': args}
            except (ValueError, TypeError, AttributeError):
                return {'tool_response_from': name, 'error': 'invalid_coordinates'}
        artifact_start = len(self._current_artifacts)
        result = super()._call_tool(image_b64, call)
        if self._evidence_options.get('robust') and result.get('status') == 'failed':
            result = {'tool_response_from': call['name'], 'error': 'tool_failed'}
        if (self._evidence_options.get('grounded_image')
                and self._normalize_tool_name(str(call.get('name', ''))) == 'GroundingDINO'):
            for artifact in self._current_artifacts[artifact_start:]:
                if artifact.get('tool_name') == 'GroundingDINO' and artifact.get('path'):
                    self._grounded_evidence_image = artifact['path']
        return result

    def _extract_json(self, text):
        if not self._evidence_options.get('robust'):
            return super()._extract_json(text)
        candidate = str(text or '').strip()
        if candidate.startswith('```'):
            candidate = re.sub(r'^```(?:json)?\s*', '', candidate)
            candidate = re.sub(r'\s*```$', '', candidate)
        try:
            # Accept complete JSON after prose/fences, but never repair a truncated plan.
            start = candidate.find('{')
            if start < 0:
                return {}
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
            return payload
        except (ValueError, TypeError):
            return {}

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
        paths = [path for path in paths if path]
        if self._evidence_options.get('coordinate_prompt') and 'Tool argument specs:' in prompt and paths:
            with Image.open(paths[0]) as image:
                width, height = image.size
            prompt += (
                f'\nCurrent image: width={width}, height={height} pixels. '
                'Use absolute pixel coordinates measured from this image; never placeholders. '
                'DrawLine horizontal requires only y coordinates (e.g. y1=10); '
                'vertical requires only x coordinates (e.g. x1=10). '
                'DrawShape bboxes must be a JSON array, not a string: '
                '[{"shape":"rectangle","coords":[1,1,10,10]}]. '
                'All coordinates must be in bounds. Do not repeat identical calls. Finish the JSON.'
            )
        original = getattr(self, '_original_evidence_image', None)
        if self._evidence_options.get('dual_image') and original and original not in paths:
            paths.insert(0, original)
        grounded = getattr(self, '_grounded_evidence_image', None)
        if self._evidence_options.get('grounded_image') and grounded and grounded not in paths:
            paths.append(grounded)
        # Identify views only when there is genuinely more than one image.
        if len(paths) > 1:
            views = self._evidence_view_labels(paths, original, grounded)
            prompt = '\n'.join(views) + '\n\n' + prompt
        return super()._run_vision_prompt(paths, prompt, dataset=dataset)

    def _evidence_view_labels(self, paths, original, grounded):
        views = []
        for index, path in enumerate(paths, start=1):
            role = ('original image' if path == original else
                    'GroundingDINO annotation (boxes are tool predictions, not ground truth)'
                    if path == grounded else 'current edited image')
            views.append(f'Image {index}: {role}.')
        return views

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records['evidence_profile'] = self.evidence_profile
        extra_records['evidence_options'] = copy.deepcopy(self._evidence_options)
        return super()._finalize_intermediate_records(extra_records, final_answer)
