import base64
import copy as cp
import json
import os
import re
import sys
import time
import uuid
import json_repair
from typing import Any, Dict, Iterable, List, Optional

import torch

from vlmeval.smp import listinstr

from .base import BaseModel
from .qwen2_vl import Qwen2VLChat




TOOL_NAME_MAP = {
    'ocr': 'ocr',
    'groundingdino': 'GroundingDINO',
    'crop': 'Crop',
    'drawline': 'DrawLine',
    'drawshape': 'DrawShape',
    'getbarinfo': 'GetBarInfo',
}

DEFAULT_EVA_TOOLS = ['ocr', 'GroundingDINO', 'DrawLine', 'DrawShape', 'GetBarInfo']

TOOL_ARG_SPECS = {
    'ocr': {'required': [], 'format': {}},
    'GroundingDINO': {'required': ['description'], 'format': {'description': 'string'}},
    'Crop': {
        'required': ['coordinates'],
        'format': {'coordinates': '[x_min, y_min, x_max, y_max] (absolute pixels)'},
    },
    'DrawLine': {
        'required': ['line_type', 'coordinates'],
        'format': {
            'line_type': 'horizontal|vertical',
            'coordinates': 'y1=..., y2=... OR x1=..., x2=... , absolute pixels',
        },
    },
    'DrawShape': {
        'required': ['bboxes'],
        'format': {'bboxes': '[{shape: rectangle|ellipse|circle, coords: [x1,y1,x2,y2]}, ...]'},
    },
    'GetBarInfo': {'required': [], 'format': {}},
}

TOOL_DESCRIPTIONS = {
    'ocr': 'Extract text from the image with bounding boxes.',
    'GroundingDINO': (
        'Locate objects in the image based on a text description. '
        "Note: a low groundingdino 'confidence' usually indicates the object is present but small or occluded, not that it is absent."
    ),
    'Crop': 'Crop the image using a bounding box.',
    'DrawLine': 'Draw horizontal or vertical lines on the image.',
    'DrawShape': 'Draw shapes (rectangle/ellipse/circle) on the image.',
    'GetBarInfo': 'Parse bar-chart/table structure from the image.',
}


class Qwen2VLToolAgent(Qwen2VLChat):
    """Plan -> tool call -> final judgment workflow based on Qwen2.5-VL."""

    def __init__(
        self,
        controller_url: str = 'http://localhost:20001',
        tool_server_root: Optional[str] = None,
        tools: Optional[List[str]] = None,
        save_intermediate: bool = True,
        **kwargs,
    ):
        self.controller_url = controller_url
        self.tool_server_root = tool_server_root
        self.save_intermediate = bool(save_intermediate)
        self.tools = list(DEFAULT_EVA_TOOLS) if tools is None else list(tools)
        self.tool_manager = None
        self._tool_init_error = None
        self._sample_context: Dict[str, Any] = {}
        self._current_artifacts: List[Dict[str, Any]] = []
        super().__init__(**kwargs)

    @staticmethod
    def _append_python_path(path: Optional[str]) -> bool:
        if not path:
            return False
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            return False
        if abs_path in sys.path:
            sys.path.remove(abs_path)
        sys.path.insert(0, abs_path)
        return True

    @staticmethod
    def _normalize_tool_name(name: str) -> Optional[str]:
        if not name:
            return None
        key = name.lower().replace('_', '').replace('-', '')
        return TOOL_NAME_MAP.get(key, name)

    @staticmethod
    def _tool_keeps_edited_image(tool_name: str) -> bool:
        return tool_name in {'Crop', 'DrawLine', 'DrawShape'}

    def set_current_sample_context(
        self,
        dataset_name: Optional[str] = None,
        index: Optional[Any] = None,
        work_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._sample_context = {
            'dataset_name': self._json_default(dataset_name),
            'index': self._json_default(index),
            'work_dir': self._json_default(work_dir),
            'model_name': self._json_default(model_name),
        }
        self._current_artifacts = []

    @staticmethod
    def _safe_path_part(value: Any) -> str:
        text = str(value if value is not None else 'unknown')
        text = re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')
        return text[:120] or 'unknown'

    def _sample_intermediate_dir(self) -> str:
        override = os.environ.get('TOOL_AGENT_INTERMEDIATE_DIR')
        dataset_name = self._safe_path_part(self._sample_context.get('dataset_name') or 'dataset')
        model_name = self._safe_path_part(self._sample_context.get('model_name') or 'model')
        sample_index = self._sample_context.get('index')
        index = self._safe_path_part(sample_index if sample_index is not None else uuid.uuid4().hex)

        if override:
            base_dir = os.path.abspath(override)
        else:
            work_dir = self._sample_context.get('work_dir')
            base_dir = os.path.join(
                os.path.abspath(work_dir or './outputs'),
                f'{model_name}_{dataset_name}_intermediates',
            )
        output_dir = os.path.join(base_dir, f'sample_{index}')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_json_file(self, filename: str, payload: Any) -> Optional[str]:
        try:
            output_dir = self._sample_intermediate_dir()
            output_path = os.path.join(output_dir, filename)
            tmp_path = f'{output_path}.tmp'
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_default)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
                f.write('\n')
            os.replace(tmp_path, output_path)
            return output_path
        except Exception:
            return None

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if hasattr(obj, 'item'):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()
            except Exception:
                pass
        return str(obj)

    def _relative_to_intermediate_dir(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            return os.path.relpath(path, self._sample_intermediate_dir())
        except Exception:
            return path

    def _finalize_intermediate_records(self, extra_records: Dict[str, Any], final_answer: Any) -> Dict[str, Any]:
        if not self.save_intermediate:
            return extra_records
        context = dict(self._sample_context)
        output_dir = self._sample_intermediate_dir()
        extra_records['sample_context'] = context
        extra_records['intermediate_dir'] = output_dir
        extra_records['final_answer'] = final_answer
        if self._current_artifacts:
            extra_records['artifacts'] = list(self._current_artifacts)

        extra_records['intermediate_trace_json'] = 'trace.json'
        json_path = self._save_json_file('trace.json', extra_records)
        if json_path:
            extra_records['intermediate_trace_json'] = self._relative_to_intermediate_dir(json_path)
        return extra_records

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        return json_repair.loads(text)

    @staticmethod
    def _normalize_yes_no_answer(answer: Optional[Any], dataset: Optional[str]) -> Optional[str]:
        if answer is None:
            return None
        if not dataset:
            return str(answer)
        dataset_l = str(dataset).lower()
        if not any(key in dataset_l for key in ('hallusionbench', 'pope', 'mme', 'amber', 'vsr')):
            return str(answer)

        text = str(answer).strip()
        if not text:
            return text

        low = text.lower()
        if re.match(r'^(yes|yeah|yep|true|correct)\b', low):
            return 'Yes'
        if re.match(r'^(no|nope|false|incorrect)\b', low):
            return 'No'

        if re.search(r'\byes\b', low) and not re.search(r'\bno\b', low):
            return 'Yes'
        if re.search(r'\bno\b', low) and not re.search(r'\byes\b', low):
            return 'No'
        # import pdb; pdb.set_trace()
        return 'No'

    @staticmethod
    def _extract_mcq_letter(answer: Optional[Any]) -> Optional[str]:
        if answer is None:
            return None
        text = str(answer).strip()
        if not text:
            return text

        upper_text = text.upper()
        if upper_text in {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}:
            return upper_text

        explicit_patterns = (
            r'\bOPTION\s*[:：-]?\s*([A-H])\b',
            r'\bANSWER\s*(?:IS|:|：|-)?\s*\(?\s*([A-H])\s*\)?\b',
            r'\bCHOICE\s*[:：-]?\s*([A-H])\b',
            r'\(([A-H])\)',
            r'\[([A-H])\]',
        )
        for pattern in explicit_patterns:
            match = re.search(pattern, upper_text)
            if match:
                return match.group(1)

        if len(upper_text) <= 8:
            match = re.search(r'\b([A-H])\b', upper_text)
            if match:
                return match.group(1)
        return text

    def _normalize_final_answer(self, answer: Optional[Any], dataset: Optional[str]) -> Optional[str]:
        return self._normalize_yes_no_answer(answer, dataset)

    def _finalize_answer(self, answer, dataset, extra_records):
        """Apply extraction/fallback only at emission, never during agent decisions."""
        from vlmeval.utils.answer_extraction import extract_answer, question_options
        from vlmeval.utils.matching_util import can_infer

        policy = os.environ.get('EVA_ANSWER_EXTRACTOR', getattr(self, 'answer_extractor_policy', 'legacy'))
        if policy.lower() == 'legacy':
            extra_records['answer_extraction'] = {'policy': 'legacy', 'defaulted': False}
            return self._normalize_final_answer(answer, dataset)
        name = str(dataset or '').lower()
        options = question_options(extra_records.get('question', ''))
        labels = getattr(self, '_benchmark_choices', None)
        if labels is not None:
            options = {label: options.get(label, '') for label in labels}
        if any(key in name for key in ('hallusionbench', 'pope', 'mme', 'amber', 'vsr')):
            task = 'binary'
        elif options:
            task = 'choice'
        elif 'mmmu' in name:
            task = 'open'
        else:
            return self._normalize_final_answer(answer, dataset)
        extraction = extract_answer(answer, task=task, choices=options, choice_matcher=can_infer)
        extra_records['answer_extraction'] = extraction
        return extraction['answer']


    @staticmethod
    def _extract_last_number(text: Any) -> Optional[float]:
        if text is None:
            return None
        nums = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', str(text))
        if not nums:
            return None
        try:
            return float(nums[-1].replace(',', ''))
        except Exception:
            return None

    def _sort_getbar_result_text_desc(self, result_text: str) -> str:
        if not isinstance(result_text, str):
            return result_text
        lines = [ln for ln in result_text.split('\n') if ln.strip()]
        if len(lines) <= 2:
            return result_text
        header = lines[0]
        rows = lines[1:]

        parsed_rows = []
        for row in rows:
            parts = [p.strip() for p in row.split('|')]
            if len(parts) < 2:
                parsed_rows.append((None, row))
                continue
            value = self._extract_last_number(parts[-1])
            parsed_rows.append((value, row))

        numeric_rows = [(v, r) for v, r in parsed_rows if v is not None]
        non_numeric_rows = [r for v, r in parsed_rows if v is None]
        if not numeric_rows:
            return result_text

        numeric_rows.sort(key=lambda x: x[0], reverse=True)
        sorted_lines = [header] + [r for _, r in numeric_rows] + non_numeric_rows
        return '\n'.join(sorted_lines)

    def _postprocess_tool_result(self, tool_name: str, tool_res: Dict[str, Any]) -> Dict[str, Any]: # specific for GetBarInfo to sort rows by value in descending order 
        if not isinstance(tool_res, dict):
            return {'tool_response_from': tool_name, 'status': 'failed', 'message': str(tool_res)}
        if tool_name != 'GetBarInfo':
            return tool_res

        chart_results = tool_res.get('chart_results')
        if not isinstance(chart_results, list):
            return tool_res
        for item in chart_results:
            if isinstance(item, dict) and isinstance(item.get('result'), str):
                item['result'] = self._sort_getbar_result_text_desc(item['result'])
        return tool_res

    def _strip_verbose_fields(self, data: Any, drop_edited_image: bool = True) -> Any:
        fields_to_drop = {'status', 'message', 'error_code', 'tool_reward'}
        if drop_edited_image:
            fields_to_drop.add('edited_image')
        if isinstance(data, dict):
            compact: Dict[str, Any] = {}
            for key, value in data.items():
                if key in fields_to_drop:
                    continue
                # compact[key] = self._strip_verbose_fields(value, drop_edited_image=drop_edited_image)
                compact[key] = value
            return compact
        if isinstance(data, list):
            return [self._strip_verbose_fields(item, drop_edited_image=drop_edited_image) for item in data]
        
        return data

    def _compact_tool_result_for_agent(self, tool_name: str, tool_res: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tool_res, dict):
            return {'tool_response_from': tool_name, 'status': 'failed', 'message': str(tool_res)}
        original_status = tool_res.get('status', 'failed')
        tool_res = self._strip_verbose_fields(
            tool_res,
            drop_edited_image=not self._tool_keeps_edited_image(tool_name),
        )

        if tool_name != 'GetBarInfo':
            return tool_res
        compact = {'tool_response_from': 'GetBarInfo', 'status': original_status}
        if original_status != 'success':
            return compact

        chart_results = tool_res.get('chart_results')
        result_texts = []
        if isinstance(chart_results, list):
            for item in chart_results:
                if isinstance(item, dict) and isinstance(item.get('result'), str):
                    result_texts.append(item['result'])
        compact['result'] = '\n\n'.join(result_texts).strip()
        return compact

    def _ensure_tool_manager(self):
        if self.tool_manager is not None:
            return
        if self._tool_init_error is not None:
            raise RuntimeError(self._tool_init_error)

        # Try importing with optional external roots first
        env_root = os.environ.get('TOOL_SERVER_ROOT') or os.environ.get('ADAREASONER_ROOT')
        roots_to_try = [
            self.tool_server_root,
            env_root,
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ]
        for root in roots_to_try:
            if root and os.path.isdir(os.path.join(root, 'tool_server')):
                if self._append_python_path(root):
                    break

        try:
            import importlib
            module = importlib.import_module('tool_server.tool_workers.tool_manager.base_manager')
            ToolManager = getattr(module, 'ToolManager')
        except Exception as err:
            self._tool_init_error = (
                'Failed to import ToolManager. Please ensure tool_server is available in PYTHONPATH. '
                f'Original error: {type(err).__name__}: {err}'
            )
            raise RuntimeError(self._tool_init_error)

        self.tool_manager = ToolManager(controller_url_location=self.controller_url, tools=self.tools)

    def _call_tool(self, image_b64: str, call: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = self._normalize_tool_name(str(call.get('name', '')))
        if tool_name not in self.tools:
            return {
                'tool_response_from': call.get('name', 'unknown'),
                'status': 'failed',
                'message': f'Unsupported tool: {call.get("name", "unknown")}',
            }
        args = dict(call.get('args') or {})
        args.setdefault('image', image_b64)
        dataset_name = (getattr(self, '_sample_context', {}).get('dataset_name')
                        or getattr(self, 'benchmark_family', None))
        empty_policy = os.environ.get('EVA_DINO_EMPTY_POLICY', getattr(self, 'dino_empty_policy', 'legacy'))
        pope_detection = tool_name == 'GroundingDINO' and dataset_name == 'POPE' and empty_policy == 'pope_success'
        if tool_name == 'GroundingDINO':
            args.pop('_eva_dataset', None)
            if pope_detection:
                args['_eva_dataset'] = 'POPE'
        try:
            self._ensure_tool_manager()
            res = self.tool_manager.call_tool(tool_name, args)
            res = self._postprocess_tool_result(tool_name, res)
            if (
                self.save_intermediate
                and isinstance(res, dict)
                and res.get('edited_image')
                and not self._tool_keeps_edited_image(tool_name)
            ):
                saved_image_path = self._save_base64_image(
                    res.get('edited_image'),
                    prefix=f'{tool_name.lower()}_edited_image' if tool_name else 'edited_image',
                )
                if saved_image_path:
                    self._current_artifacts.append({
                        'tool_name': tool_name,
                        'type': 'edited_image',
                        'path': saved_image_path,
                    })
                res.pop('edited_image', None)
            res = self._compact_tool_result_for_agent(tool_name, res)
            if pope_detection and isinstance(res, dict):
                detections = res.get('detections', res.get('rows'))
                if isinstance(detections, list) and not detections:
                    res.pop('note', None)
            return res
        except Exception as err:
            return {
                'tool_response_from': tool_name,
                'status': 'failed',
                'message': f'{type(err).__name__}: {err}',
            }

    @staticmethod
    def _extract_image_and_question(message: List[Dict[str, str]]) -> tuple[Optional[str], str]:
        image_path = None
        texts = []
        for item in message:
            if item.get('type') == 'image' and image_path is None:
                image_path = item.get('value')
            elif item.get('type') == 'text':
                texts.append(item.get('value', ''))
        return image_path, '\n'.join(texts).strip()

    @staticmethod
    def _image_to_b64(image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _save_base64_image(self, image_b64: str, prefix: str = 'edited_image') -> Optional[str]:
        if not image_b64:
            return None
        raw_b64 = image_b64.strip()
        if raw_b64.startswith('data:') and ',' in raw_b64:
            raw_b64 = raw_b64.split(',', 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception:
            return None

        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            suffix = '.png'
        elif image_bytes.startswith(b'\xff\xd8'):
            suffix = '.jpg'
        else:
            suffix = '.png'

        output_dir = self._sample_intermediate_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{prefix}_{uuid.uuid4().hex}{suffix}')
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return output_path

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []
        return [call for call in tool_calls if isinstance(call, dict)]

    def _build_plan_prompt(
        self,
        question: str,
        include_final_answer: bool = True,
        extra_context: Optional[str] = None,
    ) -> str:
        schema = (
            '{"plan": [string], "tool_calls": [{"name": string, "args": object}], "final_answer": string}'
            if include_final_answer
            else '{"plan": [string], "tool_calls": [{"name": string, "args": object}]}'
        )
        prompt = (
            'You are a careful visual reasoning agent. First decide whether tools are actually needed. '
            'Only propose tool calls when they are likely to improve accuracy for reading tiny text, locating an object, zooming in, or extracting structured evidence. '
            f'Only output valid JSON with schema: {schema}.\n'
            'If tools are unnecessary, set tool_calls=[].\n'
            'Do NOT include image in args; image is injected by system.\n'
            f'Available tools: {", ".join(self.tools)}\n'
            f'Tool descriptions: {json.dumps({k: TOOL_DESCRIPTIONS.get(k, "") for k in self.tools}, ensure_ascii=False)}\n'
            f'Tool argument specs: {json.dumps({k: TOOL_ARG_SPECS.get(k, {}) for k in self.tools}, ensure_ascii=False)}\n'
        )
        if extra_context:
            prompt += f'\nPrevious evidence and tool history:\n{extra_context}\n'
        prompt += f'\nQuestion:\n{question}'
        return prompt

    def _build_direct_answer_prompt(self, question: str) -> str:
        return (
            'Answer the question directly from visual evidence. '
            'Only output valid JSON with schema: {"final_answer": string}. '
            'Keep exactly the required answer format from the question.\n\n'
            f'Question:\n{question}'
        )

    def _fallback_direct_answer(self, image_path: Optional[Any], question: str, dataset=None) -> Optional[str]:
        fallback_prompt = self._build_direct_answer_prompt(question)
        fallback_raw = self._run_vision_prompt(image_path, fallback_prompt, dataset=dataset)
        fallback_payload = self._extract_json(fallback_raw)
        if isinstance(fallback_payload, dict):
            value = fallback_payload.get('final_answer')
            if value is not None:
                return value
        text = str(fallback_raw).strip()
        return text if text else None

    def _build_evidence_synthesis_prompt(
        self,
        question: str,
        plan: Iterable[str],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        tool_results_for_verify = self._strip_verbose_fields(tool_results, drop_edited_image=True)
        payload = {
            'question': question,
            'plan': plan,
            'tool_calls': tool_calls,
            'tool_results': tool_results_for_verify,
        }
        return (
            'You are a strict evidence synthesizer. Read the question, plan, and tool results carefully, then produce the best supported answer.\n'
            'Treat tool_results as evidence, but do not let noisy tool outputs override obvious direct visual evidence.\n'
            'Do not invent measurements or conclusions that are not supported by the tool outputs.\n'
            'If the tool results are inconclusive, return the most conservative answer.\n'
            'Only output valid JSON with schema: '
            '{"explanation": string, "final_answer": string}.\n\n'
            f'{json.dumps(payload, ensure_ascii=False)}'
        )

    def _build_evidence_consistency_check_prompt(
        self,
        question: str,
        plan: Iterable[str],
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        evidence_payload: Dict[str, Any],
    ) -> str:
        tool_results_for_check = self._strip_verbose_fields(tool_results, drop_edited_image=True)
        payload = {
            'question': question,
            'plan': plan,
            'tool_calls': tool_calls,
            'tool_results': tool_results_for_check,
            'evidence_payload': evidence_payload,
        }
        return (
            'You are a strict evidence-reasoning consistency checker.\n'
            'Step 1 produced tool calls and step 2 produced a candidate answer from tool evidence. Your job is to judge whether the candidate answer is fully supported by the tool results.\n'
            'Only change the answer when there is a clear contradiction or missing support. Otherwise preserve the candidate answer.\n'
            'Only output valid JSON with schema: '
            '{"explanation": string, "final_answer": string}.\n\n'
            f'{json.dumps(payload, ensure_ascii=False)}'
        )

    def _run_vision_prompt(self, image_path: Optional[Any], prompt: str, dataset=None) -> str:
        msg = []
        image_paths = []
        if isinstance(image_path, (list, tuple)):
            image_paths = [path for path in image_path if path]
        elif image_path is not None:
            image_paths = [image_path]

        for path in image_paths:
            msg.append({'type': 'image', 'value': path})
        msg.append({'type': 'text', 'value': prompt})
        return super().generate_inner(msg, dataset=dataset)

    def generate_inner(self, message, dataset=None):
        image_path, question = self._extract_image_and_question(message)
        extra_records = {
            'method': 'ToolAgent',
            'controller_url': self.controller_url,
            'question': question,
            'original_image_path': image_path,
            'tool_results': [],
        }

        plan_prompt = self._build_plan_prompt(question)
        plan_raw = self._run_vision_prompt(image_path, plan_prompt, dataset=dataset)
        
        plan_payload = self._extract_json(plan_raw)
        if not isinstance(plan_payload, dict):
            plan_payload = {}
        extra_records['plan_prompt'] = plan_prompt
        extra_records['plan_raw'] = plan_raw
        extra_records['plan_payload'] = plan_payload

        plan = plan_payload.get('plan') or []
        if isinstance(plan, str):
            plan = [plan]
        tool_calls = self._normalize_tool_calls(plan_payload.get('tool_calls'))
        direct_answer = plan_payload.get('final_answer')
        plan_payload['final_answer'] = direct_answer

        # For no-tool cases, always use a dedicated direct-answer pass instead of
        # relying on the plan-stage answer (which is optimized for planning JSON).
        if not tool_calls:
            direct_answer = self._fallback_direct_answer(image_path, question, dataset=dataset)
            if direct_answer is None:
                direct_answer = plan_payload.get('final_answer')
            direct_answer = self._finalize_answer(direct_answer, dataset, extra_records)
            plan_payload['final_answer'] = direct_answer
        if not tool_calls:
            if self.save_intermediate:
                self._finalize_intermediate_records(extra_records, direct_answer)
                return {'prediction': direct_answer, 'extra_records': extra_records}
            return direct_answer

        current_image_path = image_path
        current_image_b64 = self._image_to_b64(current_image_path) if current_image_path else ''

        for call in tool_calls:
            if isinstance(call, dict):
                tool_result = self._call_tool(current_image_b64, call)
                extra_records['tool_results'].append(tool_result)

                tool_name = self._normalize_tool_name(str(call.get('name', '')))
                if self._tool_keeps_edited_image(tool_name):
                    edited_image = tool_result.get('edited_image') if isinstance(tool_result, dict) else None
                    saved_image_path = self._save_base64_image(
                        edited_image,
                        prefix=f'{tool_name.lower()}_edited_image' if tool_name else 'edited_image',
                    )
                    if saved_image_path:
                        current_image_path = saved_image_path
                        current_image_b64 = self._image_to_b64(current_image_path)
                        if isinstance(tool_result, dict):
                            tool_result['saved_edited_image_path'] = saved_image_path
                            tool_result.pop('edited_image', None)

        verify_image_paths = [current_image_path] if current_image_path else []
        
        synthesis_prompt = self._build_evidence_synthesis_prompt(question, plan, tool_calls, extra_records['tool_results'])
        synthesis_raw = self._run_vision_prompt(verify_image_paths, synthesis_prompt, dataset=dataset)
        synthesis_payload = self._extract_json(synthesis_raw)
        if not isinstance(synthesis_payload, dict):
            synthesis_payload = {}
        extra_records['synthesis_prompt'] = synthesis_prompt
        extra_records['synthesis_raw'] = synthesis_raw
        extra_records['evidence_payload'] = synthesis_payload

        consistency_prompt = self._build_evidence_consistency_check_prompt(
            question,
            plan,
            tool_calls,
            extra_records['tool_results'],
            synthesis_payload,
        )
        consistency_raw = self._run_vision_prompt(verify_image_paths, consistency_prompt, dataset=dataset)
        consistency_payload = self._extract_json(consistency_raw)
        if not isinstance(consistency_payload, dict):
            consistency_payload = {}
        extra_records['consistency_prompt'] = consistency_prompt
        extra_records['consistency_raw'] = consistency_raw
        extra_records['consistency_payload'] = consistency_payload

        final_answer = consistency_payload.get('final_answer')
        if final_answer is None:
            final_answer = synthesis_payload.get('final_answer')
        if final_answer is None:
            final_answer = direct_answer
        if final_answer is None:
            final_answer = self._fallback_direct_answer(verify_image_paths, question, dataset=dataset)
            extra_records['final_fallback_used'] = True

        final_answer = self._finalize_answer(final_answer, dataset, extra_records)

        if self.save_intermediate:
            self._finalize_intermediate_records(extra_records, final_answer)
            return {'prediction': final_answer, 'extra_records': extra_records}
        return final_answer


class Qwen2VLToolAgentMultiStep(Qwen2VLToolAgent):
    """Evidence-conditioned tool acquisition with a configurable round limit."""

    def __init__(self, max_tool_rounds: int = 2, **kwargs):
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        super().__init__(**kwargs)

    def generate_inner(self, message, dataset=None):
        image_path, question = self._extract_image_and_question(message)
        extra_records = {
            'method': 'ToolAgentMultiStep',
            'controller_url': self.controller_url,
            'question': question,
            'original_image_path': image_path,
            'tool_rounds': [],
            'tool_results': [],
        }
        
        current_image_path = image_path
        current_image_b64 = self._image_to_b64(current_image_path) if current_image_path else ''
        all_plan = []
        all_tool_calls = []

        extra_context = None
        for round_idx in range(self.max_tool_rounds):
            plan_prompt = self._build_plan_prompt(
                question,
                include_final_answer=True,
                extra_context=extra_context,
            )
            plan_raw = self._run_vision_prompt(current_image_path, plan_prompt, dataset=dataset)
            plan_payload = self._extract_json(plan_raw)
            if isinstance(plan_payload, list):
                merged = {}
                for item in plan_payload:
                    if isinstance(item, dict):
                        merged.update(item)
                plan_payload = merged
            if not isinstance(plan_payload, dict):
                plan_payload = {}
            plan = plan_payload.get('plan') or []
            if isinstance(plan, str):
                plan = [plan]
                
            tool_calls = self._normalize_tool_calls(plan_payload.get('tool_calls'))

            round_record = {
                'round': round_idx + 1,
                'plan_prompt': plan_prompt,
                'plan_raw': plan_raw,
                'extra_context': extra_context,
                'plan_payload': plan_payload,
                'tool_results': [],
            }
            extra_records['tool_rounds'].append(round_record)

            if not tool_calls:
                break

            all_plan.extend(plan)
            all_tool_calls.extend(tool_calls)

            for call in tool_calls:
                tool_result = self._call_tool(current_image_b64, call)
                round_record['tool_results'].append(tool_result)
                extra_records['tool_results'].append(tool_result)

                tool_name = self._normalize_tool_name(str(call.get('name', '')))
                if self._tool_keeps_edited_image(tool_name):
                    edited_image = tool_result.get('edited_image') if isinstance(tool_result, dict) else None
                    saved_image_path = self._save_base64_image(
                        edited_image,
                        prefix=f'{tool_name.lower()}_edited_image' if tool_name else 'edited_image',
                    )
                    if saved_image_path:
                        current_image_path = saved_image_path
                        current_image_b64 = self._image_to_b64(current_image_path)
                        tool_result['saved_edited_image_path'] = saved_image_path
                        tool_result.pop('edited_image', None)

            extra_context = json.dumps(
                {
                    'previous_tool_calls': all_tool_calls,
                    'previous_tool_results': self._strip_verbose_fields(
                        extra_records['tool_results'],
                        drop_edited_image=True,
                    ),
                    'instruction': (
                        'Decide whether another tool call is needed. '
                        'If evidence is already enough, set tool_calls=[].'
                    ),
                },
                ensure_ascii=False,
            )

        if not extra_records['tool_results']:
            direct_answer = self._fallback_direct_answer(image_path, question, dataset=dataset)
            final_answer = self._finalize_answer(direct_answer, dataset, extra_records)
            if self.save_intermediate:
                self._finalize_intermediate_records(extra_records, final_answer)
                return {'prediction': final_answer, 'extra_records': extra_records}
            return final_answer

        verify_image_paths = [current_image_path] if current_image_path else []
        synthesis_prompt = self._build_evidence_synthesis_prompt(
            question,
            all_plan,
            all_tool_calls,
            extra_records['tool_results'],
        )
        synthesis_raw = self._run_vision_prompt(verify_image_paths, synthesis_prompt, dataset=dataset)
        synthesis_payload = self._extract_json(synthesis_raw)
        if not isinstance(synthesis_payload, dict):
            synthesis_payload = {}
        extra_records['synthesis_prompt'] = synthesis_prompt
        extra_records['synthesis_raw'] = synthesis_raw
        extra_records['evidence_payload'] = synthesis_payload

        consistency_prompt = self._build_evidence_consistency_check_prompt(
            question,
            all_plan,
            all_tool_calls,
            extra_records['tool_results'],
            synthesis_payload,
        )
        consistency_raw = self._run_vision_prompt(verify_image_paths, consistency_prompt, dataset=dataset)
        consistency_payload = self._extract_json(consistency_raw)
        if not isinstance(consistency_payload, dict):
            consistency_payload = {}
        extra_records['consistency_prompt'] = consistency_prompt
        extra_records['consistency_raw'] = consistency_raw
        extra_records['consistency_payload'] = consistency_payload

        final_answer = consistency_payload.get('final_answer') or synthesis_payload.get('final_answer')
        if final_answer is None:
            final_answer = self._fallback_direct_answer(verify_image_paths, question, dataset=dataset)

        final_answer = self._finalize_answer(final_answer, dataset, extra_records)

        if self.save_intermediate:
            self._finalize_intermediate_records(extra_records, final_answer)
            return {'prediction': final_answer, 'extra_records': extra_records}
        return final_answer

































