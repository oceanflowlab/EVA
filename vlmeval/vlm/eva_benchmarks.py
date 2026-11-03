"""Opt-in EVA variants for choice reasoning and object-existence benchmarks."""
import copy
import os
import re

from PIL import Image

from .eva_tuning import Qwen2VLTuningToolAgent


BENCHMARK_PROFILES = {'MMMU': {'task_focus': {'style': 'observe24', 'focus': True, 'native_order': True}}, 'MMStar': {'compact': {'style': 'observe12', 'direct_min_pixels': 802816, 'compression': 'budget', 'detection_limit': 24, 'dedup': True, 'failure_hint': True}}, 'POPE': {'compact': {'style': 'observe12', 'direct_min_pixels': 802816, 'compression': 'budget', 'detection_limit': 24, 'dedup': True, 'failure_hint': True}}}


def option_letters(question):
    if '\nOptions:\n' in question:
        question = question.rsplit('\nOptions:\n', 1)[1]
    return tuple(dict.fromkeys(re.findall(r'^\s*([A-Z])[.)]\s+', question, re.MULTILINE)))


def normalize_choice(answer, choices):
    if answer is None:
        return None
    text = str(answer).strip()
    upper = text.upper()
    if upper in choices:
        return upper
    match = re.fullmatch(
        r'(?:(?:THE\s+)?(?:FINAL\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:=\-]?\s*)?'
        r'[\[(]?([A-Z])[\])]?\s*[.]?', upper)
    if match and match.group(1) in choices:
        return match.group(1)
    # Never turn an ambiguous explanation or an invalid option into a guessed letter.
    return text


class Qwen2VLBenchmarkToolAgent(Qwen2VLTuningToolAgent):
    def __init__(self, benchmark_family='MMMU', benchmark_variant='baseline', **kwargs):
        if benchmark_family not in BENCHMARK_PROFILES:
            raise ValueError(f'Unknown benchmark family: {benchmark_family}')
        if benchmark_variant not in BENCHMARK_PROFILES[benchmark_family]:
            raise ValueError(f'Unknown benchmark variant: {benchmark_variant}')
        self.benchmark_family = benchmark_family
        self.benchmark_variant = benchmark_variant
        self._benchmark_options = copy.deepcopy(BENCHMARK_PROFILES[benchmark_family][benchmark_variant])
        super().__init__(tuning_profile='sparse_simple_line_coords', **kwargs)
        self._tuning_options.update({key: value for key, value in self._benchmark_options.items()
                                    if key in ('direct_min_pixels', 'compression', 'detection_limit',
                                               'dedup', 'failure_hint', 'visual_verifier')})
        # This marker enables direct prompt/raw-response tracing, not HallusionBench prompts.
        self._tuning_options['direct_prompt'] = 'benchmark'
        if self._benchmark_options.get('native_order'):
            self._evidence_options['native_order'] = True

    def _validate_benchmark_dataset(self, dataset):
        expected = {'MMMU': 'mmmu', 'MMStar': 'mmstar', 'POPE': 'pope'}[self.benchmark_family]
        if not dataset or expected not in dataset.lower():
            raise ValueError(f'{self.benchmark_family} model cannot evaluate {dataset}')

    def generate_inner(self, message, dataset=None):
        self._validate_benchmark_dataset(dataset)
        self._benchmark_images = []
        normalized = []
        for item in message:
            if item.get('type') == 'image':
                path = os.path.abspath(item['value'])
                if path not in self._benchmark_images:
                    self._benchmark_images.append(path)
                index = self._benchmark_images.index(path) + 1
                normalized.append({'type': 'text', 'value': f'[Source Image {index}]'})
                normalized.append({'type': 'image', 'value': path})
            elif item.get('type') == 'text':
                normalized.append(dict(item))
            else:
                raise ValueError('EVA benchmark variants accept only text and image inputs')
        _, question = self._extract_image_and_question(normalized)
        self._benchmark_choices = option_letters(question)
        if self.benchmark_family == 'MMStar' and not self._benchmark_choices:
            raise ValueError('MMStar input is missing option labels')
        self._benchmark_tool_targets = []
        self._benchmark_dimensions = []
        for path in self._benchmark_images:
            with Image.open(path) as image:
                self._benchmark_dimensions.append(image.size)
        return super().generate_inner(normalized, dataset)

    def _answer_rule(self):
        if self.benchmark_family == 'POPE':
            return 'final_answer must be exactly Yes or No.'
        choices = getattr(self, '_benchmark_choices', ())
        if choices:
            return 'final_answer must be exactly one of these option letters: ' + ', '.join(choices) + '.'
        return 'No options are supplied: final_answer must be the concise answer, including units if needed.'

    def _focus_instruction(self):
        if not self._benchmark_options.get('focus'):
            return ''
        return {
            'MMMU': (' Read the relevant diagram labels, table entries and units. '
                     'Distinguish given values from quantities to be inferred; compare the actual options.'),
            'MMStar': (' Check the named objects, counts, attributes and comparison direction. '
                       'Use visible differences to distinguish the options, not scene stereotypes.'),
            'POPE': (' Judge only whether the requested object is visibly present. '
                     'Do not substitute an associated object or a likely scene companion. '
                     'A missed detection is not proof of absence, and a spurious detection is not proof of presence.'),
        }[self.benchmark_family]

    def _build_plan_prompt(self, question, include_final_answer=True, extra_context=None):
        prompt = super()._build_plan_prompt(question, include_final_answer, extra_context)
        prompt += '\nAnswer format: ' + self._answer_rule() + self._focus_instruction()
        if self.benchmark_family == 'POPE' and self._benchmark_options.get('focus'):
            prompt += ('\nIf GroundingDINO is necessary, describe the requested object rather than '
                       'sending the whole question as its description. Do not force a tool call.')
        if len(getattr(self, '_benchmark_images', [])) > 1:
            prompt += (
                '\nAll source images are available. Source Image IDs in the question are stable. '
                'To use a tool on a particular original source, put image_index (1-based integer) '
                'inside args; the system injects that image. Without image_index, tools use the '
                'current working image. An explicit image_index always selects the original source, '
                'not a previous annotation. Coordinates must match that source image.'
            )
            for index, (width, height) in enumerate(self._benchmark_dimensions, 1):
                prompt += f'\nSource Image {index}: width={width}, height={height} pixels.'
        return prompt

    def _build_direct_answer_prompt(self, question):
        style = (self._benchmark_options['style'] if getattr(self, '_direct_answer_active', False)
                 else 'short')
        schema = '{"final_answer": string}'
        if style != 'short':
            words = {'observe8': 8, 'observe12': 12, 'observe24': 24}[style]
            schema = ('{"observation":"one question-relevant visible fact, at most '
                      + str(words) + ' words", "final_answer": string}')
        claim = ''
        if self._benchmark_options.get('claim_check'):
            claim = (
                ' Check the identity of the requested object, not an associated object. '
                'Choose the answer from the visible scene, not from what usually occurs in this scene.'
                if self.benchmark_family == 'POPE' else
                ' Compare the supplied options with the visible evidence. '
                'Check negation, units and comparison direction before selecting the option letter.'
            )
        return ('Inspect the image and answer the exact question. '
                'Use visible evidence and the supplied question, not a remembered image description. '
                'Return only valid JSON with schema: ' + schema + '. '
                + self._answer_rule() + self._focus_instruction() + claim + '\n\nQuestion:\n' + question)

    def _build_evidence_synthesis_prompt(self, question, plan, tool_calls, tool_results):
        return (super()._build_evidence_synthesis_prompt(question, plan, tool_calls, tool_results)
                + '\n' + self._answer_rule() + self._focus_instruction())

    def _build_evidence_consistency_check_prompt(self, question, plan, tool_calls, tool_results, evidence_payload):
        return (super()._build_evidence_consistency_check_prompt(
            question, plan, tool_calls, tool_results, evidence_payload) + '\n' + self._answer_rule())

    def _normalize_final_answer(self, answer, dataset):
        if self.benchmark_family != 'POPE':
            choices = getattr(self, '_benchmark_choices', ())
            return normalize_choice(answer, choices) if choices else (str(answer).strip() if answer is not None else None)
        if answer is None:
            return None
        text = str(answer).strip()
        if text.lower().rstrip('.') in ('yes', 'true'):
            return 'Yes'
        if text.lower().rstrip('.') in ('no', 'false'):
            return 'No'
        return text

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
        paths = [path for path in paths if path]
        sources = getattr(self, '_benchmark_images', [])
        for source in sources if len(sources) > 1 else []:
            if source not in paths:
                paths.append(source)
        return super()._run_vision_prompt(paths, prompt, dataset)

    def _evidence_view_labels(self, paths, original, grounded):
        sources = getattr(self, '_benchmark_images', [])
        return [f'View {i}: ' + (f'original Source Image {sources.index(path) + 1}.' if path in sources
                                else 'current tool-edited working image; annotations are not measurements.')
                for i, path in enumerate(paths, 1)]

    def _call_tool(self, image_b64, call):
        args = call.get('args')
        if isinstance(args, dict) and 'image_index' in args:
            call = copy.deepcopy(call)
            index = call['args'].pop('image_index')
            images = getattr(self, '_benchmark_images', [])
            record = {'image_index': index, 'tool': call.get('name')}
            self._benchmark_tool_targets.append(record)
            if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(images):
                record['error'] = 'invalid_image_index'
                self._tool_call_metrics.append({'tool': call.get('name'), 'latency_seconds': 0.0})
                return {'tool_response_from': call.get('name'), 'error': 'invalid_image_index'}
            record['source_image_path'] = images[index - 1]
            image_b64 = self._image_to_b64(images[index - 1])
        return super()._call_tool(image_b64, call)

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records.update(benchmark_family=self.benchmark_family, benchmark_variant=self.benchmark_variant,
                             benchmark_options=copy.deepcopy(self._benchmark_options),
                             benchmark_input_images=list(self._benchmark_images),
                             benchmark_option_letters=list(self._benchmark_choices),
                             benchmark_tool_targets=self._benchmark_tool_targets)
        return super()._finalize_intermediate_records(extra_records, final_answer)
