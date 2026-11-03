"""InternVL3 backend for the shared EVA policy; no Qwen model is instantiated."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from PIL import Image
import torch

from .eva_benchmarks import Qwen2VLBenchmarkToolAgent
from .eva_tuning import TUNING_PROFILES
from .internvl.utils import load_image
from .multi_agent import DEFAULT_EVA_TOOLS


INTERNVL_EVA_PROFILES = {'baseline': {'style': 'short'}, 'observe12': {'style': 'observe12'}}
INTERNVL_FAMILIES = ('MMMU', 'MMStar', 'HallusionBench', 'POPE')
DEFAULT_INTERNVL_PATH = 'OpenGVLab/InternVL3-14B'


def resolve_checkpoint(model_path):
    local = Path(model_path).expanduser()
    if local.is_dir():
        return str(local.resolve())
    if local.is_absolute() or str(model_path).startswith('.'):
        raise FileNotFoundError(f'InternVL checkpoint directory does not exist: {local}')
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=str(model_path), revision=os.environ.get('EVA_INTERNVL_REVISION'))


def checkpoint_info(model_path):
    root = Path(model_path)
    config_path = root / 'config.json'
    index_path = root / 'model.safetensors.index.json'
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(f'Incomplete InternVL checkpoint: {root}; set EVA_INTERNVL_MODEL_PATH')
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in shards):
        raise FileNotFoundError(f'InternVL checkpoint has missing/empty shards: {root}')
    return {'model_path': str(root.resolve()), 'num_layers': config['llm_config']['num_hidden_layers'],
            'weight_bytes': index.get('metadata', {}).get('total_size',
                          sum((root / name).stat().st_size for name in shards)),
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
            'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()}


def internvl_device_map(num_layers, num_gpus):
    """Follow the checkpoint README: reserve GPU 0 for vision and output modules."""
    if num_layers < 1 or num_gpus < 1:
        raise ValueError('Positive layer and GPU counts required')
    chunk = math.ceil(num_layers / (num_gpus - 0.5))
    allocations = [math.ceil(chunk * 0.5)] + [chunk] * (num_gpus - 1)
    mapping, layer = {}, 0
    for gpu, count in enumerate(allocations):
        for _ in range(count):
            if layer == num_layers:
                break
            mapping[f'language_model.model.layers.{layer}'] = gpu
            layer += 1
    if layer != num_layers:
        raise ValueError('Device map does not cover all language layers')
    for name in ('vision_model', 'mlp1', 'language_model.model.embed_tokens',
                 'language_model.model.norm', 'language_model.model.rotary_emb', 'language_model.lm_head'):
        mapping[name] = 0
    mapping[f'language_model.model.layers.{num_layers - 1}'] = 0
    return mapping


def per_image_tile_budget(image_count, max_tiles, total_tiles):
    if max_tiles < 1 or total_tiles < 1 or image_count < 0:
        raise ValueError('Invalid tile budget')
    if image_count > total_tiles:
        raise ValueError('Too many source images for tile budget; no images will be silently dropped')
    return min(max_tiles, total_tiles // image_count) if image_count else 0


class InternVL3Backend:
    def __init__(self, model_path, generation):
        import transformers
        from transformers import AutoModel, AutoTokenizer
        attention = os.environ.get('EVA_INTERNVL_TEXT_ATTN', 'sdpa')
        if attention not in ('sdpa', 'eager'):
            raise ValueError('EVA_INTERNVL_TEXT_ATTN must be sdpa or eager')
        model_path = resolve_checkpoint(model_path)
        info = checkpoint_info(model_path)
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError('InternVL3-14B EVA requires CUDA; select visible GPUs with GPU=0,1')
        free = sum(torch.cuda.mem_get_info(i)[0] for i in range(count))
        if free < info['weight_bytes'] + 4 * 1024 ** 3:
            raise RuntimeError('Not enough free GPU memory for BF16 weights plus inference headroom. '
                               'Use two free 24GB GPUs, e.g. GPU=0,1, or a larger GPU. '
                               'CPU offload is not enabled.')
        mapping = internvl_device_map(info['num_layers'], count)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True, device_map=mapping, use_flash_attn=False).eval()
        # The checkpoint constructor forces eager attention when FlashAttention is off.
        # Switch the text decoder afterwards to avoid quadratic attention matrices.
        self.model.language_model.set_attn_implementation(attention)
        if self.model.language_model.config._attn_implementation != attention:
            raise RuntimeError(f'InternVL text attention did not switch to {attention}')
        print(f'InternVL text attention: {attention}; visible GPUs: {count}', flush=True)
        self.generation = dict(generation)
        self.device = torch.device('cuda:0')
        self.num_image_token = self.model.num_image_token
        self.provenance = dict(info, torch_version=torch.__version__, transformers_version=transformers.__version__,
                               cuda_version=torch.version.cuda, dtype='bfloat16', use_flash_attn=False,
                               text_attention=attention,
                               device_map=mapping, gpu_names=[torch.cuda.get_device_name(i) for i in range(count)])

    @torch.inference_mode()
    def generate(self, paths, prompt, max_tiles=7, total_tiles=28, upscale=False):
        limit = per_image_tile_budget(len(paths), max_tiles, total_tiles)
        tensors, counts = [], []
        for path in paths:
            # load_image adds a thumbnail for multi-tile images; reserve its slot.
            grid_limit = limit - 1 if limit > 1 else 1
            pixels = load_image(path, max_num=grid_limit, upscale=upscale)
            if pixels.shape[0] > limit:
                raise RuntimeError('InternVL image preprocessing exceeded the tile budget')
            counts.append(pixels.shape[0])
            tensors.append(pixels)
        pixels = torch.cat(tensors).to(device=self.device, dtype=torch.bfloat16) if tensors else None
        views = ''.join(f'View {i}: <image>\n' for i in range(1, len(paths) + 1))
        response = self.model.chat(
            self.tokenizer, pixel_values=pixels,
            question=views + prompt.replace('<image>', '[image reference]'),
            num_patches_list=counts, generation_config=copy.deepcopy(self.generation), verbose=False)
        return response, counts


class InternVL3EVAToolAgent(Qwen2VLBenchmarkToolAgent):
    """Reuse EVA orchestration/persistence, composing an independent InternVL engine."""
    def __init__(self, benchmark_family='HallusionBench', benchmark_variant='baseline',
                 model_path=DEFAULT_INTERNVL_PATH, controller_url='http://localhost:20001',
                 tool_server_root=None, save_intermediate=True, max_tool_rounds=2,
                 base_max_tiles=7, total_max_tiles=28, max_new_tokens=2048,
                 do_sample=False, use_custom_prompt=False):
        if benchmark_family not in INTERNVL_FAMILIES or benchmark_variant not in INTERNVL_EVA_PROFILES:
            raise ValueError('Unknown InternVL EVA benchmark or variant')
        if max_tool_rounds != 2 or use_custom_prompt:
            raise ValueError('InternVL EVA fixes two rounds and uses dataset-native prompts')
        if base_max_tiles < 1 or total_max_tiles < base_max_tiles or max_new_tokens < 1:
            raise ValueError('Invalid InternVL tile/generation budget')
        self.benchmark_family, self.benchmark_variant = benchmark_family, benchmark_variant
        self.answer_extractor_policy = 'compatible'
        self.dino_empty_policy = 'pope_success' if benchmark_family == 'POPE' else 'legacy'
        self._benchmark_options = copy.deepcopy(INTERNVL_EVA_PROFILES[benchmark_variant])
        if benchmark_family == 'MMMU' and benchmark_variant == 'task_focus':
            self._benchmark_options['native_order'] = True
        self.tuning_profile = 'sparse_simple_line_coords'
        self._tuning_options = copy.deepcopy(TUNING_PROFILES[self.tuning_profile])
        self._tuning_options.update({k: v for k, v in self._benchmark_options.items()
                                    if k in ('compression', 'detection_limit', 'dedup', 'failure_hint', 'visual_verifier')})
        self._tuning_options['direct_prompt'] = 'internvl'
        self.evidence_profile = 'baseline'
        self._evidence_options = {'native_order': True} if self._benchmark_options.get('native_order') else {}
        self.controller_url, self.tool_server_root = controller_url, tool_server_root
        self.save_intermediate, self.max_tool_rounds = bool(save_intermediate), 2
        self.tools = list(DEFAULT_EVA_TOOLS)
        self.tool_manager, self._tool_init_error = None, None
        self._sample_context, self._current_artifacts = {}, []
        self._use_custom_prompt = False
        self.min_pixels = self.max_pixels = None  # Qwen pixel-floor settings do not apply to InternVL.
        self.base_max_tiles, self.total_max_tiles = base_max_tiles, total_max_tiles
        self.model_path = model_path
        self.generate_kwargs = {'do_sample': bool(do_sample), 'max_new_tokens': max_new_tokens,
                                'repetition_penalty': 1.0}
        self.backend = InternVL3Backend(model_path, self.generate_kwargs)

    def use_custom_prompt(self, dataset):
        return False

    def _validate_benchmark_dataset(self, dataset):
        if self.benchmark_family == 'HallusionBench':
            if dataset != 'HallusionBench':
                raise ValueError(f'HallusionBench model cannot evaluate {dataset}')
        else:
            super()._validate_benchmark_dataset(dataset)

    def _answer_rule(self):
        if self.benchmark_family == 'HallusionBench':
            return 'final_answer must be exactly Yes or No.'
        return super()._answer_rule()

    def _normalize_final_answer(self, answer, dataset):
        if self.benchmark_family != 'HallusionBench':
            return super()._normalize_final_answer(answer, dataset)
        if answer is None:
            return None
        text = str(answer).strip()
        return {'yes': 'Yes', 'true': 'Yes', 'no': 'No', 'false': 'No'}.get(text.lower().rstrip('.'), text)

    def _focus_instruction(self):
        if self.benchmark_family == 'HallusionBench':
            return (' Judge the exact depicted relationship, not a remembered illusion description. '
                    'A drawing is an annotation, not a measurement.' if self._benchmark_options.get('focus') else '')
        return super()._focus_instruction()

    def _run_vision_prompt(self, image_path, prompt, dataset=None):
        start = time.perf_counter()
        paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
        paths = [p for p in paths if p]
        sources = getattr(self, '_benchmark_images', [])
        if len(sources) > 1:
            paths.extend(p for p in sources if p not in paths)
        if len(paths) > 1:
            prompt = '\n'.join(self._evidence_view_labels(paths, None, None)) + '\n\n' + prompt
        if 'Tool argument specs:' in prompt and paths:
            with Image.open(paths[0]) as image:
                width, height = image.size
            prompt += (f'\nCurrent working image: width={width}, height={height} pixels. '
                       'DrawLine uses integer absolute pixel arrays: vertical x positions, horizontal y positions. '
                       'Coordinates must be inside the selected image bounds.')
        direct = getattr(self, '_direct_answer_active', False)
        cap = self._benchmark_options.get('direct_max_tiles', self.base_max_tiles) if direct else self.base_max_tiles
        upscale = bool(direct and self._benchmark_options.get('direct_upscale'))
        options = {'upscale': True} if upscale else {}
        response, counts = self.backend.generate(paths, prompt, max_tiles=cap,
                                                 total_tiles=self.total_max_tiles, **options)
        tokens = len(self.backend.tokenizer.encode(response, add_special_tokens=False))
        self._model_call_metrics.append({'stage': 'direct_answer' if direct else 'agent',
            'latency_seconds': time.perf_counter() - start, 'generated_tokens_estimate': tokens,
            'max_tiles_per_image': cap, 'total_max_tiles': self.total_max_tiles,
            'image_upscale_2x': upscale,
            'actual_tiles_per_image': counts, 'input_visual_tokens': sum(counts) * self.backend.num_image_token})
        if direct:
            self._direct_answer_records.append({'prompt': prompt, 'raw_response': response,
                'prompt_mode': self._benchmark_options['style'], 'max_tiles_per_image': cap,
                'image_upscale_2x': upscale,
                'actual_tiles_per_image': counts})
        return response

    def _finalize_intermediate_records(self, extra_records, final_answer):
        extra_records['backend'] = 'InternVL3-14B'
        extra_records['backend_parameters'] = {'base_max_tiles': self.base_max_tiles,
            'total_max_tiles': self.total_max_tiles, 'generation': copy.deepcopy(self.generate_kwargs),
            'provenance': copy.deepcopy(self.backend.provenance)}
        return super()._finalize_intermediate_records(extra_records, final_answer)
