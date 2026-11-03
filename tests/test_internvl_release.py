import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from eva.evaluate import execution_config
from eva.release import INTERNVL_VARIANTS, VARIANTS, model_registry
from vlmeval.vlm import eva_internvl as core
from vlmeval.vlm.multi_agent import Qwen2VLToolAgent
from vlmeval.vlm.qwen2_vl import Qwen2VLChat


@pytest.fixture
def agents(monkeypatch):
    class Backend:
        def __init__(self, path, generation):
            self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
            self.num_image_token = 256
            self.provenance = {'dtype': 'bfloat16', 'text_attention': 'sdpa'}
            self.responses = []
            self.calls = []

        def generate(self, paths, prompt, **kwargs):
            self.calls.append({'paths': paths, 'prompt': prompt, **kwargs})
            return self.responses.pop(0), [1] * len(paths)
    monkeypatch.setattr(core, 'InternVL3Backend', Backend)
    def reject_qwen(*args, **kwargs):
        raise AssertionError('InternVL must not instantiate a Qwen model')
    monkeypatch.setattr(Qwen2VLChat, '__init__', reject_qwen)
    monkeypatch.delenv('EVA_ANSWER_EXTRACTOR', raising=False)
    monkeypatch.delenv('EVA_DINO_EMPTY_POLICY', raising=False)
    return {dataset: model_registry('internvl3')[alias]() for dataset, (alias, _) in INTERNVL_VARIANTS.items()}


def test_internvl_registry_is_separate():
    assert set(model_registry('internvl3')) == {value[0] for value in INTERNVL_VARIANTS.values()}
    assert not set(model_registry('internvl3')) & set(model_registry('qwen'))
    with pytest.raises(ValueError):
        model_registry('other')


def test_selected_profiles_and_tile_budgets(agents):
    for dataset, agent in agents.items():
        assert agent.benchmark_variant == INTERNVL_VARIANTS[dataset][1]
        assert agent.base_max_tiles == (13 if dataset == 'POPE' else 7)
        assert agent.total_max_tiles == 28
        assert agent.max_tool_rounds == 2
        assert len(agent.tools) == 5 and 'Crop' not in agent.tools
        assert agent.generate_kwargs == {'do_sample': False, 'max_new_tokens': 2048, 'repetition_penalty': 1.0}
        assert agent.answer_extractor_policy == 'compatible'
    assert agents['MMStar']._tuning_options['sparse_gate'] == 'strict'
    assert agents['MMStar']._tuning_options['round_call_limit'] == 1
    assert agents['POPE']._tuning_options['round_call_limit'] == 1
    assert agents['HallusionBench']._hallusion_options['symmetric']


@pytest.mark.parametrize('dataset', INTERNVL_VARIANTS)
def test_unrouted_pipeline_uses_internvl_and_saves_trace(agents, tmp_path, dataset):
    agent = agents[dataset]
    image = tmp_path / 'image.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    answer = 'Yes' if dataset in ('HallusionBench', 'POPE') else 'B'
    agent.backend.responses = ['{"plan":[],"need_tools":false,"tool_calls":[]}',
                               'FINAL: B' if dataset == 'MMStar' else json.dumps({'final_answer': answer})]
    agent.set_current_sample_context(dataset, 0, str(tmp_path), INTERNVL_VARIANTS[dataset][0])
    result = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                   {'type': 'text', 'value': 'Question?\nA. first\nB. second'}], dataset)
    assert result['prediction'] == answer
    record = result['extra_records']
    assert record['answer_extraction']['policy'] == 'compatible_v1'
    assert record['efficiency']['model_calls'] == 2
    assert record['efficiency']['tool_calls'] == 0
    assert record['backend'] == 'InternVL3-14B'
    assert all(call['max_tiles'] == agent.base_max_tiles for call in agent.backend.calls)
    assert (Path(record['intermediate_dir']) / 'trace.json').is_file()
    if dataset == 'MMStar':
        assert 'Output only FINAL:' in agent.backend.calls[1]['prompt']
    if dataset == 'HallusionBench':
        assert all('same evidential standard' in call['prompt'] for call in agent.backend.calls)


def test_pope_gate_suppresses_calls_and_uses_native_direct_prompt(agents, tmp_path):
    agent = agents['POPE']
    image = tmp_path / 'image.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    agent.backend.responses = [json.dumps({'need_tools': False, 'tool_calls': [
        {'name': 'GroundingDINO', 'args': {'description': 'person'}}]}), 'Yes']
    agent.set_current_sample_context('POPE', 0, str(tmp_path), 'EVA-InternVL3-POPE')
    result = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                   {'type': 'text', 'value': 'Is there a person?'}], 'POPE')
    assert result['extra_records']['pope_tool_need_gates'][0]['suppressed_calls'] == 1
    assert result['extra_records']['efficiency']['tool_calls'] == 0
    assert agent.backend.calls[1]['prompt'] == '[Source Image 1]\nIs there a person?'


@pytest.mark.parametrize('dataset', INTERNVL_VARIANTS)
def test_routed_pipeline_keeps_two_rounds_and_evidence_verification(agents, tmp_path, dataset):
    agent = agents[dataset]
    image = tmp_path / 'image.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    answer = 'Yes' if dataset in ('HallusionBench', 'POPE') else 'B'
    agent.backend.responses = [
        '{"plan":["Locate person"],"need_tools":true,"tool_calls":[{"name":"GroundingDINO","args":{"description":"person"}}]}',
        '{"plan":[],"need_tools":false,"tool_calls":[]}',
        json.dumps({'final_answer': answer}), json.dumps({'final_answer': answer}),
    ]
    captured = {}
    def call(name, args):
        captured.update(args)
        return {'tool_response_from': name, 'status': 'success', 'detections': [
            {'label': 'person', 'confidence': .9, 'bbox': {'x_min': 1, 'y_min': 2, 'x_max': 80, 'y_max': 70}}]}
    agent.tool_manager = SimpleNamespace(call_tool=call)
    agent.set_current_sample_context(dataset, 1, str(tmp_path), INTERNVL_VARIANTS[dataset][0])
    result = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                   {'type': 'text', 'value': 'Question?\nA. one\nB. two'}], dataset)
    assert result['prediction'] == answer
    record = result['extra_records']
    assert record['efficiency']['model_calls'] == 4
    assert record['efficiency']['tool_calls'] == 1
    assert len(record['tool_rounds']) == 2
    assert 'synthesis_raw' in record and 'consistency_raw' in record
    assert ('_eva_dataset' in captured) == (dataset == 'POPE')


def test_compatible_defaults_do_not_leak_between_models(agents):
    extra = {'question': 'What is the value?'}
    mmmu = agents['MMMU_DEV_VAL']
    mmmu._benchmark_choices = ()
    assert mmmu._finalize_answer('Unresolved calculation', 'MMMU_DEV_VAL', extra) != 'A'
    extra = {'question': 'Is there a person?'}
    assert agents['POPE']._finalize_answer('Unresolved', 'POPE', extra) == 'No'
    assert extra['answer_extraction']['defaulted']
    assert agents['HallusionBench'].dino_empty_policy == 'legacy'
    assert agents['POPE'].dino_empty_policy == 'pope_success'


def test_mmmu_preserves_multiple_source_images(agents, tmp_path):
    agent = agents['MMMU_DEV_VAL']
    images = [tmp_path / 'first.png', tmp_path / 'second.png']
    for image in images:
        Image.new('RGB', (100, 80), 'white').save(image)
    agent.backend.responses = ['{"tool_calls":[]}', '{"final_answer":"B"}']
    agent.set_current_sample_context('MMMU_DEV_VAL', 0, str(tmp_path), 'EVA-InternVL3-MMMU')
    agent.generate_inner([{'type': 'image', 'value': str(image)} for image in images] + [
        {'type': 'text', 'value': 'Compare images.\nA. first\nB. second'}], 'MMMU_DEV_VAL')
    assert all(call['paths'] == [str(image) for image in images] for call in agent.backend.calls)


@pytest.mark.parametrize('gpus', [1, 2, 3, 4])
def test_device_map_covers_all_layers(gpus):
    mapping = core.internvl_device_map(48, gpus)
    assert all(f'language_model.model.layers.{i}' in mapping for i in range(48))
    assert all(0 <= device < gpus for device in mapping.values())
    assert mapping['vision_model'] == mapping['language_model.model.layers.47'] == 0


@pytest.mark.parametrize('count, cap', [(1, 7), (2, 7), (4, 7), (1, 13)])
def test_image_preprocessing_respects_tile_budget(tmp_path, count, cap):
    image = tmp_path / 'wide.png'
    Image.new('RGB', (1200, 450), 'white').save(image)
    limit = core.per_image_tile_budget(count, cap, 28)
    pixels = core.load_image(str(image), max_num=limit-1 if limit > 1 else 1)
    assert 1 <= pixels.shape[0] <= limit
    assert pixels.shape[1:] == (3, 448, 448)
    assert count * pixels.shape[0] <= 28


def test_checkpoint_resolver_and_missing_shards(tmp_path, monkeypatch):
    assert core.resolve_checkpoint(str(tmp_path)) == str(tmp_path.resolve())
    with pytest.raises(FileNotFoundError):
        core.resolve_checkpoint(str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError):
        core.checkpoint_info(str(tmp_path))
    (tmp_path / 'config.json').write_text('{"llm_config":{"num_hidden_layers":48}}')
    (tmp_path / 'model.safetensors.index.json').write_text('{"weight_map":{"weight":"part.safetensors"}}')
    with pytest.raises(FileNotFoundError):
        core.checkpoint_info(str(tmp_path))
    (tmp_path / 'part.safetensors').write_bytes(b'test')
    assert core.checkpoint_info(str(tmp_path))['num_layers'] == 48
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda **kwargs: str(tmp_path))
    assert core.resolve_checkpoint('OpenGVLab/InternVL3-14B') == str(tmp_path)


@pytest.mark.parametrize('backbone', ['qwen', 'internvl3'])
@pytest.mark.parametrize('dataset', INTERNVL_VARIANTS)
def test_runtime_manifest_uses_backbone_specific_defaults(monkeypatch, backbone, dataset):
    for name in ('EVA_ANSWER_EXTRACTOR', 'EVA_DINO_EMPTY_POLICY'):
        monkeypatch.delenv(name, raising=False)
    config = execution_config(SimpleNamespace(backbone=backbone, sample_n=0, sample_seed=42), dataset)
    internvl = backbone == 'internvl3'
    assert config['answer_extractor'] == ('compatible' if internvl else 'legacy')
    assert config['dino_empty_policy'] == ('pope_success' if internvl and dataset == 'POPE' else 'legacy')
    assert config['base_max_tiles'] == ((13 if dataset == 'POPE' else 7) if internvl else None)


@pytest.mark.parametrize('backbone', ['qwen', 'internvl3'])
def test_shell_launcher_dry_run_from_another_directory(tmp_path, backbone):
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if key not in ('EVA_ANSWER_EXTRACTOR', 'EVA_DINO_EMPTY_POLICY', 'WORK_DIR')}
    env.update(BACKBONE=backbone, GPU='2,3', DATA='HallusionBench MMMU_DEV_VAL MMStar POPE',
               DRY_RUN='1', PATH=str(Path(sys.executable).parent) + os.pathsep + env['PATH'])
    result = subprocess.run(['bash', str(root / 'scripts/run_eva.sh')], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    remaining = result.stdout.strip()
    records = []
    while remaining:
        record, end = json.JSONDecoder().raw_decode(remaining)
        records.append(record)
        remaining = remaining[end:].lstrip()
    assert len(records) == 4 and all(record['backbone'] == backbone for record in records)
    assert all(record['answer_extractor'] == ('compatible' if backbone == 'internvl3' else 'legacy')
               for record in records)
