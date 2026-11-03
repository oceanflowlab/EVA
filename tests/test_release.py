import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest

from eva.evaluate import configure_native_scoring, execution_config, reported_metrics, select_run
from eva.release import TOOLS, VARIANTS, model_registry
from vlmeval.vlm.multi_agent import Qwen2VLToolAgent, Qwen2VLToolAgentMultiStep
from vlmeval.vlm.qwen2_vl import Qwen2VLChat
from vlmeval.vlm.eva_tuning import adapt_line_coordinates
from vlmeval.vlm.eva_mmmu_followup import explicit_final_answer


@pytest.fixture
def agents(monkeypatch):
    def init(self, **kwargs):
        self.min_pixels = kwargs['min_pixels']
        self.max_pixels = kwargs['max_pixels']
        self.generate_kwargs = {k: kwargs[k] for k in (
            'max_new_tokens', 'top_p', 'top_k', 'temperature', 'repetition_penalty', 'do_sample')}
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text)))
    monkeypatch.setattr(Qwen2VLChat, '__init__', init)
    monkeypatch.delenv('EVA_ANSWER_EXTRACTOR', raising=False)
    monkeypatch.delenv('EVA_DINO_EMPTY_POLICY', raising=False)
    return {dataset: model_registry()[alias]() for dataset, (alias, _) in VARIANTS.items()}


def test_registry_is_qwen_only():
    assert len(model_registry('qwen')) == 4
    assert set(model_registry('qwen')) == {alias for alias, _ in VARIANTS.values()}
    assert len(model_registry()) == 8
    assert len(TOOLS) == 5 and 'Crop' not in TOOLS


def test_mmmu_report_is_validation_only():
    native = pd.DataFrame({'split': ['dev', 'validation'], 'Overall': [.5066667, .5144444],
                           'Science': [.4, .5]})
    report = reported_metrics('MMMU_DEV_VAL', native)
    assert list(report.columns) == ['split', 'Val Acc.']
    assert list(report['split']) == ['validation']
    assert report['Val Acc.'].iloc[0] == pytest.approx(51.44444)
    assert len(native) == 2 and 'Overall' in native


def test_mmmu_report_without_validation_does_not_substitute_dev():
    native = pd.DataFrame({'split': ['dev'], 'Overall': [.5]})
    assert reported_metrics('MMMU_DEV_VAL', native).empty


def test_other_benchmark_reports_unchanged():
    native = pd.DataFrame({'split': ['Overall'], 'acc': [88.12]})
    assert reported_metrics('POPE', native) is native


def test_selected_configurations(agents):
    for agent in agents.values():
        assert agent.tools == TOOLS
        assert agent.max_tool_rounds == 2
        assert agent._tuning_options['sparse_gate'] == 'balanced'
        assert not agent._benchmark_options.get('direct_review') if hasattr(agent, '_benchmark_options') else True
    hall = agents['HallusionBench']
    assert hall.tuning_profile == 'avg_mid1152_obs12'
    assert hall._tuning_options['direct_prompt'] == 'observe12'
    assert hall._tuning_options['direct_min_pixels'] == 903168
    assert agents['MMMU_DEV_VAL']._evidence_options['native_order']
    assert agents['MMMU_DEV_VAL']._followup_options['parse_final']
    assert agents['MMStar']._followup_options['mode'] == 'answer_first'
    assert agents['MMStar']._tuning_options['direct_min_pixels'] == 1003520
    assert agents['POPE']._benchmark_options['contrast']
    assert not agents['POPE']._followup_options.get('parse_final')


@pytest.mark.parametrize('dataset', VARIANTS)
def test_prompts_keep_five_tools_and_one_replan(agents, dataset):
    agent = agents[dataset]
    agent._benchmark_choices = ('A', 'B')
    agent._benchmark_images = []
    prompt = agent._build_plan_prompt('Which object?', extra_context='known evidence')
    assert all(name in prompt for name in TOOLS)
    assert 'Crop' not in prompt
    assert 'at most one replanning round' in prompt


def test_direct_answer_prompts(agents):
    for agent in agents.values():
        agent._direct_answer_active = True
        agent._benchmark_choices = ('A', 'B')
    mmmu = agents['MMMU_DEV_VAL']._build_direct_answer_prompt('Question')
    assert '<final_answer>' in mmmu and '96 words' in mmmu
    mmstar = agents['MMStar']._build_direct_answer_prompt('Question')
    assert 'final_answer FIRST' in mmstar and '64 words' in mmstar
    pope = agents['POPE']._build_direct_answer_prompt('Question')
    assert '{"reasoning": string, "final_answer": string}' in pope


def test_legacy_extraction_does_not_default_no(agents):
    records = {'question': 'Is there a person?'}
    answer = agents['POPE']._finalize_answer('Unresolved', 'POPE', records)
    assert answer == 'Unresolved'
    assert records['answer_extraction']['policy'] == 'legacy'


@pytest.mark.parametrize('text, expected', [
    ('<final_answer>B</final_answer>', 'B'),
    ('{"final_answer":"B","reasoning":"A is wrong"}', 'B'),
    ('The answer could be A or B', None),
    ('FINAL: A\nANSWER: B', None),
])
def test_explicit_answers(text, expected):
    assert explicit_final_answer(text, ('A', 'B')) == expected


@pytest.mark.parametrize('coordinates', [[12, 25], '[12, 25]', '12,25', 'x1=12, x2=25'])
def test_line_coordinate_adapter(coordinates):
    result = adapt_line_coordinates({'line_type': 'vertical', 'coordinates': coordinates}, 100, 80)
    assert result['coordinates'] == 'x1=12, x2=25'


@pytest.mark.parametrize('coordinates', [[100], [-1], [1.5], [True], []])
def test_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        adapt_line_coordinates({'line_type': 'vertical', 'coordinates': coordinates}, 100, 80)


def test_zero_index_trace_is_stable(tmp_path, agents):
    agent = agents['HallusionBench']
    agent.set_current_sample_context('HallusionBench', 0, str(tmp_path), 'EVA-Qwen')
    first = agent._sample_intermediate_dir()
    second = agent._sample_intermediate_dir()
    assert first == second and first.endswith('sample_0')
    Qwen2VLToolAgent._finalize_intermediate_records(agent, {}, 'Yes')
    trace = Path(first) / 'trace.json'
    assert json.loads(trace.read_text())['final_answer'] == 'Yes'
    assert not list(tmp_path.rglob('trace.md'))


@pytest.mark.parametrize('dataset', VARIANTS)
def test_unrouted_path_two_calls(tmp_path, agents, monkeypatch, dataset):
    agent = agents[dataset]
    image = tmp_path / 'image.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    answer = 'Yes' if dataset in ('POPE', 'HallusionBench') else 'B'
    responses = iter(['{"plan":[], "tool_calls":[]}', json.dumps({'final_answer': answer})])
    monkeypatch.setattr(Qwen2VLChat, 'generate_inner', lambda *a, **kw: next(responses))
    agent.set_current_sample_context(dataset, 0, str(tmp_path), 'EVA-Qwen')
    record = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                  {'type': 'text', 'value': 'Question?\nA. first\nB. second'}], dataset)
    assert record['prediction'] == answer
    efficiency = record['extra_records']['efficiency']
    assert efficiency['model_calls'] == 2 and efficiency['tool_calls'] == 0
    trace = Path(record['extra_records']['intermediate_dir']) / 'trace.json'
    assert trace.is_file()


@pytest.mark.parametrize('policy, has_marker', [('legacy', False), ('pope_success', True)])
def test_pope_empty_policy_marker(agents, monkeypatch, policy, has_marker):
    agent = agents['POPE']
    agent.save_intermediate = False
    agent._raw_tool_observations = []
    agent._tuning_question = ''
    captured = {}
    def call(name, args):
        captured.update(args)
        return {'status': 'success', 'detections': []}
    agent.tool_manager = SimpleNamespace(call_tool=call)
    monkeypatch.setenv('EVA_DINO_EMPTY_POLICY', policy)
    Qwen2VLToolAgent._call_tool(agent, 'encoded', {'name': 'GroundingDINO', 'args': {'description': 'person'}})
    assert ('_eva_dataset' in captured) == has_marker


def test_resume_guards_configuration(tmp_path):
    config = {'model_alias': 'EVA-Qwen', 'dataset': 'MMStar', 'profile': 'x'}
    first = select_run(tmp_path, config, False)
    assert select_run(tmp_path, config, True) == first
    with pytest.raises(ValueError, match='configuration/source changed'):
        select_run(tmp_path, dict(config, profile='other'), True)
    assert select_run(tmp_path, config, False) != first


def test_historical_scoring_context(monkeypatch):
    import os
    from vlmeval.utils.matching_util import can_infer_option
    monkeypatch.delenv('VERBOSE', raising=False)
    assert can_infer_option('Reasoned solution leading to option A.', {'A': 'one', 'B': 'two'}) == 'A'
    configure_native_scoring()
    assert can_infer_option('Reasoned solution leading to option A.', {'A': 'one', 'B': 'two'}) is False
    assert can_infer_option('A', {'A': 'one', 'B': 'two'}) == 'A'
    assert os.environ['PRED_FORMAT'] == 'tsv'


def test_http_manager_preserves_response(monkeypatch):
    from tool_server.tool_workers.tool_manager.base_manager import ToolManager
    payload = {'status': 'success', 'detections': [{'label': 'TEST'}]}
    class Session:
        def post(self, url, json, **kwargs):
            if url.endswith('/list_models'):
                data = {'models': TOOLS}
            elif url.endswith('/get_worker_address'):
                data = {'address': 'http://localhost:12345'}
            else:
                data = payload
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr('requests.Session', Session)
    manager = ToolManager(tools=TOOLS)
    assert not manager.session.trust_env
    assert manager.call_tool('ocr', {'image': 'x'}) is payload


def test_http_manager_rejects_missing_tools(monkeypatch):
    from tool_server.tool_workers.tool_manager.base_manager import ToolManager
    monkeypatch.setattr(ToolManager, '_post', lambda *a: {'models': []})
    with pytest.raises(RuntimeError, match='Tools missing'):
        ToolManager(tools=TOOLS)


def test_native_pope_category_expansion(tmp_path):
    from vlmeval.dataset.utils.yorn import POPE_rating
    path = tmp_path / 'predictions.tsv'
    pd.DataFrame({'index': [0, 1], 'answer': ['Yes', 'No'], 'extracted': ['Yes', 'Yes'],
                  'score': [True, False], 'category': ['random,popular', 'random']}).to_csv(path, sep='\t', index=False)
    score = POPE_rating(str(path)).iloc[0]
    assert score['acc'] == pytest.approx(200 / 3)
    assert score['precision'] == pytest.approx(200 / 3)
    assert score['recall'] == 100
    assert score['Overall'] == 80


@pytest.mark.parametrize('dataset', VARIANTS)
@pytest.mark.parametrize('second_tool', [False, True])
def test_routed_path_and_round_limit(tmp_path, agents, monkeypatch, dataset, second_tool):
    agent = agents[dataset]
    image = tmp_path / 'input.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    call = {'name': 'ocr', 'args': {}}
    first = json.dumps({'plan': ['Read the label'], 'tool_calls': [call]})
    second = json.dumps({'plan': [], 'tool_calls': [{'name': 'GetBarInfo', 'args': {}}] if second_tool else []})
    answer = 'Yes' if dataset in ('HallusionBench', 'POPE') else 'B'
    responses = iter([first, second, json.dumps({'final_answer': answer}), json.dumps({'final_answer': answer})])
    monkeypatch.setattr(Qwen2VLChat, 'generate_inner', lambda *a, **kw: next(responses))
    def tool(name, args):
        assert 'image' in args
        if name == 'ocr':
            return {'tool_response_from': name, 'status': 'success', 'detections': [
                {'label': 'TEST', 'confidence': .99, 'bbox': {'x_min': 1, 'y_min': 2, 'x_max': 50, 'y_max': 25}}]}
        return {'tool_response_from': name, 'status': 'success', 'chart_results': [{'result': 'A,20'}]}
    agent.tool_manager = SimpleNamespace(call_tool=tool)
    agent.set_current_sample_context(dataset, 1, str(tmp_path), 'EVA-Qwen')
    result = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                   {'type': 'text', 'value': 'Question?\nA. one\nB. two'}], dataset)
    assert result['prediction'] == answer
    record = result['extra_records']
    assert len(record['tool_rounds']) == 2
    assert record['efficiency']['model_calls'] == 4
    assert record['efficiency']['tool_calls'] == 1 + int(second_tool)
    assert len(record['raw_tool_observations']) == 1 + int(second_tool)


def test_edited_image_is_local_and_sent_to_model(tmp_path, agents, monkeypatch):
    agent = agents['HallusionBench']
    image = tmp_path / 'input.png'
    Image.new('RGB', (100, 80), 'white').save(image)
    edited = Image.new('RGB', (100, 80), 'red')
    encoded = BytesIO()
    edited.save(encoded, format='PNG')
    edited_b64 = base64.b64encode(encoded.getvalue()).decode()
    responses = iter([
        '{"plan":[],"tool_calls":[{"name":"DrawLine","args":{"line_type":"vertical","coordinates":[20]}}]}',
        '{"plan":[],"tool_calls":[]}', '{"final_answer":"Yes"}', '{"final_answer":"Yes"}',
    ])
    messages = []
    def generate(self, message, dataset=None):
        messages.append(message)
        return next(responses)
    monkeypatch.setattr(Qwen2VLChat, 'generate_inner', generate)
    agent.tool_manager = SimpleNamespace(call_tool=lambda *a: {
        'tool_response_from': 'DrawLine', 'status': 'success', 'edited_image': edited_b64})
    agent.set_current_sample_context('HallusionBench', 0, str(tmp_path), 'EVA-Qwen')
    result = agent.generate_inner([{'type': 'image', 'value': str(image)},
                                   {'type': 'text', 'value': 'Is the line vertical?'}], 'HallusionBench')
    saved = Path(result['extra_records']['tool_results'][0]['saved_edited_image_path'])
    assert saved.is_absolute() and saved.exists()
    assert any(item.get('value') == str(saved) for item in messages[1])
    serialized = (Path(result['extra_records']['intermediate_dir']) / 'trace.json').read_text()
    assert edited_b64 not in serialized
    assert str(saved) in serialized
