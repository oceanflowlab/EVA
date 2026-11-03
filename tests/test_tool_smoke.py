import base64
from io import BytesIO
import json

from PIL import Image, ImageDraw
import pytest
import requests

from tool_service_tests import test_five_tools as smoke


def test_bundled_inputs_work_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = smoke.prepare_inputs(tmp_path / 'inputs', smoke.FIXTURES / 'astronaut.png')
    assert all(path.is_file() for path in paths)
    assert paths[0].read_bytes() == (smoke.FIXTURES / 'ocr.png').read_bytes()
    assert paths[2].read_bytes() == (smoke.FIXTURES / 'bar_chart.png').read_bytes()
    assert Image.open(paths[1]).size == (512, 512)


def session_factory(failed_tool=None, registered=None, controller_error=False):
    class Session:
        def __init__(self):
            self.trust_env = True

        def post(self, url, json, **kwargs):
            assert not self.trust_env
            if controller_error:
                raise requests.ConnectionError('Controller unavailable')
            if url.endswith('/list_models'):
                result = {'models': smoke.TOOLS if registered is None else registered}
            elif url.endswith('/get_worker_address'):
                result = {'address': 'http://worker/' + json['model']}
            else:
                tool = url.split('/')[-2]
                if tool == failed_tool:
                    result = {'status': 'failed', 'error_code': 1, 'message': 'Model not ready'}
                elif tool == 'ocr':
                    result = {'status': 'success', 'detections': [{'label': 'EVA TOOL TEST TOTAL 12345'}]}
                elif tool == 'GetBarInfo':
                    result = {'status': 'success', 'chart_results': [{'result': 'Alpha 20\nBeta 40\nGamma 60'}]}
                else:
                    source = Image.open(BytesIO(base64.b64decode(json['image']))).convert('RGB')
                    ImageDraw.Draw(source).rectangle((20, 30, 260, 230), outline='red', width=5)
                    buffer = BytesIO()
                    source.save(buffer, format='PNG')
                    result = {'status': 'success', 'edited_image': base64.b64encode(buffer.getvalue()).decode()}
                    if tool == 'GroundingDINO':
                        assert json['description'] == 'person'
                        result['detections'] = [{'label': 'person', 'bbox': {
                            'x_min': 20, 'y_min': 30, 'x_max': 260, 'y_max': 230}}]
            class Response:
                def raise_for_status(self):
                    pass
                def json(self):
                    return result
            return Response()
    return Session


def test_all_five_tools_pass_with_default_fixtures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke.requests, 'Session', session_factory())
    output = tmp_path / 'result'
    assert smoke.main(['--output-dir', str(output)]) == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['passed'] and len(summary['tools']) == 5
    assert 'Overall: PASS (5/5 tools)' in capsys.readouterr().out
    for record in summary['tools']:
        assert record['passed']
        assert (output / record['tool'] / 'result.json').exists()
    assert summary['tools'][1]['response']['edited_image_path'].endswith('.png')
    assert 'edited_image' not in summary['tools'][1]['response']


@pytest.mark.parametrize('tool', smoke.TOOLS)
def test_failed_worker_is_not_reported_as_pass(tmp_path, monkeypatch, capsys, tool):
    monkeypatch.setattr(smoke.requests, 'Session', session_factory(failed_tool=tool))
    assert smoke.main(['--output-dir', str(tmp_path)]) == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert not summary['passed']
    assert sum(record['passed'] for record in summary['tools']) == 4
    assert 'Overall: FAIL (4/5 tools)' in capsys.readouterr().out


@pytest.mark.parametrize('error', [False, True])
def test_missing_controller_or_registrations_fails(tmp_path, monkeypatch, error):
    monkeypatch.setattr(smoke.requests, 'Session', session_factory(registered=[], controller_error=error))
    assert smoke.main(['--output-dir', str(tmp_path)]) == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert len(summary['tools']) == 5 and all(not row['passed'] for row in summary['tools'])


def test_wrong_recognition_content_fails():
    ocr = smoke.validate('ocr', {'status': 'success', 'detections': [{'label': 'OTHER'}]}, None, [])
    chart = smoke.validate('GetBarInfo', {'status': 'success', 'chart_results': [{'result': 'Alpha 99'}]}, None, [])
    assert not all(ocr.values())
    assert not all(chart.values())
