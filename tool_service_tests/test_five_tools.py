#!/usr/bin/env python3
"""Exercise online workers with inspectable inputs, responses and image artifacts."""

import argparse
import base64
import io
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
TOOLS = ['ocr', 'GroundingDINO', 'DrawLine', 'DrawShape', 'GetBarInfo']


def prepare_inputs(directory, photo):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('ocr.png', 'bar_chart.png'):
        shutil.copyfile(FIXTURES / name, directory / name)
    with Image.open(photo) as image:
        image.convert('RGB').save(directory / 'objects.png')
    return directory / 'ocr.png', directory / 'objects.png', directory / 'bar_chart.png'


def post(session, url, payload, timeout):
    response = session.post(url, json=payload, timeout=(10, timeout))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object from {url}')
    return result


def save_images(value, directory, paths):
    if isinstance(value, dict):
        saved = {}
        for key, item in value.items():
            if key == 'edited_image' and isinstance(item, str) and item:
                encoded = item.split(',', 1)[1] if item.startswith('data:') else item
                image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
                image.load()
                path = directory / f'edited_image_{len(paths)}.png'
                image.save(path)
                paths.append(path)
                saved['edited_image_path'] = str(path.resolve())
            else:
                saved[key] = save_images(item, directory, paths)
        return saved
    if isinstance(value, list):
        return [save_images(item, directory, paths) for item in value]
    return value


def validate(tool, result, source, images):
    checks = {'status_success': result.get('status') == 'success',
              'no_error_code': result.get('error_code', 0) == 0}
    if tool == 'ocr':
        labels = ' '.join(str(d.get('label', '')) for d in result.get('detections', []))
        checks['recognized_expected_text'] = all(word in labels.upper() for word in ['EVA', '12345'])
    elif tool == 'GroundingDINO':
        detections = result.get('detections', [])
        checks['nonempty_detections'] = bool(detections)
        width, height = Image.open(source).size
        checks['valid_boxes'] = bool(detections) and all(
            0 <= d.get('bbox', {}).get('x_min', -1) < d.get('bbox', {}).get('x_max', -1) <= width
            and 0 <= d.get('bbox', {}).get('y_min', -1) < d.get('bbox', {}).get('y_max', -1) <= height
            for d in detections)
    elif tool == 'GetBarInfo':
        tables = '\n'.join(str(item.get('result', '')) for item in result.get('chart_results', [])
                           if isinstance(item, dict))
        checks['nonempty_table'] = bool(tables.strip())
        # Known fixtures distinguish a successful HTTP response from useful perception.
        checks['expected_labels'] = all(label.lower() in tables.lower() for label in ['Alpha', 'Beta', 'Gamma'])
        checks['expected_values'] = all(re.search(rf'\b{value}(?:\.0+)?\b', tables) for value in [20, 40, 60])
    if tool in ['GroundingDINO', 'DrawLine', 'DrawShape']:
        checks['edited_image_decodes'] = bool(images)
    if tool in ['DrawLine', 'DrawShape'] and images:
        original = Image.open(source).convert('RGB')
        edited = Image.open(images[0]).convert('RGB')
        checks['image_changed'] = bool(ImageChops.difference(original.resize(edited.size), edited).getbbox())
        pixels = edited.get_flattened_data() if hasattr(edited, 'get_flattened_data') else edited.getdata()
        checks['red_annotation_pixels'] = sum(r > 150 and r > g * 1.4 and r > b * 1.4
                                              for r, g, b in pixels) > 30
    return {key: bool(value) for key, value in checks.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', default='http://localhost:20001')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--photo', type=Path, default=FIXTURES / 'astronaut.png')
    parser.add_argument('--description', default='person')
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not args.photo.is_file():
        parser.error(f'Test photo does not exist: {args.photo}')
    output = (args.output_dir or ROOT / 'outputs' / 'tool_service_tests' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text, photo, chart = prepare_inputs(output / 'inputs', args.photo)
    cases = [
        ('ocr', text, {}),
        ('GroundingDINO', photo, {'description': args.description}),
        ('DrawLine', chart, {'line_type': 'horizontal', 'coordinates': 'y1=350'}),
        ('DrawShape', photo, {'bboxes': [{'shape': 'rectangle', 'coords': [20, 30, 260, 230]}]}),
        ('GetBarInfo', chart, {}),
    ]
    session = requests.Session()
    session.trust_env = False
    summary = {'controller': args.controller, 'output_dir': str(output), 'tools': []}
    try:
        registered = post(session, args.controller.rstrip('/') + '/list_models', {}, args.timeout)['models']
        summary['registered_tools'] = registered
    except Exception as error:
        registered = []
        summary['controller_error'] = f'{type(error).__name__}: {error}'
    for tool, source, params in cases:
        directory = output / tool
        directory.mkdir(exist_ok=True)
        record = {'tool': tool, 'input_image': str(source.resolve()), 'params': params}
        start = time.perf_counter()
        try:
            if tool not in registered:
                raise RuntimeError(f'{tool} is not registered with the controller')
            worker = post(session, args.controller.rstrip('/') + '/get_worker_address', {'model': tool}, args.timeout)['address']
            if not worker:
                raise RuntimeError(f'No online worker address for {tool}')
            record['worker_address'] = worker
            payload = dict(params, image=base64.b64encode(source.read_bytes()).decode('ascii'))
            result = post(session, worker.rstrip('/') + '/worker_generate', payload, args.timeout)
            images = []
            record['response'] = save_images(result, directory, images)
            record['checks'] = validate(tool, result, source, images)
            record['passed'] = all(record['checks'].values())
        except Exception as error:
            record['passed'] = False
            record['error'] = f'{type(error).__name__}: {error}'
        record['latency_seconds'] = round(time.perf_counter() - start, 3)
        (directory / 'result.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
        summary['tools'].append(record)
        failed = [key for key, value in record.get('checks', {}).items() if not value]
        print(f"{tool:15} {'PASS' if record['passed'] else 'FAIL'}  {record['latency_seconds']:.3f}s  {record.get('error', ', '.join(failed))}", flush=True)
    summary['passed'] = all(record['passed'] for record in summary['tools'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    passed = sum(record['passed'] for record in summary['tools'])
    print(f"Overall: {'PASS' if summary['passed'] else 'FAIL'} ({passed}/{len(TOOLS)} tools)", flush=True)
    print(f'Results: {output}', flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
