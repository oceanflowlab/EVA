#!/usr/bin/env python3
"""Start exactly five local workers, with isolated OCR/chart interpreters."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eva.release import TOOLS

CHART_REVISION = 'c5630abae1d940eafe0697512a0325494b02ab42'


def commands(args):
    controller = f'http://localhost:{args.port}'
    result = [('controller', args.python, args.gpu,
               ['-m', 'tool_server.tool_workers.online_workers.controller', '--host', '127.0.0.1', '--port', str(args.port)])]
    definitions = [
        ('ocr', 'ocr_worker', args.ocr_python, args.gpu, 8,
         ['--ocr_version', 'PP-OCRv6', '--text_detection_model_name', 'PP-OCRv6_medium_det',
          '--text_recognition_model_name', 'PP-OCRv6_medium_rec']),
        ('GroundingDINO', 'grounding_dino_worker', args.python, args.gpu, 9,
         ['--model_path', args.dino_checkpoint, '--model_config', args.dino_config]),
        ('DrawLine', 'drawline_worker', args.python, '', 12, []),
        ('DrawShape', 'draw_shape_worker', args.python, '', 13, []),
        ('GetBarInfo', 'get_bar_worker', args.chart_python, args.chart_gpu, 16,
         ['--backend', 'paddleocr_vl', '--model_path', 'PaddlePaddle/PaddleOCR-VL-1.6',
          '--model_revision', CHART_REVISION, '--max_concurrency', '1', '--max_new_tokens', '2048']),
    ]
    for name, module, python, gpu, offset, extra in definitions:
        port = args.port + offset
        command = ['-m', f'tool_server.tool_workers.online_workers.{module}',
                   '--host', '127.0.0.1', '--port', str(port), '--worker_addr', f'http://localhost:{port}',
                   '--controller_addr', controller] + extra
        result.append((name, python, gpu, command))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--ocr-python', default=os.environ.get('OCR_PYTHON', sys.executable))
    parser.add_argument('--chart-python', default=os.environ.get('CHART_PYTHON', sys.executable))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--chart-gpu', default='0')
    parser.add_argument('--port', type=int, default=20001)
    parser.add_argument('--dino-checkpoint', default=os.environ.get('DINO_CHECKPOINT'))
    parser.add_argument('--dino-config', default=os.environ.get('DINO_CONFIG'))
    parser.add_argument('--log-dir', type=Path, default=ROOT / 'logs' / datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--startup-timeout', type=float, default=900)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.dino_checkpoint or not args.dino_config:
        parser.error('Supply --dino-checkpoint and --dino-config (or DINO_CHECKPOINT/DINO_CONFIG).')
    if not 1 <= args.port <= 65519 or args.startup_timeout <= 0:
        parser.error('Invalid port/startup timeout')
    args.log_dir = args.log_dir.resolve()
    args.dino_checkpoint = str(Path(args.dino_checkpoint).resolve())
    args.dino_config = str(Path(args.dino_config).resolve())
    for name in ('python', 'ocr_python', 'chart_python'):
        value = getattr(args, name)
        if os.path.sep in value:
            setattr(args, name, str(Path(value).absolute()))
    plan = commands(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for path in (args.dino_checkpoint, args.dino_config):
        if not Path(path).is_file():
            parser.error(f'DINO file does not exist: {path}')
    for offset in (0, 8, 9, 12, 13, 16):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port + offset))
    import requests
    session = requests.Session()
    session.trust_env = False
    processes = []
    args.log_dir.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    def wait_ready(url, payload):
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited with {process.returncode}; inspect {args.log_dir}')
            try:
                response = session.post(url, json=payload, timeout=(2, 5))
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                time.sleep(2)
        raise TimeoutError(f'Not ready: {url}; inspect {args.log_dir}')

    try:
        for name, python, gpu, command in plan:
            env = os.environ.copy()
            env.update(PYTHONPATH=str(ROOT) + os.pathsep + env.get('PYTHONPATH', ''),
                       CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                       EVA_TOOL_LOG_DIR=str(args.log_dir))
            for key in ('NO_PROXY', 'no_proxy'):
                env[key] = 'localhost,127.0.0.1,' + env.get(key, '')
            env.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            env.setdefault('PADDLE_PDX_MODEL_SOURCE', 'huggingface')
            with (args.log_dir / f'{name}.log').open('w') as log:
                process = subprocess.Popen([python] + command, cwd=args.log_dir, env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((name, process))
            port = command[command.index('--port') + 1]
            endpoint = '/list_models' if name == 'controller' else '/worker_get_status'
            wait_ready(f'http://localhost:{port}{endpoint}', {})
            print(f'{name}: ready (pid={process.pid}, port={port})', flush=True)
        registered = wait_ready(f'http://localhost:{args.port}/list_models', {})['models']
        if set(TOOLS) - set(registered):
            raise RuntimeError(f'Missing registrations: {set(TOOLS) - set(registered)}')
        print(f'All five tools ready. Logs: {args.log_dir}. Ctrl+C stops these processes.', flush=True)
        while True:
            time.sleep(5)
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited; inspect {args.log_dir}')
    except KeyboardInterrupt:
        pass
    finally:
        for _, process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    main()
