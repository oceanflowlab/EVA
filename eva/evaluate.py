"""Single-process, resumable entry point using native VLMEvalKit inference/scoring."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

from .release import TOOLS, VARIANTS, VARIANTS_BY_BACKBONE

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def source_digest():
    digest = hashlib.sha256()
    for directory in ('eva', 'vlmeval', 'tool_server'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def execution_config(args, dataset):
    backbone = args.backbone
    internvl = backbone == 'internvl3'
    variants = VARIANTS_BY_BACKBONE[backbone]
    return {
        'backbone': backbone,
        'dataset': dataset, 'model_alias': variants[dataset][0], 'profile': variants[dataset][1],
        'model_path': (os.environ.get('EVA_INTERNVL_MODEL_PATH', 'OpenGVLab/InternVL3-14B') if internvl
                       else os.environ.get('EVA_MODEL_PATH', 'Qwen/Qwen2.5-VL-7B-Instruct')),
        'model_revision': os.environ.get('EVA_INTERNVL_REVISION') if internvl else None,
        'sample_n': args.sample_n, 'sample_seed': args.sample_seed,
        'judge': 'exact_matching', 'max_tool_rounds': 2, 'tools': TOOLS,
        'scorer_compatibility': 'native_verbose_1_tsv',
        'answer_extractor': os.environ.get('EVA_ANSWER_EXTRACTOR', 'compatible' if internvl else 'legacy'),
        'dino_empty_policy': os.environ.get('EVA_DINO_EMPTY_POLICY',
                                           'pope_success' if internvl and dataset == 'POPE' else 'legacy'),
        'torch_dtype': 'bfloat16' if internvl else os.environ.get('EVA_TORCH_DTYPE', 'auto'),
        'attention': (os.environ.get('EVA_INTERNVL_TEXT_ATTN', 'sdpa') if internvl
                      else os.environ.get('EVA_ATTN_IMPLEMENTATION', 'flash_attention_2')),
        'base_max_tiles': (13 if dataset == 'POPE' else 7) if internvl else None,
        'total_max_tiles': 28 if internvl else None,
        'controller': os.environ.get('EVA_CONTROLLER_URL', 'http://localhost:20001'),
        'source_sha256': source_digest(),
    }


def select_run(work_dir, config, reuse):
    parent = work_dir / config['model_alias'] / config['dataset']
    parent.mkdir(parents=True, exist_ok=True)
    pointer = parent / 'latest.json'
    if reuse and pointer.exists():
        name = json.loads(pointer.read_text())['run']
        if Path(name).name != name:
            raise ValueError('Invalid latest run pointer')
        run_dir = parent / name
        previous = json.loads((run_dir / 'manifest.json').read_text())
        if previous['config'] != config:
            raise ValueError('Resume configuration/source changed. Use REUSE=0 and a new run.')
    else:
        run_dir = parent / datetime.now(timezone.utc).strftime('T%Y%m%d-%H%M%S-%fZ')
        run_dir.mkdir()
        packages = {}
        for name in ('torch', 'torchvision', 'transformers', 'accelerate', 'qwen-vl-utils',
                     'numpy', 'pandas', 'flash-attn'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        atomic_json(run_dir / 'manifest.json', {'config': config, 'packages': packages,
                    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'command': sys.argv, 'python': sys.version, 'status': 'created'})
        atomic_json(pointer, {'run': run_dir.name})
    return run_dir


def check_tools(controller):
    import requests
    session = requests.Session()
    session.trust_env = False
    def post(url, payload):
        response = session.post(url, json=payload, timeout=(5, 30))
        response.raise_for_status()
        return response.json()
    registered = post(controller.rstrip('/') + '/list_models', {})['models']
    missing = set(TOOLS) - set(registered)
    if missing:
        raise RuntimeError(f'Tools not registered: {sorted(missing)}')
    statuses = {}
    for tool in TOOLS:
        address = post(controller.rstrip('/') + '/get_worker_address', {'model': tool})['address']
        if not address:
            raise RuntimeError(f'No worker address for {tool}')
        statuses[tool] = post(address.rstrip('/') + '/worker_get_status', {})
    return statuses


def reported_metrics(dataset_name, score):
    if dataset_name != 'MMMU_DEV_VAL':
        return score
    validation = score['split'].astype(str).str.strip().str.lower().isin(['val', 'validation'])
    result = score.loc[validation, ['split', 'Overall']].copy()
    result.rename(columns={'Overall': 'Val Acc.'}, inplace=True)
    result['Val Acc.'] *= 100
    return result.reset_index(drop=True)


def run_one(args, dataset_name):
    config = execution_config(args, dataset_name)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data_job
    from vlmeval.smp import get_pred_file_path, load

    run_dir = select_run(args.work_dir.resolve(), config, args.reuse)
    manifest_path = run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    print(f'Run: {run_dir}', flush=True)
    try:
        dataset = build_dataset(dataset_name)
        if args.sample_n > 0 and args.sample_n < len(dataset.data):
            dataset.data = dataset.data.sample(n=args.sample_n, random_state=args.sample_seed).sort_index()
        indices = [str(index) for index in dataset.data['index']]
        if len(set(indices)) != len(indices):
            raise ValueError('Dataset contains duplicate indices')
        if manifest.get('indices') is not None and manifest['indices'] != indices:
            raise ValueError('Dataset changed since the resumed run')
        manifest['indices'] = indices
        manifest['status'] = 'running'
        atomic_json(manifest_path, manifest)
        prediction_file = Path(get_pred_file_path(str(run_dir), config['model_alias'], dataset_name, use_env_format=True))
        complete = False
        if prediction_file.exists():
            table = load(str(prediction_file))
            complete = ('prediction' in table and not table['index'].duplicated().any()
                        and set(table['index'].astype(str)) == set(indices)
                        and table['prediction'].notna().all())
        if args.mode != 'eval' and not complete:
            manifest['tool_status_before_inference'] = check_tools(config['controller'])
            atomic_json(manifest_path, manifest)
            infer_data_job(config['model_alias'], str(run_dir), config['model_alias'], dataset,
                           verbose=args.verbose, retry_failed=True)
        elif args.mode == 'eval' and not complete:
            raise ValueError('No complete predictions to score. Resume the inference first.')
        if args.mode != 'infer':
            score = dataset.evaluate(str(prediction_file), model='exact_matching', nproc=args.workers)
            report = reported_metrics(dataset_name, score)
            print(report.to_string(index=False) if not report.empty else 'No validation samples in this subset.', flush=True)
            atomic_json(run_dir / 'metrics.json', report.to_dict(orient='records'))
            manifest['status'] = 'complete'
        else:
            manifest['status'] = 'inferred'
        manifest['prediction_file'] = str(prediction_file)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(manifest_path, manifest)


def configure_native_scoring():
    # Upstream VERBOSE also controls the legacy long-answer A-article guard.
    # Freeze that historical behavior independently of console output verbosity.
    os.environ['VERBOSE'] = '1'
    os.environ['PRED_FORMAT'] = 'tsv'
    os.environ['EVAL_FORMAT'] = 'csv'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=list(VARIANTS_BY_BACKBONE), default='qwen')
    parser.add_argument('--data', nargs='+', choices=list(VARIANTS), default=['HallusionBench'])
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--sample-n', type=int, default=0, help='0 = complete benchmark')
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--mode', choices=['all', 'infer', 'eval'], default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.work_dir is None:
        args.work_dir = Path('outputs') / f'eva_{args.backbone}'
    if args.sample_n < 0 or args.workers < 1:
        parser.error('--sample-n must be >=0 and --workers must be >=1')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one process; CUDA_VISIBLE_DEVICES may contain multiple GPUs for device_map=auto.')
    configure_native_scoring()
    if args.mode == 'eval' and not args.reuse:
        parser.error('--mode eval requires --reuse')
    for dataset in args.data:
        run_one(args, dataset)


if __name__ == '__main__':
    main()
