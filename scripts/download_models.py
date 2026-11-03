#!/usr/bin/env python3
"""Download model snapshots explicitly, without adding weights to the repository."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen', 'internvl3', 'chart'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--revision')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    repo = {'qwen': 'Qwen/Qwen2.5-VL-7B-Instruct', 'internvl3': 'OpenGVLab/InternVL3-14B',
            'chart': 'PaddlePaddle/PaddleOCR-VL-1.6'}[args.model]
    revision = args.revision or ('c5630abae1d940eafe0697512a0325494b02ab42' if args.model == 'chart' else None)
    path = snapshot_download(repo_id=repo, revision=revision, local_dir=args.output_dir)
    print(path)


if __name__ == '__main__':
    main()
