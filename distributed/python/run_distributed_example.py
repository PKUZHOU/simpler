#!/usr/bin/env python3
"""
Run a distributed (multi-card) simpler example.

Usage:
    python distributed/python/run_distributed_example.py \
        -k examples/a2a3/host_build_graph/treduce_distributed/kernels \
        -g examples/a2a3/host_build_graph/treduce_distributed/golden.py \
        -p a2a3 --nranks 8
"""

import argparse
import logging
import os
import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
simpler_root = script_dir.parent.parent

sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(simpler_root / "python"))
sys.path.insert(0, str(simpler_root / "examples" / "scripts"))


def main():
    parser = argparse.ArgumentParser(
        description="Run a distributed (multi-card) simpler example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-k", "--kernels", required=True,
                        help="Path to kernels directory with kernel_config.py")
    parser.add_argument("-g", "--golden", default=None,
                        help="Path to golden.py (data generation + verification)")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5"],
                        help="Target platform (default: a2a3)")
    parser.add_argument("--nranks", type=int, default=None,
                        help="Number of ranks (default: from kernel_config)")
    parser.add_argument("--root", type=int, default=None,
                        help="Root rank (default: from kernel_config)")
    parser.add_argument("--build-dir", default=None,
                        help="Build cache directory")
    parser.add_argument("--artifact-dir", default=None,
                        help="Artifact output directory")
    parser.add_argument("--orch-func", default=None,
                        help="Orchestration function name")
    parser.add_argument("--skip-compile", action="store_true",
                        help="Skip compilation")
    parser.add_argument("--skip-build-worker", action="store_true",
                        help="Skip building distributed_worker")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip golden verification")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    from distributed_runner import DistributedRunner

    golden_path = args.golden
    if golden_path is None:
        candidate = Path(args.kernels).resolve().parent / "golden.py"
        if candidate.exists():
            golden_path = str(candidate)
            logging.getLogger(__name__).info(f"Auto-detected golden: {golden_path}")

    runner = DistributedRunner(
        kernels_dir=args.kernels,
        golden_path=golden_path,
        platform=args.platform,
        nranks=args.nranks,
        root=args.root,
        build_dir=args.build_dir,
        artifact_dir=args.artifact_dir,
        orch_func=args.orch_func,
    )

    if not args.skip_compile:
        runner.compile()

    if golden_path:
        runner.prepare_data()

    if not args.skip_build_worker:
        runner.build_worker()

    success = runner.run()

    if success and golden_path and not args.skip_verify:
        success = runner.verify()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
