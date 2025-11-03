#!/usr/bin/env python3
"""
Runner for CageCTR in RecBole.

Usage examples:
  # Minimal: base model config only (you must prepare dataset dir and atomic files first)
  python run_example/run_cage_ctr.py --dataset ml1m_ctx --config_files config/cage_ctr/cage_ctr.yaml

  # With an extra dataset-specific override
  python run_example/run_cage_ctr.py --dataset criteo --config_files config/cage_ctr/cage_ctr.yaml,config/cage_ctr/criteo.yaml

Notes:
- The dataset should exist under ./dataset/<dataset_name>/ with RecBole atomic files.
- See docs:
  - [docs/source/user_guide/usage/running_new_dataset.rst](docs/source/user_guide/usage/running_new_dataset.rst)
  - [docs/source/user_guide/data/dataset_download.rst](docs/source/user_guide/data/dataset_download.rst)
  - [docs/source/developer_guide/customize_models.rst](docs/source/developer_guide/customize_models.rst)
"""

import argparse
from logging import getLogger

from recbole.quick_start import run_recbole


def main():
    parser = argparse.ArgumentParser(description="Run CageCTR with RecBole")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset folder name under ./dataset/")
    parser.add_argument(
        "--config_files",
        type=str,
        required=True,
        help="Comma-separated list of YAML config files. First one should be config/cage_ctr/cage_ctr.yaml",
    )
    args = parser.parse_args()
    config_files = [p.strip() for p in args.config_files.split(",") if p.strip()]

    # model is taken from YAML, but allow override by passing model='CageCTR'
    result = run_recbole(model="CageCTR", dataset=args.dataset, config_file_list=config_files)

    logger = getLogger()
    logger.info("CageCTR run complete.")
    logger.info(f"Best valid result: {result[0] if isinstance(result, (list, tuple)) else result}")


if __name__ == "__main__":
    main()