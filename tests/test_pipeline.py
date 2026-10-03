"""The Snakemake pipeline parses and lists its rules (no data needed)."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which('snakemake') is None, reason='needs the pipeline extra')
def test_pipeline_parses():
    result = subprocess.run(['snakemake', '-s', 'pipelines/Snakefile', '--list-rules'],
                            cwd=REPO, capture_output=True, text=True, check=True)
    rules = set(result.stdout.split())
    assert {'all', 'sorghum_all', 'sorghum_train_rmsprop', 'sorghum_train_lbfgs',
            'sorghum_heritability', 'sorghum_plot_heritability', 'sorghum_gwas'} <= rules
