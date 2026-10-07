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
    assert {'all', 'sorghum_all', 'sorghum_train_h2opt',
            'sorghum_heritability', 'sorghum_plot_heritability', 'sorghum_gwas',
            'simulation_all', 'sim_encode', 'sim_train_h2opt', 'sim_baseline',
            'sim_genotypes', 'sim_gwas', 'sim_evaluate', 'sim_roc', 'sim_plot_traits',
            'sim_plot_gwas', 'simulation_gwas_all', 'sorghum_gwas_all',
            'sorghum_genotype_pcs', 'sorghum_baseline',
            'sorghum_choose_noise', 'sim_noise_sweep', 'sim_plot_noise_sweep', 'miscanthus_all',
            'misc_images', 'misc_score_noise', 'misc_choose_noise', 'misc_train_h2opt',
            'misc_baseline', 'misc_heritability', 'misc_plot_heritability'} <= rules
