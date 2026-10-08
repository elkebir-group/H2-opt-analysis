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
    assert {'all', 'sorghum_all', 'sorghum_train_linear_h2opt', 'sorghum_score_heritability',
            'sorghum_plot_heritability', 'sorghum_run_gwas', 'simulation_all',
            'sim_simulate_spectra', 'sim_train_linear_h2opt', 'sim_train_hybrid_h2opt',
            'sim_train_baselines', 'sim_prepare_gwas_genotypes', 'sim_run_gwas',
            'sim_score_traits', 'sim_score_gwas', 'sim_plot_traits', 'sim_plot_gwas',
            'simulation_gwas_all', 'sorghum_gwas_all', 'sorghum_prepare_genotype_pcs',
            'sorghum_train_baselines', 'sorghum_choose_linear_h2opt_noise',
            'sim_sweep_linear_h2opt_noise', 'sim_plot_linear_h2opt_noise_sweep', 'miscanthus_all',
            'misc_prepare_images', 'misc_score_cnn_h2opt_noise', 'misc_choose_cnn_h2opt_noise',
            'misc_train_cnn_h2opt', 'misc_train_baselines', 'misc_score_heritability',
            'misc_plot_heritability', 'arabidopsis_all', 'arabi_prepare_features',
            'arabi_choose_linear_h2opt_noise', 'arabi_train_linear_h2opt',
            'arabi_train_baselines', 'arabi_score_heritability', 'arabi_plot_heritability'} <= rules
