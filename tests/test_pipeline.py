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
            'sim_simulate_spectra', 'sim_choose_linear_h2opt_noise', 'sim_train_linear_h2opt',
            'sim_choose_hybrid_cnn_level', 'sim_train_hybrid_h2opt',
            'sim_train_baselines', 'sim_prepare_gwas_genotypes', 'sim_run_gwas',
            'sim_score_traits', 'sim_score_gwas', 'sim_plot_traits', 'sim_plot_gwas',
            'sim_plot_paper_figures', 'sim_plot_paper_null', 'sim_plot_paper_linear',
            'simulation_paper_figures',
            'simulation_gwas_all', 'sorghum_gwas_all', 'sorghum_prepare_genotype_pcs',
            'sorghum_train_baselines', 'sorghum_choose_linear_h2opt_noise',
            'sim_sweep_linear_h2opt_noise', 'sim_plot_linear_h2opt_noise_sweep', 'miscanthus_all',
            'misc_prepare_images', 'misc_train_baselines', 'misc_score_heritability',
            'misc_plot_heritability', 'misc_choose_linear_h2opt_noise',
            'misc_train_linear_h2opt', 'misc_choose_hybrid_cnn_level',
            'misc_train_hybrid_h2opt',
            'image_simulation_all',
            'img_train_linear_h2opt', 'img_fixed_levels', 'img_train_hybrid_h2opt',
            'image_simulation_gwas_all', 'img_run_gwas', 'img_score_gwas', 'img_plot_gwas',
            'image_simulation_null_gwas_all', 'img_null_gwas', 'simulation_null_gwas_all',
            'sim_null_gwas',
            'img_train_baselines', 'img_score_traits', 'img_plot_traits', 'img_build_helios',
            'img_render_helios',
            'arabidopsis_all', 'arabi_prepare_features',
            'arabi_choose_linear_h2opt_noise', 'arabi_train_linear_h2opt',
            'arabi_train_baselines', 'arabi_score_heritability', 'arabi_plot_heritability',
            'misc_choose_pch_ridge', 'miscanthus_gwas_all', 'misc_prepare_genotypes',
            'misc_run_gwas', 'miscanthus_paper_figures', 'misc_score_ndvi',
            'misc_date_correlations', 'misc_measured_correlations',
            'misc_plot_paper_heritability', 'misc_plot_paper_correlations',
            'misc_plot_paper_gwas', 'misc_plot_paper_main', 'sorghum_paper_figures',
            'sorghum_plot_paper_main', 'sorghum_plot_paper_heritability',
            'sorghum_plot_paper_gwas', 'arabidopsis_gwas_all', 'arabi_prepare_genotypes',
            'arabi_run_gwas', 'arabidopsis_paper_figures', 'arabi_paper_pathways',
            'arabi_trait_coefficients',
            'arabi_paper_main_figure', 'arabi_paper_heritability', 'arabi_paper_coefficients',
            'arabi_paper_gwas', 'estimator_check_all', 'check_estimator_inputs',
            'check_lme4_heritability', 'check_plot_estimators',
            'arabidopsis_enrichment_inputs', 'arabi_prepare_precise_masses',
            'arabi_enrichment_inputs', 'arabidopsis_pathway_enrichment',
            'arabi_download_pathway_library', 'arabi_pathway_enrichment',
            'sorghum_null_gwas_all', 'sorghum_null_gwas', 'sorghum_cross_fit_traits',
            'sorghum_crossfit_gwas', 'arabidopsis_null_gwas_all', 'arabi_null_gwas',
            'arabi_cross_fit_traits', 'arabi_crossfit_gwas', 'miscanthus_null_gwas_all',
            'misc_null_gwas', 'misc_cross_fit_traits', 'misc_crossfit_gwas'} <= rules
