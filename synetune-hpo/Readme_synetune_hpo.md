Creating a blackbox and running HPO in Syne Tune
=================================================

The following work was performed on Syne Tune 0.13.0.

Files and directories are listed in workflow order.

## Create blackbox for Vanillix results

`01_ae_vanillix_import.py` imports from results of Vanillix runs from former compiled results:

	real_ae_results_vanillix_tcga_RNA.parquet
	real_ae_results_vanillix_tcga_METH.parquet
	real_ae_results_vanillix_tcga_DNA.parquet

	real_ae_results_vanillix_schc_RNA.parquet
	real_ae_results_vanillix_schc_METH.parquet
into the blackbox for vanillix.
	
## Create blackbox for Varix results

`01_ae_varix_import.py` imports from results of Varix runs from former compiled results:

	real_ae_results_varix_tcga_RNA.parquet
	real_ae_results_varix_tcga_METH.parquet
	real_ae_results_varix_tcga_DNA.parquet

	real_ae_results_varix_schc_RNA.parquet
	real_ae_results_varix_schc_METH.parquet
into the blackbox for varix.

### Helper Scripts
	
`01a_check_blackbox_properties.py` Check, whether a blackbox actually works. So it can be loaded and
new samples drawn.
	
`01a_check_surrogate.py` checks, whether a surrogate model trained on a blackbox can return
new metrics results.
		
`01b_kfold_cv_on_surrogate_model.py` does k-fold cross validation of XGBoost models, when run with default hyperparameters
for XGBoost.

`01b_kfold_cv_on_surrogate_model_with_hpo.py` does k-fold cross validation of hyperparameter optimized XGBoost models.

## Launch a single HPO optimization method (without transfer)

`02_launch_ae_single_scheduler.py` launches a single scheduler to run on a certain blackbox and task.

## Launch a HPO optimization method (with transfer evaluations from other tasks)

`03_launch_ae_transfer_learning.py` launches a scheduler to run on a certain blackbox and task, that now also uses
transfer evaluations.
			
### Helper scripts

`98_xgb_reg.py` and `98_xgb_reg2.py` investigate tuning XGBoost on a training set with multiple output values for the same input values ("multi-valued (random or parametric) function"). The aim was to check, whether XGBoost can handle this.

`99_gp_regression.py` ivestigates, if Gaussion processes regression could be done on a training set with multiple output values for the same input values ("multi-valued (random or parametric) function"). This question arised very late in time.

`scan_objective_evaluations.py` analyzes Syne Tune blackboxes for the shape of objective evaluations.
Intended to look for multi-seed and/or multi-fidelity scenarios in blackboxes.
