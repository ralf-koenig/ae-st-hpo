# ae-st-hpo

This repository provides glue code written in Python to combine results from Autoencodix training runs with Syne Tune 
HPO to make use of the knowledge of previous Autoencodix runs in Syne Tune.  

See [autoencodix-hpo/README.md](autoencodix-hpo/README.md) for the steps to run Autoencodix on many hyperparameter 
combinations. For the actual runs, you need to run Autoencodix on a (SLURM) cluster.

See [synetune-hpo/README.md](synetune-hpo/README.md) for the steps in Syne Tune to create blackboxes and run HPO
algorithms. This cen be done on an average PC with Syne Tune installed.

See [Master Thesis (PDF)](Meta_hyperparameter_optimization_of_autoencoders_for_multi_omics_data_integration.pdf) for 
background information, related work, results, discussion.
