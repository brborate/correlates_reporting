#!/bin/bash
ml fhR/4.0.4-foss-2020b
ml jbigkit
#sbatch -c10 --mem 11G --time=7-0 run_cvsl_with_riskscore_on_vaccinees.sh
# sbatch -c10 --mem 11G --time=7-0  --array=1-12 --constraint=gizmok code/run_cvsl_varsets.sh
sbatch -c10 --mem 11G --time=7-0 --constraint=gizmok code/run_cvsl_varsets.sh