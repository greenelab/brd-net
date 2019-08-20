# This script calls run_plier.R and evaluate_models.py multiple times in order to guage
# evaluate the results of the number of PLIER variables on the models' performance

# The number of processes you want to run at a time for paralellizing run_plier.R
NUM_PROCESSES=4

PLIER_OUT='../data/plier_out'

# K_VALS stores the different values of k (number of PCs) to be used by PLIER.
# This array is intentionally reverse sorted for more efficient scheduling (you want
# the longest jobs to be loaded first)
K_VALS=(100 50 10)


# Pass each K_value to run_plier and execute the different instances in paralell
# The code before the pipe prints all the values of K on its own line
# The code after the pipe tells bash to run run_plier.R once for each value of K,
# But using no more than NUM_PROCESSES threads to do so
printf "$PLIER_OUT %s\n" "${K_VALS[@]}" | xargs -n 2 --max-procs=$NUM_PROCESSES Rscript run_plier.R 2> /dev/null

echo "All run_plier instances completed"

python evaluate_models.py $PLIER_OUT ../data/classifier_healthy.tsv ../data/classifier_disease.tsv --epochs 400
