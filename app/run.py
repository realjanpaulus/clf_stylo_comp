import argparse
from itertools import product
import logging
import subprocess
import sys
import time

def main():

	program_st = time.time()

	### run logging handler ###

	logging.basicConfig(level=logging.DEBUG, filename="../logs/run.log", filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	
	vectorization_methods = ["bow", "zscore", "tfidf", "cos"]
	max_features = [500, 1000, 2000, 3000]
	n_grams = [(1,1), (1,2), (2,2)]

	# ==================================================
	# Experiment 1: all classifications duration test  #
	# ==================================================
	if args.experiment == 1:
		logging.info("Starting experiment 1: all classifications duration test.")
		vectorization_methods = ["bow"]
		max_features = [1000]
		n_grams = [(1,1)]

	cartesian_inputs = list(product(vectorization_methods, max_features, n_grams))
	for idx, t in enumerate(cartesian_inputs):
		
		logging.info(f"Argument combination {idx+1}/{len(cartesian_inputs)}.")
		logging.info(f"Vectorization method: {t[0]}.")
		logging.info(f"Max Features: {t[1]}.")
		logging.info(f"N-grams: {t[2]}.")

		command = f"python classification.py {args.path} -cr {args.classruns} -cn {args.corpus_name} -mf {t[1]} -ng {t[2][0]} {t[2][1]} -nj {args.n_jobs} -vm {t[0]}"
		
		if args.use_tuning:
			command += " -ut"
		if args.save_date:
			command += " -sd"
		if args.visualization:
			command += " -v"
		
		subprocess.call(["bash", "-c", command])
		print("\n")
	program_duration = float(time.time() - program_st)
	logging.info(f"Overall run-time: {int(program_duration)/60} minute(s).")
	


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="run", description="Runs classification script with multiple arguments.")
	parser.add_argument("path", type=str, help="Path to the corpus.")
	parser.add_argument("--classruns", "-cr", type=int, default=10, help="Sets the number of classification runs.")
	parser.add_argument("--corpus_name", "-cn", type=str, nargs="?", default="prose", help="Indicates the name of the corpus for the output file.")
	parser.add_argument("--experiment", "-e", type=int, default=1, help="Indicates the experiment number.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--use_tuning", "-ut", action="store_true", help="Indicates if hyperparameter optimization should be used.")
	parser.add_argument("--visualization", "-v", action="store_true", help="Indicates if results should be visualized.")
	
	args = parser.parse_args()

	main()