import argparse
from itertools import product
import logging
import subprocess
import sys
import time

def main():

	program_st = time.time()

	### run logging handler ###

	logging.basicConfig(level=logging.DEBUG, 
						filename=f"../logs/run_{args.corpus_name}_e{args.experiment}.log", 
						filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	
	vectorization_methods = ["bow", "zscore", "tfidf", "cos"]
	max_features = [200, 300, 500, 1000, 2000, 3000]
	n_grams = [(1,1), (1,2), (2,2)]

	# ==================================================
	# Experiment 1: all classifications duration test  #
	# ==================================================
	if args.experiment == 0:
		logging.info("Starting experiment 0: all classifier duration test.")
		vectorization_methods = ["bow"]
		max_features = [1000]
		n_grams = [(1,1)]
	elif args.experiment == 1:
		logging.info("Starting experiment 1: different mfw for bow.")
		vectorization_methods = ["bow"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(1,1)]
	elif args.experiment == 2:
		logging.info("Starting experiment 2: different mfw for zscore.")
		vectorization_methods = ["zscore"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(1,1)]
	elif args.experiment == 3:
		logging.info("Starting experiment 3: different mfw for tfidf")
		vectorization_methods = ["tfidf"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(1,1)]
	elif args.experiment == 4:
		logging.info("Starting experiment 4: different mfw for cosine.")
		vectorization_methods = ["cos"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(1,1)]
	elif args.experiment == 5:
		logging.info("Starting experiment 5: different mfw and ngrams (2,2) for bow.")
		vectorization_methods = ["bow"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(2,2)]
	elif args.experiment == 6:
		logging.info("Starting experiment 6: different mfw and ngrams (2,2) for zscore.")
		vectorization_methods = ["zscore"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(2,2)]
	elif args.experiment == 7:
		logging.info("Starting experiment 7: different mfw and ngrams (2,2) for tfidf")
		vectorization_methods = ["tfidf"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(2,2)]
	elif args.experiment == 8:
		logging.info("Starting experiment 8: different mfw and ngrams (2,2) for cosine.")
		vectorization_methods = ["cos"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(2,2)]
	elif args.experiment == 9:
		logging.info("Starting experiment 9: different mfw and ngrams (3,3) for bow.")
		vectorization_methods = ["bow"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(3,3)]
	elif args.experiment == 10:
		logging.info("Starting experiment 10: different mfw and ngrams (3,3) for zscore.")
		vectorization_methods = ["zscore"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(3,3)]
	elif args.experiment == 11:
		logging.info("Starting experiment 11: different mfw and ngrams (3,3) for tfidf")
		vectorization_methods = ["tfidf"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(3,3)]
	elif args.experiment == 12:
		logging.info("Starting experiment 12: different mfw and ngrams (3,3) for cosine.")
		vectorization_methods = ["cos"]
		max_features = [200, 300, 500, 1000, 2000, 3000]
		n_grams = [(3,3)]


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