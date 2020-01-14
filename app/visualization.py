import argparse
import logging
import numpy as np
import pandas as pd
import re
import sys
from typing import Dict, List, Optional, Tuple
from utils import visualize

### visualization logging handler ###
logging.basicConfig(level=logging.INFO, filename="../logs/visualization.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def main():

	logging.info("Reading csv.")
	results = pd.read_csv(args.path, index_col=0)
	logging.info("Plotting graph.")

	#TODO: überprüfen, auch max_features
	if args.clf_visualization:
		results.set_index("clf", inplace=True)
		visualize(results, 
				  args.visualization_method,
				  classruns = 0,
				  cross_validation = 0,
				  max_features = "",
				  clf_visualization = True,
				  output_name = args.path[len(args.directoryname+"clf_tables/"):args.path.find(".csv")])
	else:
		# Example filename: "../data/tables/classification_speeches(bow_1_(1, 1)).csv"
		if args.filename:
			filename = args.filename
		else:
			filename = args.path

		
		"""
		if args.duration_visualization and not filename:
			logging.info("No specific filename is given. Empty Parameter names will be used.")
			vectorization_method = ""
			classruns = 0
			max_features = 0
			ngram = (1,1)"""

		if args.duration_visualization:
			pre = "clf_durations"
			visualization_method = "pie"
		else:
			pre = "classification"
			visualization_method = args.visualization_method

		corpus_name = re.findall(r"^[^\(]+", filename[len(args.directoryname):])
		corpus_name = corpus_name[0][len(pre)+1:]
		parameter_names = filename[filename.find(corpus_name)+len(corpus_name)+1:-5].split("_")


		vectorization_method = parameter_names[0]
		classruns = parameter_names[1]
		max_features = parameter_names[2]
		ngram = parameter_names[3]
		if ngram[-2:] == "))":
			ngram = parameter_names[3][:-1]
		
		visualization_methods = ["bar_vertical", "bar_horizontal", "pie"]
		if visualization_method not in visualization_methods:
			logging.info(f"The visualization method '{visualization_method}' isn't available. The available visualization methods are: {visualization_methods}.")
			sys.exit()
		visualize(results=results, 
				  visualization_method=visualization_method,
				  classruns=classruns,
				  cross_validation = args.cross_validation,
				  max_features=max_features,
				  ngram=ngram, 
				  output_name=corpus_name,
				  save_date=args.save_date,
				  vectorization_method=vectorization_method)


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="visualization", description="Visualization of the classifications.")
	parser.add_argument("path", type=str, help="Path to the classification results as csv-file.")
	parser.add_argument("--clf_visualization", "-clf", action="store_true", help="Indicates if specific classifier results should be visualized.")
	parser.add_argument("--cross_validation", "-cv", type=int, default=0, help="Indicates the cross_validation value.")
	parser.add_argument("--directoryname", "-dm", type=str, default="../data/tables/", help="Name of the directory path.")
	parser.add_argument("--duration_visualization", "-dv", action="store_true", help="Indicates if classifier durations should be visualized.")
	parser.add_argument("--filename", "-fn", type=str, help="Special filename for information extraction.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--visualization_method", "-vm", type=str, default="bar_vertical", help="Indicates the Visualization Method. Possible values are 'bar_vertical', 'bar_horizontal', 'pie'.")
	
	args = parser.parse_args()

	main()
