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
	if args.clf_visualization:
		results.set_index("clf", inplace=True)
		visualize(results, 
				  args.visualization_method,
				  classruns = 0,
				  cross_validation = 0,
				  clf_visualization = True,
				  output_name = args.path[len(args.directory_name+"clf_tables/"):args.path.find(".csv")])
	else:
		#TODO: besser machen
		if args.duration_visualization:
			vectorization_method = ""
			classruns = 1
			ngram = (1,1)
		else:
			corpus_name = re.findall(r"^[^\(]+", args.path[len(args.directory_name):])
			corpus_name = corpus_name[0][len("classification")+1:]
			parameter_names = args.path[args.path.find(corpus_name)+len(corpus_name)+1:-5].split("_")

			vectorization_method = parameter_names[0]
			classruns = parameter_names[1]
			ngram = parameter_names[2]
		
		visualization_methods = ["bar_vertical", 
								 "bar_horizontal",
								 "pie"]
		if args.visualization_method not in visualization_methods:
			logging.info(f"The visualization method '{visualization_method}' isn't available. The available visualization methods are: {visualization_methods}.")
			sys.exit()
		visualize(results, 
				  args.visualization_method,
				  classruns,
				  cross_validation = 0, #can't be obtained by visualization.py
				  ngram=ngram, 
				  save_date=args.save_date,
				  vectorization_method=vectorization_method)
	

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="visualization", description="Visualization of the classifications.")
	parser.add_argument("path", type=str, help="Path to the classification results as csv-file.")
	parser.add_argument("--clf_visualization", "-clf", action="store_true", help="Indicates if specific classifier results should be visualized.")
	parser.add_argument("--directory_name", "-dm", type=str, default="../data/tables/", help="Name of the directory path.")
	parser.add_argument("--duration_visualization", "-dv", action="store_true", help="Indicates if classifier durations should be visualized.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--visualization_method", "-vm", type=str, default="bar_vertical", help="Indicates the Visualization Method. Possible values are 'bar_vertical', 'bar_horizontal', 'pie'.")
	
	args = parser.parse_args()

	main()