import argparse
import glob
import logging
import pandas as pd
from utils import split_tables_by_clf, concat_tables, visualize

### analysis logging handler ###
logging.basicConfig(level=logging.INFO, filename="../logs/analysis.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def main():
	logging.info("Reading csv files.")
	table = concat_tables(args.dir_path, args.corpus_name)
	logging.info("Writing csv files.")
	output_path = args.dir_path+"clf_tables/"
	split_tables_by_clf(table, output_path)
	logging.info(r"Saving classifier figures to data/figures/clf_results/")

	visualization_methods = ["bar_vertical", "bar_horizontal"]
	if args.visualization_method not in visualization_methods:
		logging.info(f"The visualization method '{visualization_method}' isn't available. The available visualization methods are: {visualization_methods}.")
		sys.exit()
	all_files = glob.glob(output_path + "/*.csv")
	for idx, filename in enumerate(all_files):
		clf_table = pd.read_csv(filename, index_col="clf")
		if args.save_date:
			subtract = 17
		else:
			subtract = 0
		clf_name = filename[len(output_path):filename.find(".csv")-subtract]
		logging.info(f"Table {idx+1}/{len(all_files)}: {clf_name}.")
		visualize(clf_table,
				  args.visualization_method,
				  classruns = 0,
				  cross_validation = 0,
				  clf_visualization = True,
				  output_name = clf_name,
				  save_date = args.save_date)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="analysis", description="analysis of the classification results.")
	parser.add_argument("dir_path", type=str, help="Path to the directory with classifications results as csv-file.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="prose", help="Name of the corpus.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the figures should be saved.")
	parser.add_argument("--visualization_method", "-vm", type=str, default="bar_vertical", help="Indicates the Visualization Method.")
	args = parser.parse_args()

	main()

