import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
	import time
	st = time.time()
	print("Wow, such empty...") #TODO
	print(time.time() - st)

if __name__ == "__main__":
    
	parser = argparse.ArgumentParser(prog="corpus_reduction", description="Reduces the german fiction corpus to texts of a specific period.")
	parser.add_argument("path", type=str, help="Path to the csv-file of the corpus.")
	args = parser.parse_args()

	main()