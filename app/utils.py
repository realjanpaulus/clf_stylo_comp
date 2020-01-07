#!/usr/bin/env python
from datetime import datetime
import glob
from nltk import word_tokenize
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(8,6), 
					 'figure.dpi':300,
					 'font.size': 8})
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict, List, Optional, Tuple, Union

# ===================
# Table of contents #
# ===================

# adjustable parameter 
# preprocessing functions
# experiment helper functions
# visualization helper functions
# analysis helper functions

# ======================
# adjustable parameter #
# ======================

fontsize_vertical = 4
fontsize_horizontal = 6
scaling_l_r = 0.01 # scaling factor for f1-scores on the bars of the classification histogram figures
scaling_t_b = 0.04 # top-bottom: smaller = more downwards

# =========================
# preprocessing functions #
# =========================

def build_chunks(lst: list, n: int) -> list:
	""" Splits a list into n chunks and stores them in a list.
	"""
	return [lst[i:i + n] for i in range(0, len(lst), n)]

def cut_author_from_text(df: pd.DataFrame, column_name: Optional[str] = "text") -> pd.DataFrame:
	""" Cuts the author from the text column of a DataFrame.
	"""
	for index, row in df.iterrows():
		if row["author"] in row["text"]:
			df.at[index, "text"] = row["text"].replace(row["author"], "")
	return df

def remove_columnname_from_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
	""" Removes the title and author within the first 200 tokens of the text column.
	"""
	for index, row in df.iterrows():
		column = row[column_name]
		text = row["text"]

		for idx, string in enumerate(text):
			cut_text = text[:idx]
			sen = [column, cut_text]
			if idx >= 200:
				break

			vectorizer = CountVectorizer()
			vec = vectorizer.fit_transform(sen).toarray()
			csim = cosine_similarity(vec)

			# utilizing the cosine similarity to find similiar strings
			if csim[0][1] >= 0.60:
				df.at[index, "text"] = text[len(cut_text):]
				break
		return df

def split_texts_into_segments(corpus: pd.DataFrame,
							  n: Optional[int] = 10000,
							  same_len: Optional[bool] = False) -> pd.DataFrame:
	""" Splits the texts of a corpus into n segments and returns them as a new corpus
		(A number is added to the file name and title to distinguish them).
		If same_len, segments with lengths smaller than n will be ignored.
	"""
	tmp_dict = {}
	
	for index, row in corpus.iterrows():
		chunks = build_chunks(word_tokenize(row["text"]), n)
		for idx_chunk, chunk in enumerate(chunks):
			
			
			new_filename = row["filename"] + "_" + str(idx_chunk + 1)
			new_title = row["title"] + "_" + str(idx_chunk + 1)
			new_textlength = len(chunk)
			if same_len:
				if new_textlength == n:
					new_text = " ".join(chunk)
					tmp_dict[new_filename] = {"author" : row["author"],
											  "title" : new_title,
											  "year" : row["year"],
											  "textlength" : new_textlength,
											  "text" : new_text
											  }
			else:
				new_text = " ".join(chunk)
				tmp_dict[new_filename] = {"author" : row["author"],
										  "title" : new_title,
										  "year" : row["year"],
										  "textlength" : new_textlength,
										  "text" : new_text
										  }
	
	new_corpus = pd.DataFrame.from_dict(tmp_dict, orient="index").reset_index()
	new_corpus.columns = ["filename", "author", "title", 
						  "year", "textlength", "text"]
	return new_corpus

def unify_texts_amount(df: pd.DataFrame,
					   by_column: str,
					   max_value: Optional[int] = 10,
					   use_smallest_amount: Optional[bool] = False,
					   columns: Optional[List[str]] = ["author", "text"]) -> pd.DataFrame:
	""" Takes a DataFrame and unifies the amount of rows per label by an 'max_value' 
		or optionally the label with the smallest amount within the DataFrame.
	"""
	d = df[by_column].value_counts().to_dict()
	if use_smallest_amount:
		max_value = d[min(d, key=d.get)]
	df = df.groupby(by_column).filter(lambda x: len(x) > max_value)

	actual_value = ""
	new_df = pd.DataFrame(columns=columns)
	df = df.sort_values(by=[by_column])
	for idx, value in enumerate(df[by_column]):
		if actual_value != value:
			actual_value = value
			new_df = new_df.append(df.loc[df[by_column] == value][0:max_value], sort=False)

	return new_df.drop_duplicates()

# =============================
# experiment helper functions #
# =============================

def document_term_matrix(corpus: pd.DataFrame,
						 vectorization_method: str, 
						 document_column: str,
						 lower: Optional[str] = True,
						 max_features: Optional[int] = 2000,
						 ngram_range: Optional[Tuple[int, int]] = (1,1),
						 sparse: Optional[bool] = False,
						 text_column: Optional[str] = "text") -> Union[pd.DataFrame, csr_matrix]:
	""" Computes a Document Term Matrix and a Matrix of token counts.
	"""
	if vectorization_method == "tfidf":
		vectorizer = TfidfVectorizer(max_features=max_features, lowercase=lower)
	elif vectorization_method == "cos":	
		vectorizer = TfidfVectorizer(max_features=max_features, lowercase=lower)
	else:
		vectorizer = CountVectorizer(max_features=max_features, lowercase=lower)
	

	vector = vectorizer.fit_transform(corpus[text_column])
	features = vectorizer.get_feature_names()

	documents = corpus[document_column]
	dtm = pd.DataFrame(vector.toarray(), index=list(documents), columns=features)

	if vectorization_method == "zscore":
		dtm = dtm.apply(z_score)
	elif vectorization_method == "cos":
		dtm = cosine_similarity(vector, vector)
	if sparse and vectorization_method != "cos":
		dtm = csr_matrix(dtm.values)
	return dtm, vector

def z_score(x: int) -> float:
	""" Computes z-score."""
	return (x-x.mean()) / x.std()

# ================================
# visualization helper functions #
# ================================

def horizontal_hist(results: pd.DataFrame, 
					classruns: int,
				  	cross_validation = int,
					clf_visualization: Optional[bool] = False,
					ngram: Optional[Tuple[int, int]] = (1,1), 
					output_name: Optional[str] = "", 
					save_date: Optional[bool] = False, 
					vectorization_method: Optional[str] = ""):
	
	del results.index.name
	results = results[["cv", "f1"]]
	ax = results.plot.barh(color=["#a05195", "#003f5c"], 
						   edgecolor="black")

	for p in ax.patches: 
		lr = 0.015 #left-right: smaller = more to the left
		tb = scaling_t_b #top-bottom: smaller = more downwards
		ax.text(p.get_width()+lr, 
				p.get_y()+tb, 
				str(np.around((p.get_width()), decimals=2)),
				fontsize=fontsize_horizontal)
	plt.xticks(np.arange(0,1.1,0.1))

	if clf_visualization:
		plt.title(r"$\bf" + f"{output_name}" + "$\n", loc="left")

		if save_date:
			figure_name = f"{output_name}_barh_({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
		else:
			figure_name = f"{output_name}_barh"
		plt.savefig(f'../data/figures/clf_results/{figure_name}.png', dpi=300, bbox_inches='tight')
	else:
		plt.title(f" Corpus: {output_name}\n Weighting: {vectorization_method} \n N-grams: {str(ngram)} \n Train-test iterations: {classruns} \n Cross-Validation: {cross_validation}", loc="left")

		if save_date:
			figure_name = f"results_barh_{output_name}({vectorization_method}_{classruns}_{ngram}) ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
		else:
			figure_name = f"results_barh_{output_name}({vectorization_method}_{classruns}_{ngram})"
		plt.savefig(f'../data/figures/results/{figure_name}.png', dpi=300, bbox_inches='tight')

def vertical_hist(results: pd.DataFrame, 
				  classruns: int,
				  cross_validation = int,
				  clf_visualization: Optional[bool] = False,
				  ngram: Optional[Tuple[int, int]] = (1,1),
				  output_name: Optional[str] = "",
				  save_date: Optional[bool] = False,
				  vectorization_method: Optional[str] = ""):
	
	del results.index.name
	ax = results.plot.bar(color=["#003f5c","#a05195"], 
						  edgecolor="black")
	for p in ax.patches: 
		ax.annotate(np.round(p.get_height(), decimals=2), 
					(p.get_x()+(p.get_width()/2.)+scaling_l_r, p.get_height()),
					 ha='center', 
					 va='center', 
					 xytext=(0, 10), 
					 textcoords='offset points',
					 fontsize=fontsize_vertical)
		plt.yticks(np.arange(0,1.1,0.1))

	if clf_visualization:
		plt.title(r"$\bf" + f"{output_name}" + "$\n", loc="left")

		if save_date:
			figure_name = f"{output_name}_bar_({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
		else:
			figure_name = f"{output_name}_bar"
		plt.savefig(f'../data/figures/clf_results/{figure_name}.png', dpi=300, bbox_inches='tight')

	else:
		plt.title(f" Corpus: {output_name}\n Weighting: {vectorization_method} \n N-grams: {str(ngram)} \n Train-test iterations: {classruns} \n Cross-Validation: {cross_validation}", loc="left")

		if save_date:
			figure_name = f"results_bar_{output_name}({vectorization_method}_{classruns}_{ngram}) ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
		else:
			figure_name = f"results_bar_{output_name}({vectorization_method}_{classruns}_{ngram})"
		plt.savefig(f'../data/figures/results/{figure_name}.png', dpi=300, bbox_inches='tight')

def pie(results: pd.DataFrame,
		classruns: int,
		cross_validation = int,
	  	ngram: Optional[Tuple[int, int]] = (1,1),
	  	output_name: Optional[str] = "",
		save_date: Optional[bool] = False,
		vectorization_method: Optional[str] = ""):
	
	plt.cla()
	plt.figure(figsize=(6,4))
	colors = ["#665191", "#ffa999", "#8eca98", 
			  "#003f5c", "#007885", "#ffa600",
			  "#003f5c", "#2f4b7c", "#a05195",
			  "#d45087", "#f95d6a", "#ff7c43"]

	results["percentage"] = results.durations / results.durations.sum()
	plt.pie(results["percentage"], 
			autopct='%.0f%%',
			labels=results["clf"],
			radius=1.,
			colors=colors[::-1],
			textprops={'fontsize': 7})
	title_text = r"\ Duration\ of\ " + f"{output_name}" + r"\ corpus"
	bold_title = r"$\bf{" + title_text + "}$ \n"
	plt.title(bold_title + f" Corpus: {output_name}\n Weighting: {vectorization_method} \n N-grams: {str(ngram)} \n Train-test iterations: {classruns} \n Cross-Validation: {cross_validation}", 
			  loc="left",
			  fontsize=7)
	if save_date:
		figure_name = f"pie_{output_name}({vectorization_method}_{classruns}_{ngram}) ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
	else:
		figure_name = f"pie_{output_name}({vectorization_method}_{classruns}_{ngram})"
	plt.savefig(f'../data/figures/durations/{figure_name}.png', dpi=300, bbox_inches='tight')

def visualize(results: pd.DataFrame, 
			  visualization_method: str,
			  classruns: int,
			  cross_validation = int,
			  clf_visualization: Optional[bool] = False,
			  ngram: Optional[Tuple[int, int]] = (1,1),
			  output_name: Optional[str] = "",
			  save_date: Optional[bool] = False,
			  vectorization_method: Optional[str] = ""):
	
	if clf_visualization:
		if visualization_method == "bar_vertical":
			vertical_hist(results, 
					  	  classruns=0, 
				  		  cross_validation = 0,
					  	  clf_visualization = True,
						  ngram=ngram, 
						  output_name=output_name,
						  save_date=save_date)
		elif visualization_method == "bar_horizontal":
			horizontal_hist(results, 
							classruns=0,
				  		  	cross_validation = 0,
							clf_visualization = True, 
							ngram=ngram, 
							output_name=output_name,
							save_date=save_date)
	else:
		vectorization_methods = {"bow": "Bag of words",
								 "cos": "Cosine similarity",
								 "tfidf": "TF-IDF",
								 "zscore": "Z-Score"}
		
		if visualization_method == "bar_vertical":
			vertical_hist(results, 
					  	  classruns,
				  		  cross_validation = cross_validation,
						  ngram=ngram, 
						  output_name=output_name,
						  save_date=save_date,
						  vectorization_method=vectorization_methods[vectorization_method])
		elif visualization_method == "bar_horizontal":
			horizontal_hist(results, 
							classruns, 
				  		  	cross_validation = cross_validation,
							ngram=ngram, 
							output_name=output_name,
							save_date=save_date,
							vectorization_method=vectorization_methods[vectorization_method])
		elif visualization_method == "pie":
			pie(results,
				classruns,
	  		  	cross_validation = cross_validation,
				ngram=ngram, 
				output_name=output_name,
				save_date=save_date,
				vectorization_method=vectorization_methods[vectorization_method])


# ===========================
# analysis helper functions #
# ============================

def concat_tables(dir_path: str, 
                  corpus_name: Optional[str] = "prose",
                  save_date: Optional[bool] = True) -> pd.DataFrame:
    
    all_files = glob.glob(dir_path + "/*.csv")
    all_tables = {}

    for filename in all_files:
        if corpus_name in filename:
            table = pd.read_csv(filename, index_col=None, header=0)

            if save_date:
                subtract = 17
            else:
                subtract = 0
            name_addition = filename[filename.find(corpus_name)+len(corpus_name)+1:filename.find(").csv")-subtract]
            table.columns = ["clf", "f1", "cv"]
            table["clf"] = table["clf"].astype(str) + ": " + name_addition
            all_tables[name_addition] = table
    return pd.concat(all_tables.values(), axis=0, ignore_index=True)

def split_tables_by_clf(table: pd.DataFrame, 
                        saving_dir_path: str):
    classifiers = ["KNN", "NSC", "MNB", "LR", "LSVM"]
    
    for clf in classifiers:
        clf_table = table[table.clf.str.contains(clf)]
        clf_table.to_csv(saving_dir_path+clf+".csv", index=False)

