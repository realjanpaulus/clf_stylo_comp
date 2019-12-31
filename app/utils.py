from nltk import word_tokenize
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict, List, Optional, Tuple, Union

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
						 document_column: str,
						 lower: Optional[str] = True,
						 max_features: Optional[int] = 2000,
						 ngram_range: Optional[Tuple[int, int]] = (1,1),
						 sparse: Optional[bool] = False,
						 text_column: Optional[str] = "text",
						 tfidf: Optional[bool] = False,
						 z_norm: Optional[bool] = False) -> Union[pd.DataFrame, csr_matrix]:
	""" Computes a Document Term Matrix and a Matrix of token counts.
	"""
	if tfidf:
		vectorizer = TfidfVectorizer(max_features=max_features, lowercase=lower)
	else:
		vectorizer = CountVectorizer(max_features=max_features, lowercase=lower)
	vector = vectorizer.fit_transform(corpus[text_column])
	features = vectorizer.get_feature_names()

	documents = corpus[document_column]
	dtm = pd.DataFrame(vector.toarray(), index=list(documents), columns=features)

	if z_norm:
		dtm = dtm.apply(z_score)
	if sparse:
		dtm = csr_matrix(dtm.values)
	return dtm, vector

def z_score(x: int) -> float:
	""" Computes z-score."""
	return (x-x.mean()) / x.std()
