import argparse
from collections import defaultdict
from datetime import datetime
import logging
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(8,6), 'figure.dpi':150})
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
import time
from utils import document_term_matrix


#TODO:
# - lr
# - extra: decision tress (da sie so lange dauern) 

### texts_to_csv logging handler ###
logging.basicConfig(level=logging.INFO, filename="../logs/classification.log", filemode="w")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def main():
	st = time.time()

	# =======================
	# predefined parameters #
	# =======================

	if args.ngram:
		ngram_range = tuple(args.ngram)
	else:
		ngram_range = (1,1)

	classruns = args.classruns
	
	# ================================
	# corpus reading and vectorizing # 
	# ================================

	corpus = pd.read_csv(args.path)

	vectorization_method = "Bag of words"
	if args.zscore:
		vectorization_method = "Z-Scores"
		dtm, vector = document_term_matrix(corpus, "title", ngram_range=ngram_range, sparse=True, z_norm=True)
		logging.info("Multinomial Naive Bayes can't be used with a z-score matrix. It won't be used.")
	else:
		if args.tfidf:
			vectorization_method = "Tf-Idf"
			dtm, vector = document_term_matrix(corpus, "title", ngram_range=ngram_range, tfidf=True, sparse=True, z_norm=False)
		else:
			dtm, vector = document_term_matrix(corpus, "title", ngram_range=ngram_range, sparse=True, z_norm=False)
	

	logging.info(f"Read and vectorized corpus ({int((time.time() - st)/60)} minute(s)).")


	# ================
	# classification # 
	# ================

	f1_dict = defaultdict(list)
	cv_dict = defaultdict(list)

	for run in list(range(1, classruns+1)):
		logging.info(f"Iteration: {run}/{classruns}")
		X_train, X_test, y_train, y_test = train_test_split(dtm, 
															corpus["author"],
															test_size=0.2,
															stratify=corpus["author"],
															shuffle=True)	
		
		# ==================================================
		# K-Nearest Neighbors + z-scores (= Burrows Delta) #
		# ==================================================
		knn_clf = KNeighborsClassifier()
		knn_model = knn_clf.fit(X_train, y_train)
		knn_y_pred = knn_model.predict(X_test)
		knn_f1_score = f1_score(y_test, knn_y_pred, average="micro")
		knn_cross_val = np.mean(cross_val_score(knn_clf, X_train, y_train, cv=2, scoring="f1_micro"))
		f1_dict["D+KNN"].append(knn_f1_score)
		cv_dict["D+KNN"].append(knn_cross_val)

		# =============================================================
		# Nearest (shrunken) Centroids + z-scores (= Burrows Delta 2) #
		# =============================================================
		nsc_clf = NearestCentroid()
		nsc_model = nsc_clf.fit(X_train, y_train)
		nsc_y_pred = nsc_model.predict(X_test)
		nsc_f1_score = f1_score(y_test, nsc_y_pred, average="micro")
		nsc_cross_val = np.mean(cross_val_score(nsc_clf, X_train, y_train, cv=2, scoring="f1_micro"))
		f1_dict["D+NSC"].append(nsc_f1_score)
		cv_dict["D+NSC"].append(nsc_cross_val)

			
		# ================================
		# Logistic Regression + z-scores #
		# ================================
		"""
		lr_clf = LogisticRegression(multi_class="multinomial", solver="saga")
		lr_model = lr_clf.fit(X_train, y_train)
		lr_y_pred = lr_model.predict(X_test)
		lr_f1_score = f1_score(y_test, lr_y_pred, average="micro")
		lr_cross_val = np.mean(cross_val_score(lr_clf, X_train, y_train, cv=2))
		f1_dict["Logistic Regression"].append(lr_f1_score)
		cv_dict["Logistic Regression"].append(lr_cross_val)
		"""
		# =================================================
		# Multinomial Naive Bayes (only without z-scores) #
		# =================================================
		
		if not args.zscore:
			mnb_clf = MultinomialNB()
			mnb_model = mnb_clf.fit(X_train, y_train)
			mnb_y_pred = mnb_model.predict(X_test)
			mnb_f1_score = f1_score(y_test, mnb_y_pred, average="micro")
			mnb_cross_val = np.mean(cross_val_score(mnb_clf, X_train, y_train, cv=2, scoring="f1_micro"))
			f1_dict["MNB"].append(mnb_f1_score)
			cv_dict["MNB"].append(mnb_cross_val)

		# ====================================
		# Support Vector Machines + z-scores #
		# ====================================
		svm_clf = SVC(kernel="linear")
		svm_model = svm_clf.fit(X_train, y_train)
		svm_y_pred = svm_model.predict(X_test)
		svm_f1_score = f1_score(y_test, svm_y_pred, average="micro")
		svm_cross_val = np.mean(cross_val_score(svm_clf, X_train, y_train, cv=2, scoring="f1_micro"))
		f1_dict["SVM"].append(svm_f1_score)
		cv_dict["SVM"].append(svm_cross_val)

	# ================
	# Saving results #
	# ================
	
	final_f1_dict = {}
	for (f1_k, f1_v), (cv_k, cv_v) in zip(f1_dict.items(), cv_dict.items()):
		if f1_k == cv_k:
			final_f1_dict[f1_k] = {"f1": np.mean(f1_v), "cv": np.mean(cv_v)}
	
	results = pd.DataFrame(final_f1_dict).round(3).T

	
	if args.save_date:
		csv_name = f"classification_{args.output_name} ({vectorization_method}) ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M}).csv"
	else:
		csv_name = f"classification_{args.output_name} ({vectorization_method}).csv"
	results.to_csv(f"../data/tables/{csv_name}.csv")

	
	ax = results.plot.bar(color=["#003f5c","#a05195"], edgecolor="black")
	for p in ax.patches: 
	    ax.annotate(np.round(p.get_height(),decimals=3), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
	plt.yticks(np.arange(0,1.1,0.1))
	plt.title(f" Corpus: {args.output_name}\n Weighting: {vectorization_method} \n N-grams: {args.ngram} \n Train-test iterations: {classruns} ", loc="left")


	if args.save_date:
		figure_name = f"results_bar_{args.output_name} ({vectorization_method}) ({datetime.now():%d.%m.%y}_{datetime.now():%H:%M}).png"
	else:
		figure_name = f"results_bar_{args.output_name} ({vectorization_method}).png"
	plt.savefig(f'../data/figures/results/{figure_name}', dpi=300, bbox_inches='tight')

	logging.info(f"Run-time: {int((time.time() - st)/60)} minute(s).")

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="classification", description="Comparison of different classifier.")
	parser.add_argument("path", type=str, help="Path to the corpus as csv-file.")
	parser.add_argument("--classruns", "-cr", type=int, nargs="?", default=20, help="Sets the number of classification runs.")
	parser.add_argument("--ngram", "-ng", type=int, nargs="?", default=(1, 1), help="Passes the ngram range.")
	parser.add_argument("--output_name", "-on", type=str, nargs="?", default="corpus", help="Indicates the name of the corpus for the output file.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--tfidf", "-tf", action="store_true", help="Indicates if tfidf should be used for data vectorization.")
	parser.add_argument("--zscore", "-zs", action="store_true", help="Indicates if z-scores normalization should be used.")

	
	args = parser.parse_args()

	main()