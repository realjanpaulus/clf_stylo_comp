#!/usr/bin/env python
import argparse
from collections import defaultdict
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score 
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import sys
import time
from utils import document_term_matrix
from utils import visualize

def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()
	clf_durations = defaultdict(list)

	# ================
	# corpus reading # 
	# ================

	corpus = pd.read_csv(args.path)

	# =======================
	# predefined parameters #
	# =======================

	n_jobs = args.n_jobs

	if args.ngram:
		ngrams = tuple(args.ngram)
	else:
		ngrams = (1,1)


	classruns = args.classruns
	train_size = 0.8
	test_size = 0.2

	# set dynamic cross validation value
	cv = int((min(corpus.groupby('author')['text'].nunique())) * train_size)
	if cv == 1:
		cv += 1
	elif cv > 10:
		cv = 10
	
	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/extended_classification_{args.corpus_name}({args.vectorization_method}_{args.max_features}_{ngrams}).log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	mpl_logger = logging.getLogger('matplotlib')
	mpl_logger.setLevel(logging.WARNING)

	logging.info(f"Dynamically determined cross validation value is {cv}.")

	

	# =============
	# vectorizing # 
	# =============

	if args.vectorization_method == "bow":
		dtm, vector = document_term_matrix(corpus, 
										   "bow", 
										   "title", 
										   max_features=args.max_features,
										   ngram_range=ngrams, 
										   sparse=True)
	elif args.vectorization_method == "zscore" or args.vectorization_method == "zcos":
		dtm, vector = document_term_matrix(corpus, 
										   "zscore", 
										   "title", 
										   max_features=args.max_features,
										   ngram_range=ngrams, 
										   sparse=True)
		logging.info("Multinomial Naive Bayes can't be used with a z-score matrix, so it won't be used.")
	elif args.vectorization_method == "tfidf":
		dtm, vector = document_term_matrix(corpus, 
										   "tfidf", 
										   "title", 
										   max_features=args.max_features,
										   ngram_range=ngrams, 
										   sparse=True)
	else:
		logging.info(f"The vectorization method '{args.vectorization_method}' isn't a available.")
		sys.exit()

	logging.info(f"Read and vectorized corpus ({int((time.time() - program_st)/60)} minute(s)).")
	logging.info(f"Vectorization method: {args.vectorization_method}.")
	
	# ================
	# classification # 
	# ================

	f1_dict = defaultdict(list)
	cv_dict = defaultdict(list)

	for run in list(range(1, classruns+1)):

		logging.info(f"Iteration: {run}/{classruns}")
		X_train, X_test, y_train, y_test = train_test_split(dtm, 
															corpus["author"],
															test_size=test_size,
															stratify=corpus["author"],
															shuffle=True)	
		

		# =========================================================================== #
		# K-Nearest Neighbors (+ z-scores = Burrows Delta)                            #
		#					  (+ z-scores & cosine = Burrows Delta (Smiths-Version))  #						   #
		# =========================================================================== #

		knn_st = time.time()

		if args.vectorization_method == "zcos":
			knn_clf = KNeighborsClassifier(metric="cosine", 
										   algorithm="brute")
		else:
			knn_clf = KNeighborsClassifier()
		knn_model = knn_clf.fit(X_train, y_train)
		knn_y_pred = knn_model.predict(X_test)
		knn_f1_score = f1_score(y_test, knn_y_pred, average="micro")
		knn_cross_val = np.mean(cross_val_score(knn_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
		if args.vectorization_method == "zscore":
			f1_dict["D-KNN"].append(knn_f1_score)
			cv_dict["D-KNN"].append(knn_cross_val)

			knn_duration = float(time.time() - knn_st)
			clf_durations["D-KNN"].append(knn_duration)
			logging.info(f"Run-time D-KNN: {knn_duration} seconds")
		elif args.vectorization_method == "zcos":
			f1_dict["SD-KNN"].append(knn_f1_score)
			cv_dict["SD-KNN"].append(knn_cross_val)

			knn_duration = float(time.time() - knn_st)
			clf_durations["SD-KNN"].append(knn_duration)
			logging.info(f"Run-time SD-KNN: {knn_duration} seconds")
		else:
			f1_dict["KNN"].append(knn_f1_score)
			cv_dict["KNN"].append(knn_cross_val)

			knn_duration = float(time.time() - knn_st)
			clf_durations["KNN"].append(knn_duration)
			logging.info(f"Run-time KNN: {knn_duration} seconds")

		# Hyperparameter optimization #

		if args.use_tuning:
			tknn_st = time.time()
			if args.vectorization_method == "zcos":
				tknn_parameters = {"metric": ["cosine"],
								   "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
								   "weights": ["uniform", "distance"],
								   "algorithm": ["brute"],
								   "leaf_size": [10, 20, 30, 40, 50],
								   "n_jobs": [n_jobs]}
			else:
				tknn_parameters = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
								   "weights": ["uniform", "distance"],
								   "algorithm": ["brute"],
								   "leaf_size": [10, 20, 30, 40, 50],
								   "p": [1, 2], 
								   "n_jobs": [n_jobs]}
							   
			# weights: 'ball_tree' and 'kd_tree' doesn't work with sparse
			# p = 1: manhatten, p = 2: euclidean (minkowski doesn't work)

			tknn_grid = GridSearchCV(knn_clf, tknn_parameters, cv=cv, scoring="f1_micro")
			tknn_grid.fit(X_train, y_train)

			tknn_clf = KNeighborsClassifier(**tknn_grid.best_params_)
			tknn_model = tknn_clf.fit(X_train, y_train)
			tknn_y_pred = tknn_model.predict(X_test)
			tknn_f1_score = f1_score(y_test, tknn_y_pred, average="micro")
			tknn_cross_val = np.mean(cross_val_score(tknn_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			if args.vectorization_method == "zscore":
				f1_dict["D-tKNN"].append(tknn_f1_score)
				cv_dict["D-tKNN"].append(tknn_cross_val)
				logging.debug(f"D-tKNN best params: {tknn_grid.best_params_}")

				tknn_duration = float(time.time() - tknn_st)
				clf_durations["D-tKNN"].append(tknn_duration)
				logging.info(f"Run-time D-tKNN: {tknn_duration} seconds")
			elif args.vectorization_method == "zcos":
				f1_dict["SD-tKNN"].append(tknn_f1_score)
				cv_dict["SD-tKNN"].append(tknn_cross_val)
				logging.debug(f"SD-tKNN best params: {tknn_grid.best_params_}")

				tknn_duration = float(time.time() - tknn_st)
				clf_durations["SD-tKNN"].append(tknn_duration)
				logging.info(f"Run-time SD-tKNN: {tknn_duration} seconds")
			else:
				f1_dict["tKNN"].append(tknn_f1_score)
				cv_dict["tKNN"].append(tknn_cross_val)
				logging.debug(f"tKNN best params: {tknn_grid.best_params_}")

				tknn_duration = float(time.time() - tknn_st)
				clf_durations["tKNN"].append(tknn_duration)
				logging.info(f"Run-time tKNN: {tknn_duration} seconds")

		# ==================================================================================== #
		# Nearest (shrunken) Centroids (+ z-scores = Burrows Delta 3)                          #
		#					           (+ z-scores & cosine = Burrows Delta 3(Smiths-Version)) #						   #
		# ==================================================================================== #
		
		nsc_st = time.time()

		if args.vectorization_method == "zcos":
			nsc_clf = NearestCentroid(metric="cosine")
		else:
			nsc_clf = NearestCentroid()
		nsc_model = nsc_clf.fit(X_train, y_train)
		nsc_y_pred = nsc_model.predict(X_test)
		nsc_f1_score = f1_score(y_test, nsc_y_pred, average="micro")
		nsc_cross_val = np.mean(cross_val_score(nsc_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
		if args.vectorization_method == "zscore":
			f1_dict["D-NSC"].append(nsc_f1_score)
			cv_dict["D-NSC"].append(nsc_cross_val)

			nsc_duration = float(time.time() - nsc_st)
			clf_durations["D-NSC"].append(nsc_duration)
			logging.info(f"Run-time D-NSC: {nsc_duration} seconds")
		elif args.vectorization_method == "zcos":
			f1_dict["SD-NSC"].append(nsc_f1_score)
			cv_dict["SD-NSC"].append(nsc_cross_val)

			nsc_duration = float(time.time() - nsc_st)
			clf_durations["SD-NSC"].append(nsc_duration)
			logging.info(f"Run-time SD-NSC: {nsc_duration} seconds")
		else:
			f1_dict["NSC"].append(nsc_f1_score)
			cv_dict["NSC"].append(nsc_cross_val)

			nsc_duration = float(time.time() - nsc_st)
			clf_durations["NSC"].append(nsc_duration)
			logging.info(f"Run-time NSC: {nsc_duration} seconds")


		# Hyperparameter optimization #
		
		if args.use_tuning:
			tnsc_st = time.time()
			if args.vectorization_method == "zcos":
				tnsc_parameters = {"metric": ['cosine'],
							       "shrink_threshold": [None]}
			else:
				tnsc_parameters = {"metric": ['euclidean', 'manhattan'],
							       "shrink_threshold": [None]}

			# metric: 'euclidean' and 'manhattan' only avaible metrics for sparse input
			# shrink_threshold: is not supported for sparse input

			tnsc_grid = GridSearchCV(nsc_clf, tnsc_parameters, cv=cv, scoring="f1_micro")
			tnsc_grid.fit(X_train, y_train)

			tnsc_clf = NearestCentroid(**tnsc_grid.best_params_)
			tnsc_model = tnsc_clf.fit(X_train, y_train)
			tnsc_y_pred = tnsc_model.predict(X_test)
			tnsc_f1_score = f1_score(y_test, tnsc_y_pred, average="micro")
			tnsc_cross_val = np.mean(cross_val_score(tnsc_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			if args.vectorization_method == "zscore":
				f1_dict["D-tNSC"].append(tnsc_f1_score)
				cv_dict["D-tNSC"].append(tnsc_cross_val)
				logging.debug(f"D-tNSC best params: {tnsc_grid.best_params_}")

				tnsc_duration = float(time.time() - tnsc_st)
				clf_durations["D-tNSC"].append(tnsc_duration)
				logging.info(f"Run-time D-tNSC: {tnsc_duration} seconds")
			elif args.vectorization_method == "zcos":
				f1_dict["SD-tNSC"].append(tnsc_f1_score)
				cv_dict["SD-tNSC"].append(tnsc_cross_val)
				logging.debug(f"SD-tNSC best params: {tnsc_grid.best_params_}")

				tnsc_duration = float(time.time() - tnsc_st)
				clf_durations["SD-tNSC"].append(tnsc_duration)
				logging.info(f"Run-time SD-tNSC: {tnsc_duration} seconds")
			else:
				f1_dict["tNSC"].append(tnsc_f1_score)
				cv_dict["tNSC"].append(tnsc_cross_val)
				logging.debug(f"tNSC best params: {tnsc_grid.best_params_}")

				tnsc_duration = float(time.time() - tnsc_st)
				clf_durations["tNSC"].append(tnsc_duration)
				logging.info(f"Run-time tNSC: {tnsc_duration} seconds")


		# ================================
		# Linear Support Vector Machines #
		# ================================
		
		lsvm_st = time.time()

		lsvm_clf = LinearSVC()
		lsvm_model = lsvm_clf.fit(X_train, y_train)
		lsvm_y_pred = lsvm_model.predict(X_test)
		lsvm_f1_score = f1_score(y_test, lsvm_y_pred, average="micro")
		lsvm_cross_val = np.mean(cross_val_score(lsvm_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
		f1_dict["LSVM"].append(lsvm_f1_score)
		cv_dict["LSVM"].append(lsvm_cross_val)

		lsvm_duration = float(time.time() - lsvm_st)
		clf_durations["LSVM"].append(lsvm_duration)
		logging.info(f"Run-time LSVM: {lsvm_duration} seconds")
		

		# Hyperparameter optimization #
		
		if args.use_tuning:

			tlsvm_st = time.time()

			tlsvm_parameters = {"penalty": ["l2"],
							   "loss": ["squared_hinge"],
							   "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
							   "max_iter": [1000, 1125, 1200, 1500, 2000]}
			
			# penalty = "l1" doesn't work with "squared_hinge" so it won't be used
			# loss = "hinge" doesn't work with "l2" so it won't be used
			tlsvm_grid = GridSearchCV(lsvm_clf, tlsvm_parameters, cv=cv, scoring="f1_micro")
			tlsvm_grid.fit(X_train, y_train)

			tlsvm_clf = LinearSVC(**tlsvm_grid.best_params_)
			tlsvm_model = tlsvm_clf.fit(X_train, y_train)
			tlsvm_y_pred = tlsvm_model.predict(X_test)
			tlsvm_f1_score = f1_score(y_test, tlsvm_y_pred, average="micro")
			tlsvm_cross_val = np.mean(cross_val_score(tlsvm_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			f1_dict["tLSVM"].append(tlsvm_f1_score)
			cv_dict["tLSVM"].append(tlsvm_cross_val)
			logging.debug(f"tLSVM best params: {tlsvm_grid.best_params_}")

			tlsvm_duration = float(time.time() - tlsvm_st)
			clf_durations["tLSVM"].append(tlsvm_duration)
			logging.info(f"Run-time tLSVM: {tlsvm_duration} seconds")

		"""
		# ============== #
		# Random Forests #
		# ============== #
		
		rf_st = time.time()

		rf_clf = RandomForestClassifier()
		rf_model = rf_clf.fit(X_train, y_train)
		rf_y_pred = rf_model.predict(X_test)
		rf_f1_score = f1_score(y_test, rf_y_pred, average="micro")
		rf_cross_val = np.mean(cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
		f1_dict["RF"].append(rf_f1_score)
		cv_dict["RF"].append(rf_cross_val)
		
		rf_duration = float(time.time() - rf_st)
		clf_durations["RF"].append(rf_duration)
		logging.info(f"Run-time RF: {rf_duration} seconds")

		# Hyperparameter optimization #
		
		if args.use_tuning:
	
			trf_st = time.time()

			trf_parameters = {"n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
							  "max_features": ["auto", "sqrt"],
							  "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
							  "min_samples_split": [2, 5, 10],
							  "min_samples_leaf": [1, 2, 3, 4],
							  "bootstrap": [True, False],
							  "n_jobs": [n_jobs]}

			# for representaion and computation in reasonable time
			trf_parameters = {"n_estimators": [100, 200, 300, 400, 500],
							  "max_depth": [10, 20, 30, 40],
							  "min_samples_split": [2, 5, 10],
							  "min_samples_leaf": [1, 2, 3, 4],
							  "n_jobs": [n_jobs]}

			trf_grid = GridSearchCV(rf_clf, trf_parameters, cv=cv, scoring="f1_micro")
			trf_grid.fit(X_train, y_train)


			trf_clf = RandomForestClassifier(**trf_grid.best_params_)
			trf_model = trf_clf.fit(X_train, y_train)
			trf_y_pred = trf_model.predict(X_test)
			trf_f1_score = f1_score(y_test, trf_y_pred, average="micro")
			trf_cross_val = np.mean(cross_val_score(trf_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			f1_dict["tRF"].append(trf_f1_score)
			cv_dict["tRF"].append(trf_cross_val)
			logging.debug(f"tRF best params: {trf_grid.best_params_}")

			trf_duration = float(time.time() - trf_st)
			clf_durations["tRF"].append(trf_duration)
			logging.info(f"Run-time tRF: {trf_duration} seconds")



		


		# =================================================
		# Multinomial Naive Bayes (only without z-scores) #
		# =================================================
		
		if args.vectorization_method != "zscore":

			mnb_st = time.time()

			mnb_clf = MultinomialNB()
			mnb_model = mnb_clf.fit(X_train, y_train)
			mnb_y_pred = mnb_model.predict(X_test)
			mnb_f1_score = f1_score(y_test, mnb_y_pred, average="micro")
			mnb_cross_val = np.mean(cross_val_score(mnb_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			f1_dict["MNB"].append(mnb_f1_score)
			cv_dict["MNB"].append(mnb_cross_val)

			mnb_duration = float(time.time() - mnb_st)
			clf_durations["MNB"].append(mnb_duration)
			logging.info(f"Run-time MNB: {mnb_duration} seconds")
		
			# Hyperparameter optimization #

			if args.use_tuning:

				tmnb_st = time.time()

				tmnb_parameters = {"alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
											 0.1, 0.5, 1.0, 2.0, 3.0],
								   "fit_prior": [True, False]}

				tmnb_grid = GridSearchCV(mnb_clf, tmnb_parameters, cv=cv, scoring="f1_micro")
				tmnb_grid.fit(X_train, y_train)

				tmnb_clf = MultinomialNB(**tmnb_grid.best_params_)
				tmnb_model = tmnb_clf.fit(X_train, y_train)
				tmnb_y_pred = tmnb_model.predict(X_test)
				tmnb_f1_score = f1_score(y_test, tmnb_y_pred, average="micro")
				tmnb_cross_val = np.mean(cross_val_score(tmnb_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
				f1_dict["tMNB"].append(tmnb_f1_score)
				cv_dict["tMNB"].append(tmnb_cross_val)
				logging.debug(f"tMNB best params: {tmnb_grid.best_params_}")

				tmnb_duration = float(time.time() - tmnb_st)
				clf_durations["tMNB"].append(tmnb_duration)
				logging.info(f"Run-time tMNB: {tmnb_duration} seconds")

		
		
		# ================================
		# Linear Support Vector Machines #
		# ================================
		
		lsvm_st = time.time()

		lsvm_clf = LinearSVC()
		lsvm_model = lsvm_clf.fit(X_train, y_train)
		lsvm_y_pred = lsvm_model.predict(X_test)
		lsvm_f1_score = f1_score(y_test, lsvm_y_pred, average="micro")
		lsvm_cross_val = np.mean(cross_val_score(lsvm_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
		f1_dict["LSVM"].append(lsvm_f1_score)
		cv_dict["LSVM"].append(lsvm_cross_val)

		lsvm_duration = float(time.time() - lsvm_st)
		clf_durations["LSVM"].append(lsvm_duration)
		logging.info(f"Run-time LSVM: {lsvm_duration} seconds")
		

		# Hyperparameter optimization #
		
		if args.use_tuning:

			tlsvm_st = time.time()

			tlsvm_parameters = {"penalty": ["l2"],
							   "loss": ["squared_hinge"],
							   "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
							   "max_iter": [1000, 1125, 1200, 1500, 2000]}
			
			# penalty = "l1" doesn't work with "squared_hinge" so it won't be used
			# loss = "hinge" doesn't work with "l2" so it won't be used
			tlsvm_grid = GridSearchCV(lsvm_clf, tlsvm_parameters, cv=cv, scoring="f1_micro")
			tlsvm_grid.fit(X_train, y_train)

			tlsvm_clf = LinearSVC(**tlsvm_grid.best_params_)
			tlsvm_model = tlsvm_clf.fit(X_train, y_train)
			tlsvm_y_pred = tlsvm_model.predict(X_test)
			tlsvm_f1_score = f1_score(y_test, tlsvm_y_pred, average="micro")
			tlsvm_cross_val = np.mean(cross_val_score(tlsvm_clf, X_train, y_train, cv=cv, scoring="f1_micro"))
			f1_dict["tLSVM"].append(tlsvm_f1_score)
			cv_dict["tLSVM"].append(tlsvm_cross_val)
			logging.debug(f"tLSVM best params: {tlsvm_grid.best_params_}")

			tlsvm_duration = float(time.time() - tlsvm_st)
			clf_durations["tLSVM"].append(tlsvm_duration)
			logging.info(f"Run-time tLSVM: {tlsvm_duration} seconds")
		
		# =====================
		# Logistic Regression #
		# =====================
		
		lr_st = time.time()

		# every solver except "liblinear" had problems with the convergence
		lr_clf = LogisticRegression(multi_class="ovr", solver="liblinear")
		lr_model = lr_clf.fit(X_train, y_train)
		lr_y_pred = lr_model.predict(X_test)
		lr_f1_score = f1_score(y_test, lr_y_pred, average="micro")
		lr_cross_val = np.mean(cross_val_score(lr_clf, X_train, y_train, cv=cv))
		f1_dict["LR"].append(lr_f1_score)
		cv_dict["LR"].append(lr_cross_val)

		lr_duration = float(time.time() - lr_st)
		clf_durations["LR"].append(lr_duration)
		logging.info(f"Run-time LR: {lr_duration} seconds")

		# Hyperparameter optimization #
		
		if args.use_tuning:

			tlr_st = time.time()

			tlr_parameters = {"penalty": ["l1"],
							  "tol": [0.0001, 0.001, 0.01, 0.1],
							  "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
							  "solver": ["liblinear"],
							  "max_iter": [1000, 2000, 3000, 4000, 5000, 6000],
							  "multi_class": ["ovr"],
							  "n_jobs": [n_jobs]}

			# solver only "saga" because its good for large datasets and it supports elasticnet-penalty
			tlr_grid = GridSearchCV(lr_clf, tlr_parameters, cv=cv, scoring="f1_micro")
			tlr_grid.fit(X_train, y_train)

			tlr_clf = LogisticRegression(**tlr_grid.best_params_)
			tlr_model = tlr_clf.fit(X_train, y_train)
			tlr_y_pred = tlr_model.predict(X_test)
			tlr_f1_score = f1_score(y_test, tlr_y_pred, average="micro")
			tlr_cross_val = np.mean(cross_val_score(tlr_clf, X_train, y_train, cv=cv))
			f1_dict["tLR"].append(tlr_f1_score)
			cv_dict["tLR"].append(tlr_cross_val)
			logging.debug(f"tLR best params: {tlr_grid.best_params_}")

			tlr_duration = float(time.time() - tlr_st)
			clf_durations["tLR"].append(tlr_duration)
			logging.info(f"Run-time tLR: {tlr_duration} seconds")
		
		"""
	
	# ================
	# Saving results #
	# ================


	final_f1_dict = {}
	for (f1_k, f1_v), (cv_k, cv_v) in zip(f1_dict.items(), cv_dict.items()):
		if f1_k == cv_k:
			final_f1_dict[f1_k] = {"f1": np.mean(f1_v), "cv": np.mean(cv_v)}
	
	results = pd.DataFrame(final_f1_dict).round(3).T

	
	if args.save_date:
		csv_name = f"extended_classification_{args.corpus_name}({args.vectorization_method}_{classruns}_{args.max_features}_{ngrams})_({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
	else:
		csv_name = f"extended_classification_{args.corpus_name}({args.vectorization_method}_{classruns}_{args.max_features}_{ngrams})"
	results.to_csv(f"../data/tables/{csv_name}.csv")

	if args.visualization:
		logging.info(r"Saving figure of results to data/figures/results/")
		visualize(results,
			  	  "bar_vertical",
			  	  classruns,
			  	  max_features=args.max_features,
			  	  cross_validation=cv,
			  	  ngram=ngrams,
			  	  output_name=args.corpus_name, 
			  	  save_date=args.save_date,
			  	  vectorization_method=args.vectorization_method)


	# =================================
	# Saving classification durations #
	# =================================

	mean_clf_durations = {}

	for k, v in clf_durations.items():
		mean_clf_durations[k] = np.mean(v)

	clf_durations_df = pd.DataFrame(mean_clf_durations.items(), columns=["clf", "durations"])


	if args.save_date:
		duration_name = f"extended_clf_durations_{args.corpus_name}({args.vectorization_method}_{classruns}_{args.max_features}_{ngrams})_({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
	else:
		duration_name = f"extended_clf_durations_{args.corpus_name}({args.vectorization_method}_{classruns}_{args.max_features}_{ngrams}"
	clf_durations_df.to_csv(f"../data/tables/{duration_name}.csv")

	
	if args.visualization:
		logging.info(r"Saving figure of classification durations to data/figures/durations/")
		visualize(clf_durations_df,
			  	  "pie",
			  	  max_features=args.max_features,
			  	  classruns=classruns,
			  	  cross_validation=cv,
			  	  ngram=ngrams, 
			  	  output_name=args.corpus_name,
			  	  save_date=args.save_date,
			  	  vectorization_method=args.vectorization_method)

	program_duration = float(time.time() - program_st)
	logging.info(f"Run-time: {int(program_duration)/60} minute(s).")

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="extended_classification", description="Comparison of different classifier in an extended version.")
	parser.add_argument("path", type=str, help="Path to the corpus as csv-file.")
	parser.add_argument("--classruns", "-cr", type=int, nargs="?", default=10, help="Sets the number of classification runs.")
	parser.add_argument("--corpus_name", "-cn", type=str, nargs="?", default="prose", help="Indicates the name of the corpus for the output file.")
	parser.add_argument("--max_features", "-mf", type=int, default=2000, help="Indicates the number of most frequent words.")
	parser.add_argument("--ngram", "-ng", type=int, nargs="*", action="store", default=(1,1), help="Passes the ngram range.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--use_tuning", "-ut", action="store_true", help="Indicates if hyperparameter optimization should be used.")
	parser.add_argument("--vectorization_method", "-vm", type=str, default="bow", help="Indicates the vectorization method. Default is 'bow'. Other possible values are 'zscore', 'tfidf' and 'cos'.")
	parser.add_argument("--visualization", "-v", action="store_true", help="Indicates if results should be visualized.")
	
	args = parser.parse_args()

	main()