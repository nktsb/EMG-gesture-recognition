"""
 -----------------------------------------------------------------------------
 Project       : EMG-gesture-recognition
 File          : emg_ml.py
 Author        : nktsb
 Created on    : 08.05.2024
 GitHub        : https://github.com/nktsb/EMG-gesture-recognition
 -----------------------------------------------------------------------------
 Copyright (c) 2024 nktsb
 All rights reserved.
 -----------------------------------------------------------------------------
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

from emg_python.emg_files import EMG_file, get_date_time
from emg_python.emg_dataset import get_sequential_fpath
from emg_python.emg_preprocessing import EMG_preprocessor
from emg_python.emg_features import EMG_feature_extractor

import pandas as pd

extracted_files_name_base = "extracted_features_"
estimated_files_name_base = "estimated_features_"

def get_train_and_test(extracted_file: EMG_file, selected_users: list, features_map: list, test_part=0.2):
    file_data = extracted_file.read_data('features_values')

    X = list()
    y = list()

    all_features_map = extracted_file.read_data('features_map')

    features_map_idx = [i for i, feature in enumerate(all_features_map) if feature in features_map]

    for gest in file_data:
        for user in file_data[gest]:
            if user in selected_users:
                for gest_file in file_data[gest][user]:
                    X.append([gest_file[i] for i in features_map_idx])
                    y.append(gest)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_part, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


class EMG_ds_handler():
    def __init__(self, ds_folder: str, features_folder: str,
                emg_preprocessor: EMG_preprocessor, \
                accel_preprocessor: EMG_preprocessor, \
                gyro_preprocessor: EMG_preprocessor, \
                emg_feature_extractor: EMG_feature_extractor, \
                accel_feature_extractor: EMG_feature_extractor, \
                gyro_feature_extractor: EMG_feature_extractor):

        if not os.path.isdir(ds_folder):
            print("Wrong dataset folder!")
            return False

        self.dataset_folder = ds_folder
        self.features_folder = features_folder

        self.sns_info = None
        self.gestures_files = dict()

        if not self.get_all_gestures_data(self):
            self.sns_info = None
            self.gestures_files = None
            return False
        
        self.all_users_list = list()
        self.all_gestures_list = list(self.gestures_files.keys())
        for gesture in self.gestures_files:
            users = list(self.gestures_files[gesture].keys())
            for user in users:
                if not user in self.all_users_list:
                    self.all_users_list.append(user)
        self.all_users_list.sort()
        self.all_gestures_list.sort()

        self.emg_preprocessor = emg_preprocessor
        self.emg_feature_extractor = emg_feature_extractor
        self.accel_preprocessor = accel_preprocessor
        self.accel_feature_extractor = accel_feature_extractor
        self.gyro_preprocessor = gyro_preprocessor
        self.gyro_feature_extractor = gyro_feature_extractor

        self.features_values = dict()
        for gesture in self.gestures_files:
            self.features_values[gesture] = dict()
            for user in self.gestures_files[gesture]:
                self.features_values[gesture][user] = list()

        self.preprocessor_info = {
            "emg_normalize_method": self.emg_preprocessor.normalize_method,
            "accel_normalize_method": self.accel_preprocessor.normalize_method,
            "gyro_normalize_method": self.gyro_preprocessor.normalize_method,
            "emg_preprocessor_info": self.emg_preprocessor.preprocessor_info,
            "accel_preprocessor_info": self.accel_preprocessor.preprocessor_info,
            "gyro_preprocessor_info": self.gyro_preprocessor.preprocessor_info
        }
        self.features_info = {
            "emg_features_info": self.emg_feature_extractor.features_info,
            "accel_features_info": self.emg_feature_extractor.features_info,
            "gyro_features_info": self.emg_feature_extractor.features_info,
        }

    @staticmethod
    def get_all_gestures_data(self):
        dataset_files = list()
        for root, dirs, files in os.walk(self.dataset_folder):
            for file in files:
                if file.endswith(".json") and not file.startswith("test"):
                    dataset_files.append(os.path.join(root, file))
        dataset_files.sort()
    
        for fpath in dataset_files:
            current_file = EMG_file(fpath)

            file_general_info = current_file.read_data('general_info')

            file_user = file_general_info['user_name']
            file_gesture = file_general_info['gesture_name']
            file_sns_info = current_file.read_data('sns_info')

            if not self.sns_info:
                self.sns_info = file_sns_info
            elif file_sns_info != file_sns_info:
                print(f"Error: DS file with wrong parameters: {fpath}")
                del current_file
                return False

            if not file_gesture in self.gestures_files:
                self.gestures_files[file_gesture] = dict()
            if not file_user in self.gestures_files[file_gesture]:
                self.gestures_files[file_gesture][file_user] = list()
            
            self.gestures_files[file_gesture][file_user].append(fpath)
            self.gestures_files[file_gesture][file_user].sort()
            del current_file

        return True

    def get_raw_data_from_file(self, fpath):
        file = EMG_file(fpath)
        sns_data = file.read_data('sns_data')
        sns_info = file.read_data('sns_info')
        file_emg_data = sns_data['emg_data']
        file_accel_data = sns_data['accel_data']
        file_gyro_data = sns_data['gyro_data']
        file_sample_rate = sns_info['sample_rate']
        del file
        return file_emg_data, file_accel_data, file_gyro_data, file_sample_rate

    @staticmethod
    def get_features_map(feature_extractor: EMG_feature_extractor):
        feat_map = list()
        for ch in feature_extractor.features_vals:
            for feature in feature_extractor.features_vals[ch]:
                feat_map.append({'channel': ch, 'feature': feature})
        return feat_map

    @staticmethod
    def get_features_vals(feature_extractor: EMG_feature_extractor):
        values_list = list()
        for ch in feature_extractor.features_vals:
            for feature in feature_extractor.features_vals[ch]:
                value = feature_extractor.features_vals[ch][feature]
                values_list.append(value)
        return values_list

    def check_if_exist(self):
        features_files = list()
        if not os.path.isdir(self.features_folder):
            print(f"There's no feature dir: '{self.features_folder}' creating...")
            os.makedirs(self.features_folder)
        for file in os.listdir(self.features_folder):
            if file.endswith(".json") and file.startswith(extracted_files_name_base):
                features_files.append(file)
        features_files.sort()
        if features_files:
            print(f"Found features files:")
            for file in features_files:
                print(f"{self.features_folder}/{file}")
            print("")

        selected_file = None

        for features_file in features_files:
            current_file = EMG_file(f"{self.features_folder}/{features_file}")
            general_info = current_file.read_data('general_info')
            sns_info = current_file.read_data('sns_info')
            preprocessor_info = current_file.read_data('preprocessor_info')
            features_info = current_file.read_data('features_info')
            del current_file
            if general_info['user_name'] != self.all_users_list:
                continue
            if general_info['gesture_name'] != self.all_gestures_list:
                continue
            if sns_info != self.sns_info:
                continue
            if preprocessor_info != self.preprocessor_info:
                continue
            if features_info != self.features_info:
                continue
            selected_file = f"{self.features_folder}/{features_file}"
            print(f"Features extracted file for current config: {selected_file}\r\n")
            break

        return selected_file

    def get_raw_X_y(self):
        X = list()
        y = list()
        for gesture in self.gestures_files:
            for user in self.gestures_files[gesture]:
                for fpath in self.gestures_files[gesture][user]:
                    file_emg_data, file_accel_data, file_gyro_data, sample_rate = \
                        self.get_raw_data_from_file(fpath)

                    self.emg_preprocessor.set_sample_rate(sample_rate)
                    self.accel_preprocessor.set_sample_rate(sample_rate)
                    self.gyro_preprocessor.set_sample_rate(sample_rate)

                    self.emg_preprocessor.set_data_ptr(file_emg_data)
                    self.accel_preprocessor.set_data_ptr(file_accel_data)
                    self.gyro_preprocessor.set_data_ptr(file_gyro_data)

                    self.emg_preprocessor.preprocess_data()
                    self.accel_preprocessor.preprocess_data()
                    self.gyro_preprocessor.preprocess_data()

                    file_X = list(self.emg_preprocessor.res_data.values()) + \
                    list(self.accel_preprocessor.res_data.values()) + \
                    list(self.gyro_preprocessor.res_data.values())

                    file_X = np.transpose(file_X)

                    X.append(file_X)
                    y.append(gesture)
        X = np.array(X)
        return X, y

    def handle(self):
        file = self.check_if_exist()
        if file:
            return file
        print("Extracting features...")
        for gesture in self.gestures_files:
            for user in self.gestures_files[gesture]:
                for fpath in self.gestures_files[gesture][user]:
                    file_emg_data, file_accel_data, file_gyro_data, sample_rate = \
                        self.get_raw_data_from_file(fpath)

                    self.emg_preprocessor.set_sample_rate(sample_rate)
                    self.accel_preprocessor.set_sample_rate(sample_rate)
                    self.gyro_preprocessor.set_sample_rate(sample_rate)
                    self.emg_feature_extractor.set_sample_rate(sample_rate)
                    self.accel_feature_extractor.set_sample_rate(sample_rate)
                    self.gyro_feature_extractor.set_sample_rate(sample_rate)

                    self.emg_preprocessor.set_data_ptr(file_emg_data)
                    self.accel_preprocessor.set_data_ptr(file_accel_data)
                    self.gyro_preprocessor.set_data_ptr(file_gyro_data)
                    self.emg_feature_extractor.set_data_ptr_and_features(self.emg_preprocessor.res_data, is_all_features=True)
                    self.accel_feature_extractor.set_data_ptr_and_features(self.accel_preprocessor.res_data, is_all_features=True)
                    self.gyro_feature_extractor.set_data_ptr_and_features(self.gyro_preprocessor.res_data, is_all_features=True)

                    self.emg_preprocessor.preprocess_data()
                    self.accel_preprocessor.preprocess_data()
                    self.gyro_preprocessor.preprocess_data()
                    self.emg_feature_extractor.extract_features()
                    self.accel_feature_extractor.extract_features()
                    self.gyro_feature_extractor.extract_features()

                    file_feat_values = list()
                    file_feat_values += self.get_features_vals(self.emg_feature_extractor)
                    file_feat_values += self.get_features_vals(self.accel_feature_extractor)
                    file_feat_values += self.get_features_vals(self.gyro_feature_extractor)
                    self.features_values[gesture][user].append(file_feat_values)
            
        self.all_features_map = self.get_features_map(self.emg_feature_extractor)
        self.all_features_map += self.get_features_map(self.accel_feature_extractor)
        self.all_features_map += self.get_features_map(self.gyro_feature_extractor)

        return self.save_in_file(self)

    @staticmethod
    def save_in_file(self):

        fpath, _ = get_sequential_fpath(self.features_folder, extracted_files_name_base, ".json")
        self.feature_data_file = EMG_file(fpath, user_name=self.all_users_list, \
                                          gesture_name=self.all_gestures_list)
        self.feature_data_file.write_data('sns_info', self.sns_info)


        self.feature_data_file.write_data('preprocessor_info', self.preprocessor_info)
        self.feature_data_file.write_data('features_info', self.features_info)
        self.feature_data_file.write_data('features_map', self.all_features_map)
        self.feature_data_file.write_data('features_values', self.features_values)

        print(f"Features values saved in file {self.feature_data_file.fpath}")
        return self.feature_data_file.fpath


class EMG_feature_estimator():
    def __init__(self, folder: str, extracted_file: EMG_file, ml_info: dict):
        self.features_folder = folder
        self.extracted_file = extracted_file
        self.extracted_fpath = self.extracted_file.fpath
        if not self.extracted_file.is_exist:
            print("Wrong extracted features file!")
            return False
        self.all_features_map = self.extracted_file.read_data('features_map')
        self.selected_users = ml_info['users']

    @staticmethod
    def check_if_exist(self):

        features_estimated_files = self.extracted_file.read_data("estimated_files")

        if features_estimated_files:
            for file in features_estimated_files:
                current_file = EMG_file(file)
                if not current_file.is_exist \
                        or current_file.read_data("original_extracted_file") != self.extracted_fpath:
                    features_estimated_files.remove(file)
                    self.extracted_file.write_data("estimated_files", features_estimated_files)
                    print("Error: estimated files removed or changed!")
                    del current_file
                    return False

                file_general_info = current_file.read_data("general_info")
                file_fe_method = current_file.read_data("features_estimation_method")
                file_orig_extracted = current_file.read_data("original_extracted_file")
                del current_file

                if file_general_info["user_name"] == self.selected_users \
                        and file_orig_extracted == self.extracted_fpath:
                    print(f"\r\nFeatures estimated file for current config: {file}\r\n")
                    return file

        return False

    def get_sorted_estimation_data(self, correlation, importance):
        features_estimation_data = list()
        for i, feature in enumerate(self.all_features_map):
            feature_key_mi = {
                "feature_map_key": feature,
                "importance": importance[i],
                "correlation": correlation[i],
                "common": (correlation[i] * importance[i])
            }
            features_estimation_data.append(feature_key_mi)
        
        sorted_list = sorted(features_estimation_data, key=lambda x: x["common"], reverse=True)
    
        return sorted_list

    def estimate_features(self):

        file = self.check_if_exist(self)
        if file:
            return file

        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_file, \
                                                              self.selected_users, self.all_features_map, 0.2)

        corr = self.get_corr(X_train, y_train)
        importance = self.mutual_info(X_train, y_train)

        features_estimation_data = self.get_sorted_estimation_data(corr, importance)

        return self.save_in_file(self, features_estimation_data)

    @staticmethod
    def save_in_file(self, features_estimation_data):
        estimated_fpath, _ = get_sequential_fpath(self.features_folder, estimated_files_name_base, ".json")
        estimated_file = EMG_file(estimated_fpath, user_name=self.selected_users, \
                                  gesture_name="Check original extracted file")

        estimated_file.write_data('original_extracted_file', self.extracted_fpath)
        estimated_file.write_data('features_estimation_method', "MI+Correlation")
        estimated_file.write_data('features_estimation_data', features_estimation_data)
        estimated_files_list = self.extracted_file.read_data("estimated_files")
        if not estimated_files_list:
            estimated_files_list = list()
        estimated_files_list.append(estimated_fpath)
        self.extracted_file.write_data("estimated_files", estimated_files_list)
        return estimated_fpath

    def get_corr(self, X, y):
        df = pd.DataFrame(X, columns=[f"Feature{i}" for i in range(len(X[0]))])
        df['Class'] = y
        correlation_matrix = df.drop('Class', axis=1).corr()

        corr_pairs = correlation_matrix.abs().unstack()
        corr_pairs = corr_pairs[corr_pairs != 1]
        high_corr_pairs = corr_pairs[corr_pairs > 0.95].drop_duplicates().index

        features_to_remove = set()
        for pair in high_corr_pairs:
            features_to_remove.add(pair[0])
        df_reduced = df.drop(columns=list(features_to_remove))

        remaining_features = df_reduced.drop('Class', axis=1).columns.tolist()
        remaining_features_idx = [df.columns.get_loc(feature) for feature in remaining_features]

        res = list()
        for i, _ in enumerate(self.all_features_map):
            res.append(1 if i in remaining_features_idx else 0)

        return res

    def mutual_info(self, X , y):

        mutual_info = mutual_info_classif(X, y)

        features_importance = mutual_info / np.max(mutual_info) # normalization

        return features_importance


class EMG_feature_selector():
    def __init__(self, fpath):
        self.estimated_fpath = fpath
    
    def select_features(self, importance_threshold: float=0.0, features_num: int=None):
        estimated_features_file = EMG_file(self.estimated_fpath)
        features_estimation_data = estimated_features_file.read_data("features_estimation_data")
        selected_features_map = list()
        for i, feature_est_data in enumerate(features_estimation_data):

            feat_map_key = feature_est_data["feature_map_key"]
            importance = feature_est_data["common"]
            if features_num and i >= features_num:
                break
            if importance <= importance_threshold:
                break
            selected_features_map.append(feat_map_key)
        return selected_features_map

class EMG_model_trainer():
    def __init__(self, ds_folder: str, features_folder: str,
                 emg_preprocessor: EMG_preprocessor, \
                 accel_preprocessor: EMG_preprocessor, \
                 gyro_preprocessor: EMG_preprocessor, \
                 emg_feature_extractor: EMG_feature_extractor, \
                 accel_feature_extractor: EMG_feature_extractor, \
                 gyro_feature_extractor: EMG_feature_extractor, \
                 ml_info: dict, models_folder: str):

        self.ds_folder = ds_folder
        self.models_folder = models_folder
        self.features_folder = features_folder
        self.ds_handler = EMG_ds_handler(self.ds_folder, features_folder, emg_preprocessor, \
                                    accel_preprocessor, gyro_preprocessor, \
                                    emg_feature_extractor, accel_feature_extractor, \
                                    gyro_feature_extractor)
        self.ml_info = ml_info

        self.all_models_list = {
            "KNN": self.knn, # K nearest neighbours
            "SVM": self.svm, # Support vector machine
            "LDA": self.lda, # Linear disctiminant analysis
            "DT": self.dt, # Decision tree
            "GB": self.gb, # Gradient boosting
            "RF": self.rf, # Random forest
            "NB": self.nb, # Naive Bayes
        }
        self.knn_num = ml_info["knn_features_num"]
        self.svm_num = ml_info["svm_features_num"]
        self.selected_users = ml_info["users"]
        self.lda_threshold = ml_info["lda_importance_threshold"]
        self.dt_threshold = ml_info["dt_importance_threshold"]
        self.gb_threshold = ml_info["gb_importance_threshold"]
        self.rf_threshold = ml_info["rf_importance_threshold"]

    def check_if_exist(self, model_name):
        current_file = EMG_file(self.estimated_file)
        models_info = current_file.read_data("models")
        if models_info:
            for model in models_info:
                file_model_name = model["model_name"]
                if model_name == file_model_name:
                    file_model_fpath = model["model_file"]
                    file_info_model_fpath = model["model_info_file"]
                    file_res_model_fpath = model["model_res_file"]
                    if bool(os.path.isfile(file_model_fpath) and os.path.getsize(file_model_fpath)) \
                            and bool(os.path.isfile(file_info_model_fpath) and os.path.getsize(file_info_model_fpath)) \
                            and bool(os.path.isfile(file_res_model_fpath) and os.path.getsize(file_res_model_fpath)):
                        print(f"{model_name} model already trained: {file_model_fpath}\r\nInfo file: {model["model_info_file"]}")
                        return True
                    else:
                        print(f"Error: model files removed")
                        models_info.remove(model)
                        try:
                            os.remove(file_model_fpath)
                            os.remove(file_info_model_fpath)
                            os.remove(file_res_model_fpath)
                        except:
                            pass
                        current_file.write_data("models", models_info)
                        return False
        return False

    def train_models(self):

        extracted_fname = self.ds_handler.handle()
        self.extracted_features_file = EMG_file(extracted_fname)

        features_estimator = EMG_feature_estimator(self.features_folder, self.extracted_features_file, self.ml_info)
        self.estimated_file = features_estimator.estimate_features()
        self.feature_selector = EMG_feature_selector(self.estimated_file)
        for model_name in self.all_models_list:
            if not self.check_if_exist(model_name):
                self.all_models_list[model_name]()
    
    def knn(self):
        selected_feature_map = self.feature_selector.select_features(features_num=self.knn_num)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = KNeighborsClassifier(n_neighbors=5)
        score, report, matrix = self.model_fit_predict(model, "KNN", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "KNN", score, report, matrix)

    def svm(self):
        selected_feature_map = self.feature_selector.select_features(features_num=self.svm_num)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = SVC(kernel='linear', C = 1.0)
        score, report, matrix = self.model_fit_predict(model, "SVM", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "SVM", score, report, matrix)

    def lda(self):
        selected_feature_map = self.feature_selector.select_features(importance_threshold=self.lda_threshold)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = LinearDiscriminantAnalysis()
        score, report, matrix = self.model_fit_predict(model, "LDA", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "LDA", score, report, matrix)

    def dt(self):
        selected_feature_map = self.feature_selector.select_features(importance_threshold=self.dt_threshold)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = DecisionTreeClassifier()
        score, report, matrix = self.model_fit_predict(model, "DT", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "DT", score, report, matrix)

    def gb(self):
        selected_feature_map = self.feature_selector.select_features(importance_threshold=self.gb_threshold)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        score, report, matrix = self.model_fit_predict(model, "GB", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "GB", score, report, matrix)

    def rf(self):
        selected_feature_map = self.feature_selector.select_features(importance_threshold=self.rf_threshold)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = RandomForestClassifier(n_estimators=100)
        score, report, matrix = self.model_fit_predict(model, "RF", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "RF", score, report, matrix)

    def nb(self):
        selected_feature_map = self.feature_selector.select_features(importance_threshold=self.rf_threshold)
        X_train, X_test, y_train, y_test = get_train_and_test(self.extracted_features_file, \
                                                              self.selected_users, selected_feature_map, 0.2)

        model = GaussianNB()
        score, report, matrix = self.model_fit_predict(model, "NB", X_train, y_train, X_test, y_test)
        self.save_model_in_file(selected_feature_map, model, "NB", score, report, matrix)

    def model_fit_predict(self, model, model_name, X_train, y_train, X_test, y_test):
        print(f"\r\n{get_date_time()}: {model_name} start training...")
        model.fit(X_train, y_train)
        print(f"{get_date_time()}: {model_name} training finished!")
        print(f"{get_date_time()}: {model_name} start testing...")
        y_pred = model.predict(X_test)
        print(f"{get_date_time()}: {model_name} testing finished!")
        score = accuracy_score(y_test, y_pred)
        print(f"{get_date_time()}: {model_name} accuracy score: {score}\r\n")
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # print(f"{get_date_time()}: {model_name} Report: \r\n{report}")
        return score, report, conf_matrix

    def save_model_in_file(self, selected_features_map, model, model_name, score, report, matrix):

        model_fpath, _ = get_sequential_fpath(self.models_folder, f"{model_name}_model_", ".pkl")

        with open(model_fpath, 'wb') as file:
            pickle.dump(model, file)

        current_est_file = EMG_file(self.estimated_file)
        models_list = current_est_file.read_data("models")
        if not models_list:
            models_list = list()

        model_res_file, _ = get_sequential_fpath(self.models_folder, f"{model_name}_model_res_", ".csv")
        model_info_file, _ = get_sequential_fpath(self.models_folder, f"{model_name}_model_info_", ".json")
        
        current_model_info_file = EMG_file(model_info_file, "Check original", "Check original")
        info = {
            'original_file': self.estimated_file,
            'features_map': selected_features_map,
            "model_name": model_name,
            "model_file": model_fpath,
            "model_score": score
        }
        current_model_info_file.write_data('model_info', info)
        del current_model_info_file
    
        report_frame = pd.DataFrame(report).transpose().round(3)
        matrix_frame = pd.DataFrame(matrix).round(3)

        report_frame.to_csv(model_res_file, mode = 'a', index=True)
        matrix_frame.to_csv(model_res_file, mode = 'a', index=True)

        print(f"Info for model {model_name} saved in file: {model_info_file}")
        print(f"Report for model {model_name} saved in file: {model_res_file}\r\n")

        models_list.append({"model_name": model_name, "model_file": model_fpath, \
                            "model_info_file": model_info_file, \
                            "model_res_file": model_res_file})
        current_est_file.write_data("models", models_list)
        del current_est_file

class EMG_classifier():
    def __init__(self, model_info_fpath: str):
        self.model_info_fpath = model_info_fpath
        self.model_info_file = EMG_file(model_info_fpath)
        if not self.model_info_file.is_exist:
            print(f"Error: model {model_info_fpath} doesn't exist")

        self.model_info = self.model_info_file.read_data("model_info")
        self.model_fpath = self.model_info['model_file']
        with open(self.model_fpath , 'rb') as file:
            self.model = pickle.load(file)
    
        self.estimated_fpath = self.model_info['original_file']
        self.estimated_file = EMG_file(self.estimated_fpath)

        self.extracted_fpath = self.estimated_file.read_data("original_extracted_file")
        self.extracted_file = EMG_file(self.extracted_fpath)

    def get_preprocessing_info(self):
        return self.extracted_file.read_data("preprocessor_info")

    def get_features_extraction_info(self):
        return self.extracted_file.read_data("features_info")

    def get_features_map(self):
        return self.model_info['features_map']

    def predict(self, X):
        y_pred = self.model.predict([X])
        proba = self.model.predict_proba([X])

        predicted_class = y_pred[0]
        predicted_idx = list(self.model.classes_).index(predicted_class)
        predicted_proba = proba[0][predicted_idx]
        # print(f"{predicted_class}: {predicted_proba}")
        result = [predicted_class, predicted_proba]

        return result