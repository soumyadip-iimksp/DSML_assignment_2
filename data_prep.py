__author__ = "Soumyadip Majumder"
__version__ = "1.0.0"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Complete"
__date__ = "03 Feb 2023"

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

class DataPreP:
    
    def __init__(self, dv_filename:str):
        self.dv_filename = dv_filename
        

    def create_datasets(self, data_path:str):
        
        self.SAVEPATH = "./models"

        self.df = pd.read_csv(data_path)
        self.X = self.df.drop(columns=["class"], axis=1).astype(np.float64)
        self.y = self.df["class"].values

        self.train_X, self.rem_X, self.train_y, self.rem_y = train_test_split(self.X, self.y, test_size=0.3, shuffle=True, random_state=143)
        self.val_X, self.test_X, self.val_y, self.test_y = train_test_split(self.rem_X, self.rem_y, test_size=0.5, shuffle=True, random_state=143)

        del self.rem_X, self.rem_y

        self.train_X = self.train_X.to_dict(orient="records")
        self.val_X = self.val_X.to_dict(orient="records")
        self.test_X = self.test_X.to_dict(orient="records")

        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(self.train_X)
        self.train_X = self.dv.transform(self.train_X)
        self.val_X = self.dv.transform(self.val_X)
        self.test_X = self.dv.transform(self.test_X)

        if not os.path.exists(self.SAVEPATH):
            os.makedirs(self.SAVEPATH)
        with open(f"{self.SAVEPATH}/{self.dv_filename}.bin", "wb") as f_out:
            pickle.dump(self.dv, f_out)

        return self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y


if __name__ == "__main__":
    dp = DataPreP("dv_logtistic")
    X_train, y_train, X_val, y_val, X_test, y_test = dp.create_datasets("./pima-indians-diabetes.csv")
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_val: ", X_val.shape)
    print("y_val: ", y_val.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)


