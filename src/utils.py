import os
import sys
import cv2
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def process_image(img_dir,image_id):
    files = os.listdir(img_dir)
    # files.sort(key=lambda x: int(x.split('.')[0]))
    ls=[]
    for f in files:
        if f.endswith('.jpg'):
            ls.append(f.strip('.jpg'))
        elif f.endswith('.png'):
            ls.append(f.strip('.png'))
        else:
            ls.append(f.strip('.jpeg'))

    img_list = []

    for f in ls:
        if f==str(image_id):
            file = f + '.jpg'
            img = cv2.imread(os.path.join(img_dir, file))
            img = cv2.resize(img,(224,224))
            img_list.append(img)
        
    img_features = np.asarray(img_list).astype(np.float32)
    img_features = img_features/255.0

    return img_features