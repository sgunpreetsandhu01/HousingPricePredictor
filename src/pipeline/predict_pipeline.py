import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import scipy
import cv2
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xg

from src.utils import load_object, process_image
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = "Models/stacked_model.pkl"
            preprocessing_path = "Models/preprocessor2.pkl"

            final_model = load_object(model_path)
            preprocessor = load_object(preprocessing_path)

            df_input = preprocessor.transform(features)
            final_input = df_input.toarray()

            prediction = final_model.predict(final_input)
            prediction = int(np.squeeze(prediction))

            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 image_id:int,
                 street:int,
                 city:str,
                 bed:int,
                 bath:int,
                 sqft:float
                 ):
        self.image_id = image_id
        self.street = street
        self.city = city
        self.bed = bed
        self.bath =bath
        self.sqft = sqft

    def get_data_as_dataframe(self):
        try:
            custom_data_dict ={
                "image_id":[self.image_id],
                "street":[self.street],
                "citi":[self.city],
                "bed":[self.bed],
                "bath":[self.bath],
                "sqft":[self.sqft]
            }
            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_frame_final(self):
        try:
            #Loading the CNN model
            load_cnn_model = tf.keras.models.load_model("Models/final_model.h5")

            #Getting the initial dataframe for CNN model
            data = CustomData(self.image_id,self.street,self.city,self.bed,self.bath,self.sqft)
            data_input = data.get_data_as_dataframe()
            print(data_input)

            #Loading the image directory
            img_dir = "static/images"

            #Loading the preprocessor for CNN model
            preprocessor1 = load_object("Models/preprocessor.pkl")

            #Preprocessing the initial data frame for CNN model
            tab_df = preprocessor1.transform(data_input)

            #Converting tab data into numpy array
            tab_input = tab_df.toarray()
            
            #Processing the image data for the CNN model
            image_data = process_image(img_dir,self.image_id)

            #CNN model predictions
            initial_predictions = load_cnn_model.predict([tab_input, image_data])

            predicted_price = int(np.squeeze(initial_predictions))

            custom_data_dict ={
                "image_id":[self.image_id],
                "street":[self.street],
                "citi":[self.city],
                "bed":[self.bed],
                "bath":[self.bath],
                "sqft":[self.sqft],
                "predicted_price":[predicted_price]
            }

            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

