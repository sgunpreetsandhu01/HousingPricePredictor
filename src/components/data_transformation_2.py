import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Models','preprocessor2.pkl')

class DataTransformationFinal:
    '''
    This function is responsible for Data transformation

    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object_final(self):
        try:
            num1_feature = ["image_id","street"]
            numerical_feature = ["bed","bath","sqft"]
            num2_feature = ["predicted_price"]
            categorical_feature =["citi"]

            num1_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median"))
                ]
            )

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler())
                ]
            )

            num2_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median"))
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("one_hot_encoder",OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num1_pipeline",num1_pipeline,num1_feature),
                    ("numerical_pipeline",numerical_pipeline,numerical_feature),
                    ("num12_pipeline",num2_pipeline,num2_feature),
                    ("categorical_pipeline",categorical_pipeline,categorical_feature)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation_final(self,input_path):
        try:
            input_df=pd.read_csv(input_path)
            

            
            preprocessing_obj=self.get_data_transformer_object_final()

            target_column_name="price"

            input_feature_train_df=input_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=input_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    input_path = "/Users/gunpreetsingh/Gunpreet/programming/Project/Models/final_data.csv"
    data_transformation = DataTransformationFinal()
    data_transformation.initiate_data_transformation_final(input_path)