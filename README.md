# Housing Price Predictor

## Project Overview

This project is a web-based application that predicts housing prices based on input features such as the number of bedrooms, bathrooms, area, and other characteristics. The app allows users to input house details and get a predicted price. Users can also upload an image of the house, which can be used in future enhancements to incorporate image-based predictions.

## Features

- Input house details (street, city, number of bedrooms, bathrooms, area) to predict the price.
- Display predicted price along with details entered by the user.
- Option to upload house images for future model integration.

## Files and Directories

- **`home.html`**: Front-end page where users enter house details to get a price prediction.
- **`predict.html`**: Page displaying the predicted house price along with user inputs.
- **`uploadimage.html`**: Page for uploading house images.
- **`index.html`**: Landing page welcoming users to the housing price portal.
- **`application.py`**: The main Flask application file that sets up routes and serves the web pages.
- **`predict_pipeline.py`**: Contains the machine learning pipeline for predicting house prices based on input data.
- **`data_transformation_2.py`**: Contains code related to transforming and preprocessing the input data for the prediction model.
- **`requirements.txt`**: Lists the Python dependencies required for running the application.

## Technologies Used

- **Flask**: For building the web server and handling routes.
- **pandas, numpy**: For data manipulation and analysis.
- **scikit-learn, catboost, xgboost**: For building and training the machine learning models.
- **tensorflow**: For building and training CNN model.
- **Bootstrap**: For responsive front-end design.

## How to Run the Project

### Prerequisites

Make sure you have **Python 3.9+** installed.

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/yourusername/HousingPricePredictor.git
cd HousingPricePredictor
```
2. Navigate to the project directory.
3. Install the required dependencies by running:

```bash
pip install -r requirements.txt
