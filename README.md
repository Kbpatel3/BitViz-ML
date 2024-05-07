# BitViz-ML

## Project Authors
- Name: Kaushal Patel
- Name: Noah Hassett
- GitHub Profile: [Kaushal Patel](https://github.com/kbpatel3/)
- GitHub Profile: [Noah Hassett](https://github.com/na245/)

## Organization
Math and Computer Science Department at [Western Carolina University](https://www.wcu.edu/)
- Course: CS 495-496 - Capstone Project

## Description
This repository contains the code for the BitViz-ML project. The project aims to develop a 
machine learning model that can predict whether a given Bitcoin transaction is fraudulent or not.
The dataset is from Elliptic Data Set, which contains information about Bitcoin transactions and
their labels. The dataset is available at https://www.kaggle.com/ellipticco/elliptic-data-set.

---
We use Random Forest Classifier to predict the labels of the transactions. The model is trained on
the features of the dataset. The features are extracted from the dataset and are used to train the
model. The model is then used to predict the labels of the transactions. The predicted labels are
then used to filter the dataset. The filtered dataset is then used to load into the Neo4J database.

## Files
- `src/ml.py`: Contains the code for generating, training, and exporting the Machine Learning 
  model.
- `src/filter.py`: Contains the code for filtering the dataset. Currently, it provides total of 7
    filters.
- `model/predicted_data_final.json`: Contains the predicted data from the model.
- `model/Final Predicted Data.rar`: Contains the predicted data from the model in RAR format.
- `data/elliptic_txs_features.csv`: Contains the features of the dataset.
    - Used in `src/ml.py` for training the model.
- `data/data_final.json`: Contains the final data before machine learning.
    - Used in `src/ml.py` for training the model.
- `data/filtered_models/data/*`: Contains the JSON files of the filtered data post ML.
    - This is the data that is used to load into the Neo4J database. It is all the nodes and 
      their group categorization (Illicit, Licit, or Unknown) and edges.
- `data/filtered_models/metadata/*`: Contains the metadata of the filtered data post ML.
    - This is the metadata that is used to load into the Neo4J database. It is all the 
      timesteps and the number of illict, unknown, or licit nodes within each timestep.

## Usage
1. Clone the repository.
    ```bash
    git clone https://github.com/Kbpatel3/BitViz-ML.git
    ```
2. Navigate to the project directory.
    ```bash
    cd BitViz-ML
    ```
3. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the `ml.py` file to generate, train, and export the Machine Learning model.
    ```bash
    python src/ml.py
    ```
5. Run the `filter.py` file to filter the dataset.
    ```bash
    python src/filter.py
    ```

## Requirements
- Python 3.8 or higher
- pandas
- sklearn
- joblib