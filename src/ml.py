"""
Filename: ml.py
Author: Kaushal Patel & Noah Hassett
Date: 05-07-2024
Description: This script includes functions to load data from JSON and CSV files, preprocess
the data, train a Random Forest Classifier model, make predictions on unknown data,
and save the trained model. It also provides functionality to query the predicted data
based on Node ID. The script is designed to work with data related to transactions and
their features, and it uses pandas for data manipulation, sklearn for machine learning,
and joblib for saving the model.
"""

# Imports for the script
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Model Save Path and Extension
MODEL_SAVE_PATH_LINUX = '../model/'
MODEL_EXTENSION = '.joblib'
MODEL_SAVE_PATH_WINDOWS = '..\model'

# Data Structures for the JSON file
nodes = []
edges = []


def get_edges_dataframe(df):
    """
    Extracts the 'Edges' column from the DataFrame and returns a new DataFrame with the edge's
    data.
    :param df: The DataFrame to extract the edges from
    :return: A new DataFrame with the edges data
    """

    # Explode the "Edges" column to separate the dictionary into individual rows
    edges_df = df.explode('Edges').reset_index(drop=True)

    # Normalize the 'Edges' dictionary
    edges_df = pd.json_normalize(edges_df[edges_df['Edges'].notna()]['Edges'])

    # Remove duplicate rows
    edges_df = edges_df.drop_duplicates()

    # Rename the columns
    edges_df.rename(columns={'id': 'Node ID', 'timestep': 'Timestep', 'group': 'Group'},
                    inplace=True)

    return edges_df


def load_json_file(file):
    """
    Loads a JSON file into a dictionary
    :param file: The file to load
    :return: A dictionary with the JSON data
    """

    # Open and load the JSON files into a dictionary
    with open(file, 'r') as f:
        data = json.load(f)

    return data


def print_divider():
    """
    Prints a divider to separate the output
    :return: None
    """

    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n")


def print_dataframe(df, header):
    """
    Prints the DataFrame with a header
    :param df: The DataFrame to print
    :param header: The header to print
    :return: None
    """

    print("\n\n" + "=" * 100)
    print(header)
    print("=" * 100 + "\n")
    print(df)
    print_divider()


def build_csv_dataframe(file):
    """
    Reads the CSV file into a DataFrame and returns it
    :param file: The CSV file to read
    :return: A DataFrame with the CSV data
    """

    # Column Names for the CSV
    column_names = ['Node ID', 'Timestep'] + [f'Feature {i}' for i in range(2, 167)]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file, header=None, names=column_names)

    # Drop the column 'Timestep' because we have it in the 'data_final' DataFrame
    df.drop(columns=['Timestep'], inplace=True)

    return df


def create_data_structure(data_final_dict):
    """
    Creates the data structure for the JSON file
    :param data_final_dict: The dictionary with the JSON data
    :return: None
    """

    # Reference the global variables to modify them
    global nodes
    global edges

    # Loop through the data in the JSON file
    for data in data_final_dict['data']:
        # Convert the group to an int64
        data['group'] = int(data['group'])

        # Append the 'id', 'group', and 'timestep' to the 'nodes' list
        nodes.append({'id': data['id'], 'group': data['group'], 'timestep': data['timestep']})

        # Loop through the edges in the data
        for edge in data['edges']:
            # Convert the group to an int64
            edge['group'] = int(edge['group'])

            # Append the 'id', 'timestep', and 'group' to the 'edges' list
            edges.append({'from': data['id'], 'id': edge['id'], 'timestep': edge['timestep'],
                          'group': edge['group']})


def build_json_dataframe(file):
    """
    Builds a DataFrame from the JSON file and returns it
    :param file: The JSON file to read
    :return: A DataFrame with the JSON data
    """

    # Open and load the JSON files into a dictionary
    data_final_dict = load_json_file(file)

    # Create the data structure for the JSON file
    create_data_structure(data_final_dict)

    # Convert the JSON data into a DataFrame
    df = pd.json_normalize(data_final_dict, record_path=['data'])

    # Rename the columns to match the JSON keys
    df.rename(
        columns={'id': 'Node ID', 'group': 'Group', 'timestep': 'Timestep', 'edges': 'Edges'},
        inplace=True)

    # Convert the 'Node ID' and 'Timestep' columns to int64 and float64 data types.
    df['Node ID'] = df['Node ID'].astype('int64')

    # Convert the 'Group' column to int64 data type
    df['Group'] = df['Group'].astype('int64')

    # Extract edges into a separate DataFrame
    edges_df = get_edges_dataframe(df)

    # Remove the edges column from the original dataframe
    df = df.drop(columns=['Edges'])

    return df, edges_df


def create_dataframe(file, is_feature):
    """
    Creates a DataFrame from the specified file. If the file is a JSON file, it will be converted
    into a DataFrame using the build_json_dataframe function. If the file is a CSV file,
    it will be read into a DataFrame using the build_csv_dataframe function.

    :param file: The file to create the DataFrame from
    :param is_feature: A boolean to indicate if the file is a feature file
    :return: A DataFrame with the data from the file and a DataFrame with the edges data if the
    """

    # Initialize the DataFrame
    df = None
    edges_df = None

    # Check if the file is a feature file
    if not is_feature:
        # Convert the JSON data into a DataFrame
        df, edges_df = build_json_dataframe(file)
    else:
        # Read the CSV file into a DataFrame
        df = build_csv_dataframe(file)

    return df, edges_df


def merge_dataframes(data_final, features):
    """
    Merges the 'data_final' and 'features' DataFrames on 'Node ID' and returns the merged DataFrame
    :param data_final: The 'data_final' DataFrame
    :param features: The 'features' DataFrame
    :return: The merged DataFrame
    """

    # Merge the DataFrames on 'Node ID'
    df = pd.merge(data_final, features, on=['Node ID'], how='inner')

    # Rename the Timestep column to 'Feature 1' since it is the first feature from the original CSV
    df.rename(columns={'Timestep': 'Feature 1'}, inplace=True)

    return df


def separate_unknown(df):
    """
    Separates the DataFrame into two DataFrames. One with the instances where the group is 3 (
    unknown group) and another with the instances where the group is not 3 (Illicit or Licit
    group)
    :param df: The DataFrame to separate
    :return: Two DataFrames. One with the instances where the group is 3 (unknown group) and
    another with the instances where the group is not 3 (Illicit or Licit group)
    """

    # Separate the instances where the group is 3 (unknown group)
    df_predict = df[df['Group'] == 3]

    # Separate the instances where the group is not 3 (Illicit or Licit group)
    df_train_test = df[df['Group'] != 3]

    # Return the DataFrames
    return df_predict, df_train_test


def extract_features_and_groups(df_train_test):
    """
    Extracts the feature columns into X and the group column into y. Returns X and y
    :param df_train_test: The DataFrame to extract the features and groups from
    :return: X and y
    """

    # Extract the feature columns into X
    X = df_train_test.iloc[:, 2:]

    # Extract the group column into y
    y = df_train_test["Group"]

    return X, y


def split_data_train_test(X, y):
    """
    Splits the data into training and testing sets and returns the split data sets
    :param X: The features
    :param y: The groups
    :return: X_train, X_test, y_train, y_test
    """

    # Initialize the user_test_size variable
    user_test_size = 0.3

    try:
        # Ask the user to enter the percentage of the data to be used for testing
        user_test_size = float(
            input("Enter the percentage of the data to be used for testing (0.0 - 1.0): "))
    except ValueError:
        print("You did not enter a valid number. Defaulting to 0.3 for a 70/30 split")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=user_test_size)

    return X_train, X_test, y_train, y_test


def create_random_forest_classifier():
    """
    Creates a Random Forest Classifier and returns it
    :return: A Random Forest Classifier
    """

    # Create the Random Forest Classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=100, verbose=3)

    return random_forest_classifier


def train_and_test_model(random_forest_classifier, X_train, y_train, X_test):
    """
    Trains the model using the training sets and predicts the response for the test dataset using
    the trained model and returns the predicted response.
    :param random_forest_classifier: The Random Forest Classifier
    :param X_train: The training features
    :param y_train: The training groups
    :param X_test: The testing features
    :return: The predicted response for the test dataset
    """

    print("Starting to train the model...")

    # Train the model using the training sets
    random_forest_classifier.fit(X_train, y_train)

    print("Model has been trained")

    print("Predicting the response for the test dataset")

    # Predict the response for the test dataset
    y_pred = random_forest_classifier.predict(X_test)

    return y_pred


def save_model(model, save_path):
    """
    Saves the model to the specified path
    :param model: The model to save
    :param save_path: The path to save the model to
    :return: None
    """

    # Save the model to the specified path
    print(f"Saving the model to {save_path}")
    joblib.dump(model, save_path)


def predict_all_unknown(model, df_predict):
    """
    Predicts the group for all unknown transactions and returns the DataFrame with the predicted
    groups.
    :param model: The trained model
    :param df_predict: The DataFrame with the unknown transactions
    :return: The DataFrame with the predicted groups
    """

    # Fit the trained model to the prediction data
    y_predict = model.predict(df_predict.iloc[:, 2:])

    # Change the 3's to the predicted values
    df_predict['Group'] = y_predict

    # Print the DataFrame
    print_dataframe(df_predict, "Predicted Data")

    return df_predict


def machine_learning(df):
    """
    Performs machine learning on the DataFrame and saves the model to the specified path. Calls the
     necessary functions to perform machine learning.
    :param df: The DataFrame to perform machine learning on
    :return: The trained model
    """

    # Separate the unknown transactions from the known transactions
    df_predict, df_train_test = separate_unknown(df)

    # Print the DataFrames
    print_dataframe(df_predict, "Predict Data")
    print_dataframe(df_train_test, "Train and Test Data")

    # Extract the features and groups
    X, y = extract_features_and_groups(df_train_test)

    # Print the X(Features) and y(Groups) data
    print_dataframe(X, "Features for Machine Learning")
    print_dataframe(y, "Groups for Machine Learning")

    print("Splitting the data into traning and testing sets")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data_train_test(X, y)

    print("Creating the Random Forest Classifier")
    random_forest_classifier = create_random_forest_classifier()

    # Train and test the model
    y_pred = train_and_test_model(random_forest_classifier, X_train, y_train, X_test)

    # Print the y_pred data
    print_dataframe(y_pred, "Predicted Data for Machine Learning")

    # Print the accuracy of the model
    print(f"Accuracy: {random_forest_classifier.score(X_test, y_test)}")

    # Saving the model
    #save_model(random_forest_classifier, MODEL_SAVE_PATH_LINUX)
    save_model(random_forest_classifier, MODEL_SAVE_PATH_WINDOWS)

    # Predict all unknown transactions
    df_predict = predict_all_unknown(random_forest_classifier, df_predict)

    # Combine the predicted data with the known data
    df = pd.concat([df_predict, df_train_test])

    # Return the updated DataFrame that was previously unknown
    return df


def user_print_input(data_final_df, features_df, combined_df):
    """
    Asks the user if they want to print the dataframes and prints them if the user enters 'y' or
    'Y' and does nothing if the user enters 'n' or 'N' or anything else.
    :param data_final_df: The data_final DataFrame
    :param features_df: The features DataFrame
    :param combined_df: The combined DataFrame
    :return: None
    """

    # Ask the user if they want to print the dataframes
    print_data = input("Do you want to print the dataframes? (y/n): ")

    # Print the DataFrames if the user enters 'y' or 'Y'
    if print_data.lower() == 'y':
        print_dataframe(data_final_df, "Data Final")
        print_dataframe(features_df, "Features")
        print_dataframe(combined_df, "Combined Data")
    else:
        print("Dataframes will not be printed")


def start_ml(combined_df):
    """
    Asks the user if they want to start machine learning and starts machine learning if the user
    enters 'y' or 'Y' and does nothing if the user enters 'n' or 'N' or anything else.
    :param combined_df: The combined DataFrame
    :return: The DataFrame with the predicted groups that were previously unknown
    """

    # Reference the global variable to modify it
    #global MODEL_SAVE_PATH_LINUX
    global MODEL_SAVE_PATH_WINDOWS

    # Initialize the return DataFrame
    df_final = None

    # Ask the user if they want to start machine learning
    start_ml_input = input("Do you want to start machine learning? (y/n): ")

    # Start machine learning if the user enters 'y' or 'Y'
    if start_ml_input.lower() == 'n':
        print("Machine Learning will not start")
    elif start_ml_input.lower() == 'y':
        print("Machine Learning will start")

        # Ask the user to enter the file name for the model to be saved
        file_name = input("Enter the file name for the final model: ")

        # Set the model save path
        #MODEL_SAVE_PATH_LINUX = MODEL_SAVE_PATH_LINUX + file_name + MODEL_EXTENSION
        MODEL_SAVE_PATH_WINDOWS = MODEL_SAVE_PATH_WINDOWS + file_name + MODEL_EXTENSION

        #print(f"Model will be saved to {MODEL_SAVE_PATH_LINUX}")
        print(f"Model will be saved to {MODEL_SAVE_PATH_WINDOWS}")

        # Start machine learning here
        df_final = machine_learning(combined_df)
    else:
        print("Invalid input. Machine Learning will not start")
        exit(0)

    return df_final


def query_data(df_predict):
    """
    Queries the predicted data and prints the results
    :param df_predict: The DataFrame with the predicted groups that were previously unknown
    :return: None
    """

    # Loop to ask the user for a Node ID to query
    while True:
        node_id = input("Enter a Node ID to query (Enter 'q' to quit): ")

        # Check if the user wants to quit
        if node_id.lower() == 'q':
            print("Query will end")
            exit(0)

        try:
            # Convert the input to an integer
            node_id = int(node_id)
        except ValueError:
            print("You did not enter a valid number. Try again")
            continue

        # Query the predicted data
        query = df_predict[df_predict['Node ID'] == node_id]

        # Print the query results
        print_dataframe(query, "Query Results")


def user_query(df_predict):
    """
    Asks the user if they want to query the predicted data and queries the predicted data if the
    user enters 'y' or 'Y' and does nothing if the user enters 'n' or 'N' or anything else.
    :param df_predict: The DataFrame with the predicted groups that were previously unknown
    :return: None
    """

    # Ask the user if they want to query the predicted data
    query_input = input("Do you want to query the predicted data? (y/n): ")

    # Query the predicted data if the user enters 'y' or 'Y'
    if query_input.lower() == 'n':
        print("Query will not start")
    elif query_input.lower() == 'y':
        print("Query will start")

        # Start querying the predicted data here
        query_data(df_predict)
    else:
        print("Invalid input. Query will not start")
        exit(0)


def update_group_values(df_final):
    global nodes
    global edges

    # Change group values in nodes and edges to the values in the combined_df

    # Update nodes
    for node in nodes:
        # Find the row in the DataFrame that matches the Node ID
        row = df_final[df_final['Node ID'] == int(node['id'])]

        # Check if the row is not empty
        if not row.empty:
            # Get the group value from the row
            group = row['Group'].values[0]

            # Update the group value in the node dictionary
            node['group'] = str(group)

    # Update edges
    for edge in edges:
        # Find the row in the DataFrame that matches the Node ID
        row = df_final[df_final['Node ID'] == int(edge['id'])]

        # Check if the row is not empty
        if not row.empty:
            # Get the group value from the row
            group = row['Group'].values[0]

            # Update the group value in the edge dictionary
            edge['group'] = str(group)


def export_data(df_final):
    """
    Exports the trained data to a JSON in the same format as the original data_final.json
    :return: None
    """
    """
    Format of the JSON file:
    {
        "data": [
            {
                "id": 0,
                "group": 1,
                "timestep": 1,
                "edges": [
                    {
                        "id": 1,
                        "timestep": 1,
                        "group": 1
                    }
                ]
            }
        ]
    }
    """

    # Change group values in nodes and edges to the values in the combined_df
    update_group_values(df_final)

    # Create the dictionary to be saved as a JSON
    data = {'data': nodes}

    # Map the edges to the nodes based on the 'from' key
    for node in nodes:
        # Add an 'edges' key to the node
        node['edges'] = []
        for edge in edges:
            # Check if the 'from' key matches the node's 'id'
            if edge['from'] == node['id']:
                # Add the edge to the node but don't include the 'from' key
                node['edges'].append({key: value for key, value in edge.items() if key != 'from'})

    # Save the dictionary as a JSON file
    print("Saving the predicted data to predicted_data_final.json")

    # Get the current directory
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, '..', 'model')

    # Check if the model directory exists
    print(os.path.exists(model_path))

    with open(os.path.join(model_path, 'predicted_data_final.json'), 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """
    The main function to run the program. Calls the necessary functions to perform the program of
    creating the DataFrames, merging them, and starting machine learning.
    :return: None
    """
    global nodes
    global edges

    # Let the user know that the program is going to be reading the data files
    print("Reading the data files...")

    # Initialize the data_final DataFrame
    #data_final_df, edges_df = create_dataframe(LINUX_PATH_1, False)
    path_to_data_final = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_final.json')
    data_final_df, edges_df = create_dataframe(path_to_data_final, False)

    # Initialize the features DataFrame
    #features_df, no_use = create_dataframe(LINUX_PATH_2, True)
    path_to_features = os.path.join(os.path.dirname(__file__), '..', 'data',
                                    'elliptic_txs_features.csv')
    features_df, no_use = create_dataframe(path_to_features, True)

    # Merge the DataFrames
    combined_df = merge_dataframes(data_final_df, features_df)

    # User input to ask if user wants to print the dataframes
    user_print_input(data_final_df, features_df, combined_df)

    # User input to ask if the user wants to start machine learning
    df_final = start_ml(combined_df)

    # Loop to query the predicted data
    if df_final is not None:
        user_query(df_final)

        # Predict the group values for both nodes and edges dictionary
        export_data(df_final)


if __name__ == '__main__':
    # Call the main function
    main()
