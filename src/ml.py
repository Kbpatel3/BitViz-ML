import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

WINDOWS_PATH_1 = 'data\data_final.json'
WINDOWS_PATH_2 = 'data\elliptic_txs_features.csv'
LINUX_PATH_1 = '../data/data_final.json'
LINUX_PATH_2 = '../data/elliptic_txs_features.csv'


def get_edges_dataframe(df):
    # Explode the "Edges" column to separate the dictionary into individual rows
    edges_df = df.explode('Edges').reset_index(drop=True)

    # Normalize the 'Edges' dictionary
    edges_df = pd.json_normalize(edges_df[edges_df['Edges'].notna()]['Edges'])

    # Remove duplicate rows
    edges_df = edges_df.drop_duplicates()

    # Rename the columns
    edges_df.rename(columns={'id': 'Node ID', 'timestep': 'Timestep', 'group': 'Group'},
                    inplace=True)

    # Print the DataFrame
    print_dataframe(edges_df, "Edges")

    # Print divider
    print_divider()

    return edges_df


def load_json_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    return data


def print_dataframe(df, header):
    print("\n\n" + "=" * 100)
    print(header)
    print("=" * 100 + "\n")
    print(df)


def print_divider():
    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n")


def build_csv_dataframe(file):
    # Column Names for the CSV
    column_names = ['Node ID', 'Timestep'] + [f'Feature {i}' for i in range(2, 167)]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file, header=None, names=column_names)

    # Drop the column 'Timestep' because we have it in the 'data_final' DataFrame
    df.drop(columns=['Timestep'], inplace=True)

    return df


def build_json_dataframe(file):
    # Open and load the JSON files into a dictionary
    data_final_dict = load_json_file(file)

    # Convert the JSON data into a DataFrame
    df = pd.json_normalize(data_final_dict, record_path=['data'])

    # Rename the columns to match the JSON keys
    df.rename(
        columns={'id': 'Node ID', 'group': 'Group', 'timestep': 'Timestep', 'edges': 'Edges'},
        inplace=True)

    # Convert the 'Node ID' and 'Timestep' columns to int64 and float64 data types.
    df['Node ID'] = df['Node ID'].astype('int64')

    # Extract edges into a separate DataFrame
    edges_df = get_edges_dataframe(df)

    # Remove the edges column from the original dataframe
    df = df.drop(columns=['Edges'])

    return df, edges_df


def create_dataframe(file, is_feature):
    # Initialize the DataFrame
    df = None

    if not is_feature:
        # Convert the JSON data into a DataFrame
        df, edges_df = build_json_dataframe(file)

        # Print the DataFrame
        print_dataframe(df, "Data Final")
    else:
        # Read the CSV file into a DataFrame
        df = build_csv_dataframe(file)

        # Print the DataFrame
        print_dataframe(df, "Features")

    # Print divider
    print_divider()

    return df


def merge_dataframes(data_final, features):
    # Merge the DataFrames on 'Node ID'
    df = pd.merge(data_final, features, on=['Node ID'], how='inner')

    # Rename the Timestep column to 'Feature 1' since it is the first feature from the original CSV
    df.rename(columns={'Timestep': 'Feature 1'}, inplace=True)

    # Print the DataFrame
    print_dataframe(df, "Combined Data")

    # Print divider
    print_divider()

    return df


def machine_learning(df):
    # Extract the feature columns into X
    X = df.iloc[:, 2:]
    print_dataframe(X, "Features for Machine Learning")
    print_divider()

    # Extract the group column into y
    y = df["Group"]
    print_dataframe(y, "Groups for Machine Learning")
    print_divider()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    print("Split the data into traning and testing sets")

    # Creating the Random Forest Classifier
    random_forest_classifier = RandomForestClassifier(n_estimators=100)

    print("Created the Random Forest Classifier")

    print("Starting to train the model...")

    # Train the model using the training sets
    random_forest_classifier.fit(X_train, y_train)

    print("Model has been trained")

    # Predict the response for the test dataset
    y_pred = random_forest_classifier.predict(X_test)

    print("Predicted the response for the test dataset")

    # Print the accuracy of the model
    print(f"Accuracy: {random_forest_classifier.score(X_test, y_test)}")


def main():
    # Initialize the data_final DataFrame
    data_final_df = create_dataframe(LINUX_PATH_1, False)

    # Initialize the features DataFrame
    features_df = create_dataframe(LINUX_PATH_2, True)

    # Merge the DataFrames
    combined_df = merge_dataframes(data_final_df, features_df)

    # Start machine learning here
    machine_learning(combined_df)


if __name__ == '__main__':
    main()
