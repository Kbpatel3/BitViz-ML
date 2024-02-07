import pandas as pd
import json


def load_json_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    return data

def create_dataframe(file, is_feature):
    # Initialize the DataFrame
    df = None

    if not is_feature:
        # Open and load the JSON files into a dictionary
        data_final_dict = load_json_file(file)

        # Convert the JSON data into a DataFrame
        df = pd.json_normalize(data_final_dict, record_path=['data'])

        # Rename the columns to match the JSON keys
        df.rename(columns={'id': 'Node ID', 'group': 'Group', 'timestep': 'Timestep', 'edges': 'Edges'}, inplace=True)

        # Print the DataFrame
        print("\n\n" + "=" * 100)
        print("Final Data DataFrame")
        print("=" * 100 + "\n")
        print(df)


        # Convert the 'Node ID' and 'Timestep' columns to int64 and float64 data types.
        df['Node ID'] = df['Node ID'].astype('int64')

    else:
        # Column Names for the CSV
        column_names = ['Node ID', 'Timestep'] + [f'Feature {i}' for i in range(2, 167)]

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, header=None, names=column_names)

        # Drop the column 'Timestep' because we have it in the 'data_final' DataFrame
        df.drop(columns=['Timestep'], inplace=True)

        # Print the DataFrame
        print("\n\n" + "=" * 100)
        print("Features DataFrame")
        print("=" * 100 + "\n")
        print(df)

    # Print divider
    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n")

    return df

def merge_dataframes(data_final, features):
    # Merge the DataFrames on 'Node ID'
    df = pd.merge(data_final, features, on=['Node ID'], how='inner')

    # Print the DataFrame
    print("\n" + "=" * 100)
    print("Merged DataFrame")
    print("=" * 100 + "\n")
    print(df)

    # Print divider
    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n")


def main():
    # Initialize the data_final DataFrame
    data_final_df = create_dataframe('data\data_final.json', False)

    # Initialize the features DataFrame
    features_df = create_dataframe('data\elliptic_txs_features.csv', True)

    # Merge the DataFrames
    combined_df = merge_dataframes(data_final_df, features_df)


if __name__ == '__main__':
    main()
