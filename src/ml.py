import pandas as pd

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def create_dataframe(file, orient):
    df = None

    # Load the JSON file into a DataFrame
    if orient:
        df = pd.read_json(file, orient='index')
    else:
        df = pd.read_json(file)

    # Print the DataFrame
    print(df)

    # Divider for the output
    print("\n" + "=" * 100 + "\n")

    # Return the DataFrame
    return df


def create_file(file, data):
    with open(file, 'w') as f:
        f.write(data)


def convert_features_to_json(file, new_file='../data/features.json'):
    # Features CSV file
    df2 = pd.read_csv(file, header=None)

    # Since the first column is the Node ID, set it as the index
    df2.set_index(0, inplace=True)

    # Convert the DataFrame to a JSON format
    json_results = df2.to_json(orient='index')

    # Save the JSON as a file
    create_file(new_file, json_results)


def main():
    # Initialize the data_final DataFrame
    data_final = create_dataframe('../data/data_final.json', False)

    # Convert the features CSV file to a JSON file (Call this function only once to create the
    # JSON file)
    convert_features_to_json("../data/elliptic_txs_features.csv")

    # Initialize the features DataFrame
    features = create_dataframe('../data/features.json', True)

    # Combine the DataFrames into a single DataFrame matching the Node ID
    combined = pd.concat([data_final, features])

    # Print the header of the combined DataFrame
    print(combined.head())

    # Divider for the output
    print("\n" + "=" * 100 + "\n")

    # Print the combined DataFrame
    print(combined)

    # Divider for the output
    print("\n" + "=" * 100 + "\n")


if __name__ == '__main__':
    main()
