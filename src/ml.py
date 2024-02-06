import pandas as pd


def create_dataframe(file, orient):
    df = None

    # Load the JSON file into a DataFrame
    if orient:
        df = pd.read_json(file, orient='index')
    else:
        df = pd.read_json(file)

    # Print the DataFrame
    print(df)

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


if __name__ == '__main__':
    main()
