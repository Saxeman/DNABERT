# Gene splice dataset
# 3 Classes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import argparse
import os

def main(args):
    csv_path = args.csv_path
    dest_path = os.path.expanduser(args.dest_path)
    raw_file = pd.read_csv(csv_path)
    format_df = convert_to_molcure_format(raw_file)
    create_train_val_test(format_df)
    return

def convert_to_molcure_format(file_df: pd.DataFrame) -> pd.DataFrame:
    """Converts splice.data into Molcure format. Prints df head and label counts

    Args:
        file_df (pd.DataFrame): splice.data file

    Returns:
        pd.DataFrame: Molcure Format dataframe. 
    """
    columns = ["seq_id", "seq_expr", "seq_label"]
    label_encoder = preprocessing.LabelEncoder()
    # Extract columns
    lab_column = file_df[file_df.columns[0]]
    sid_column = file_df[file_df.columns[1]]
    seq_column = file_df[file_df.columns[2]]
    
    # Create dataframe and rename
    format_df = pd.concat((sid_column, seq_column, lab_column), axis=1)
    format_df.columns = columns
    
    # Categorize class variable
    label_encoder.fit(format_df.seq_label)
    format_df["seq_label"] = label_encoder.transform(format_df.seq_label)
    
    # Count labels and print
    classes_value = format_df["seq_label"].value_counts()
    print("** Samples per class ***")
    print(classes_value)
    print("** *** *** *** *** * ***")
    print(format_df.head())
    return format_df

def create_train_val_test(format_df: pd.DataFrame, train_size: float=0.8) -> bool:
    """Creates train, validation and test datasets with the 
    train_size variable distribution

    Args:
        train_size (float, optional): Train dataset size. Defaults to 0.8.

    Returns:
        bool: True if succesful
    """
    train_df, val_df = train_test_split(format_df, train_size=train_size, stratify=format_df["seq_label"])
    val_df, test_df = train_test_split(val_df, train_size=0.5)
    
    # Print distributions
    train_counts = train_df["seq_label"].value_counts()
    print(f"Train distribution: \n{train_counts}")
    val_counts = val_df["seq_label"].value_counts()
    print(f"Val distribution: \n{val_counts}")
    test_counts = test_df["seq_label"].value_counts()
    print(f"Test distribution: \n{test_counts}")
    # Save to file
    train_df.to_csv("dataset_files/train.csv", index=False)
    val_df.to_csv("dataset_files/val.csv", index=False)
    test_df.to_csv("dataset_files/test.csv", index=False)
    return



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train, test and validation data from a given source",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
    parser.add_argument("--csv_path", type=str, help="Data Source location")
    parser.add_argument("--dest_path", type=str, help="Destination location of the generated files")
    args = parser.parse_args()
    main(args) 