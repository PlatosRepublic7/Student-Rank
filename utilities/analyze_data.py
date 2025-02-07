import argparse
import pandas as pd
import numpy as np


def count_column_entries(df: pd.DataFrame) -> pd.DataFrame:
    count_columns = ['Company'] + [x for x in range(1, 11)]
    count_df = pd.DataFrame(columns=count_columns)
    i = 0
    for col_name, col_data in df.items():
        if col_name == 'Student':
            continue
        else:
            row_data = []
            column_counts = [0 for x in range(10)]
            row_data.append(col_name)
        for value in col_data:
            column_counts[value-1] += 1
        row_data = row_data + column_counts
        count_df.loc[i] = row_data
        i += 1

    return count_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help=".csv file which will be analyzed")
    args = parser.parse_args()
    
    # Read in file, and make dataframe
    df = pd.read_csv(args.file)
    c_df = count_column_entries(df)

    c_df.to_csv('test_data_counts.csv', index=False)

    return

if __name__ == '__main__':
    main()