import argparse
import os
import azureml.core
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset

run = Run.get_context()
ws = run.experiment.workspace

def write_output(df, path):
    os.makedirs(path, exist_ok=True)
    print("%s created" % path)
    df.to_csv(path + "/part-00000", index=False)

print("Split the data into train and test")

parser = argparse.ArgumentParser("split")
parser.add_argument("--target_column", type=str, help="input split features")
parser.add_argument("--input_dataset", type=str, help="input split filename")
parser.add_argument("--output_split_train_x", type=str, help="output split train features")
parser.add_argument("--output_split_train_y", type=str, help="output split train labels")
parser.add_argument("--output_split_test_x", type=str, help="output split test features")
parser.add_argument("--output_split_test_y", type=str, help="output split test labels")

args = parser.parse_args()

print("Argument 1(input_dataset): %s" % args.input_dataset)
print("Argument 2(target_column): %s" % args.target_column)
print("Argument 3(output_train_x_split): %s" % args.output_split_train_x)
print("Argument 4(output_train_y_split): %s" % args.output_split_train_y)
print("Argument 5(output_test_x_split): %s" % args.output_split_test_x)
print("Argument 6(output_test_y_split): %s" % args.output_split_test_y)

# joining the filename with the folder path to fetch the data from the passed file
train_ds = Dataset.get(ws, args.input_dataset)
train_df = train_ds.to_pandas_dataframe()

#train_df = pd.read_csv(args.input_dataset)
print(train_df.head(5))

le = LabelEncoder()
le.fit(train_df[args.target_column].values)
y = pd.DataFrame({'target':le.transform(train_df[args.target_column].values)})

# need to drop any BOUGHT CATEGORY columns - we do not know these at runtime
cols_to_drop = train_df.columns[train_df.columns.str.contains(pat = 'BOUGHT_CATEGORY_')].tolist()
cols_to_drop.append("ACCOUNT_NUMBER")

X = train_df.drop(cols_to_drop, axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=223)

if not (args.output_split_train_x is None and args.output_split_test_x is None and args.output_split_train_y is None and args.output_split_test_y is None):
    write_output(x_train, args.output_split_train_x)
    write_output(y_train, args.output_split_train_y)
    write_output(x_test, args.output_split_test_x)
    write_output(y_test, args.output_split_test_y)

