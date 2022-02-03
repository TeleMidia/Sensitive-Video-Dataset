import os
from sys import argv
import numpy as np
import argparse
import glob
import sys
import pandas as pd

################
# FILE CHECKER #
################
def check_files(path, remove_defective):
    filepaths = glob.glob(path)
    print(f"Scanning {len(filepaths)} files.")
    defective_filepaths = []
    for file in filepaths:
        if not features_checker(file, remove_defective):
            defective_filepaths.append(file)
    print(f'There were {len(defective_filepaths)}/{len(filepaths)} defective videos features!')
    return defective_filepaths

def features_checker(filepath, remove_defective):
    df = pd.read_csv(filepath,index_col=0)

    #checking if dataframe isnt empty
    if df is None or df.empty:
        print(f'{filepath}: Empty file')
        if remove_defective:
            print(df.shape)
            os.remove(filepath)
        return False

    # checking audio features
    audio_df = df.iloc[:, -129:-1]
    if audio_df.sum(axis=1).sum() == 0:
        print(f'{filepath}: Zeroed audio features')
        return False

    # checking frames features
    frames_df = df.iloc[:, :-129]
    if frames_df.sum(axis=1).sum() == 0:
        print(f'{filepath}: Zeroed frames features')
        return False

    return True


if __name__ == u'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,help='features files path, with a asterisc on the end!',required=True)
    parser.add_argument('--remove-defective', type=bool, help='To delete or not defective features files.', required=False, default=False)
    args = parser.parse_args()
    check_files(args.path, args.remove_defective)

