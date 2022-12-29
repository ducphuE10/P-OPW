import sys
import os
import pandas as pd
import argparse
import numpy as np

def get_list_of_datasets():
    data_summary_path = './UCR_data/DataSummary.xlsx'
    df = pd.read_excel(data_summary_path)
    df.columns = df.columns.str.strip()

    #convert Langth, Train, Test to int and remove not int row
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df['Train'] = pd.to_numeric(df['Train'], errors='coerce')
    df['Test'] = pd.to_numeric(df['Test'], errors='coerce')
    #sort by Length **2 * Train * Test
    df['L2TT'] = df['Length'] ** 2 * df['Train'] * df['Test']
    df_sort = df.sort_values(by=['L2TT'], ascending=[True])

    list_dir = df_sort['Name'].tolist()
    return list_dir

# list_dir = os.listdir('./UCR_data/UCRArchive_2018')
# print(list_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='baseline')
    parser.add_argument('--method', type=str, default='opw')

    args = parser.parse_args()


    list_dir = get_list_of_datasets()
    list_dir = [x for x in list_dir if not x.startswith('Dodger')]
    experiment_file_path = './exp/experiments_results_newopw.csv'
    df_write = pd.read_csv(experiment_file_path, index_col=0)

    for dataset in list_dir[:44]:
        print(dataset)
        # if dataset in df_write.index:
        #     print('Skip')
        #     continue
        if dataset in df_write.index:
            if not np.isnan(df_write.loc[dataset, args.method]):
                print('Skip')
                continue

        if args.experiment != 'baseline':
            os.system(f'python main.py --dataset {dataset}  \
                -tl1 0.3 \
                -tl2 0.3 \
                -m topw1 \
                -l1 50 \
                -l2 0.1 \
                --delta 1 \
                -m topw1 \
                --trend_method l1 \
                -e experiments_results_28  ')
        else:
            os.system(f'python baseline_main.py --dataset {dataset} -m {args.method}')