from utils import *
from data import Dataloader as dl
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from typing import Literal


def helper(dataset: Literal["syn_100", "musk_v2"], method: Literal["ours", "passive", "lev_score_samp"]) -> None:
    if dataset == "syn_100":
        schedule = list(range(100, 2000, 100)) + [2000]
        npzfile = np.load("./data/syn_100.npz")
        gt_weight = npzfile["weight"]
        gt_bias = npzfile["bias"]
    elif dataset == "musk_v2":
        schedule = list(range(20, 1400, 20)) + [1400]
        gt_weight = None
        gt_bias = None
    evaluator = Evaluator(schedule=schedule)
    df_result = pd.DataFrame()
    df_result[('schedule', '')] = schedule
    folder = f"outputs/{method}/{dataset}/"
    csv_path = f"{folder}/results.csv"
    train_data, train_labels, test_data, test_labels = dl.load_data(
        dataset=dataset)
    files = [filename for filename in os.listdir(
        folder) if filename.endswith(".npz")]
    avg_acc_trai, var_acc_trai, avg_acc_test, var_acc_test, avg_acc_quer, var_acc_quer, prev_num_samp = \
        [], [], [], [], [], [], 0
    avg_l2, var_l2 = [], []
    for filename in tqdm(files, desc="#files gone through"):
        file_path = osp.join(folder, filename)
        query_data, query_labels = dl.load_queries_from_file(
            file_path=file_path)
        train_set_accuracies, test_set_accuracies, query_set_accuracies, l2_to_gt = evaluator.eval(
            query_data=query_data,
            query_labels=query_labels,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            gt_weight=gt_weight,
            gt_bias=gt_bias)

        df_result[(filename, 'train_set_accuracies')
                  ] = pd.Series(train_set_accuracies)
        df_result[(filename, 'test_set_accuracies')
                  ] = pd.Series(test_set_accuracies)
        df_result[(filename, 'query_set_accuracies')
                  ] = pd.Series(query_set_accuracies)

        # l2 for syn_100
        if dataset == "syn_100":
            df_result[(filename, 'l2')] = pd.Series(l2_to_gt)
            avg_l2, var_l2 = update_avg_and_var(
                l2_to_gt, avg_l2, var_l2, prev_num_samp
            )
            df_result[('average', 'l2')] = pd.Series(avg_l2)
            df_result[('variance', 'l2')] = pd.Series(var_l2)

        # call update_avg_and_var
        avg_acc_trai, var_acc_trai = update_avg_and_var(
            train_set_accuracies, avg_acc_trai, var_acc_trai, prev_num_samp)
        avg_acc_test, var_acc_test = update_avg_and_var(
            test_set_accuracies, avg_acc_test, var_acc_test, prev_num_samp)
        avg_acc_quer, var_acc_quer = update_avg_and_var(
            query_set_accuracies, avg_acc_quer, var_acc_quer, prev_num_samp)

        # update prev_num_samp
        prev_num_samp += 1

        # average
        df_result[('average', 'train_set_accuracies')
                  ] = pd.Series(avg_acc_trai)
        df_result[('average', 'test_set_accuracies')] = pd.Series(avg_acc_test)
        df_result[('average', 'query_set_accuracies')
                  ] = pd.Series(avg_acc_quer)

        # variance
        df_result[('variance', 'train_set_accuracies')
                  ] = pd.Series(var_acc_trai)
        df_result[('variance', 'test_set_accuracies')
                  ] = pd.Series(var_acc_test)
        df_result[('variance', 'query_set_accuracies')
                  ] = pd.Series(var_acc_quer)

        # Update the MultiIndex for columns
        df_result.columns = pd.MultiIndex.from_tuples(df_result.columns)

        # save
        df_result.to_csv(csv_path, index=False)
        print(f"Results were saved at {csv_path}")


if __name__ == "__main__":
    helper("syn_100", "ours")
    helper("musk_v2", "ours")
    helper("syn_100", "passive")
    helper("musk_v2", "passive")
    helper("syn_100", "lev_score_samp")
    helper("musk_v2", "lev_score_samp")
