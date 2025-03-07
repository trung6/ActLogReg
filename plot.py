import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _plot_helper(results_path: str, start_index: int, end_index: int, plot_test: bool, method: str, color_: str = "green") -> None:
    # read off accuracies from csv file
    df = pd.read_csv(results_path,
                     header=[0, 1], skipinitialspace=True)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    assert end_index <= len(
        df["schedule"]), "end_index exceeds length of schedule"

    # count the number of runs
    unique_names = list(df.columns.get_level_values(0).unique())
    unique_names.remove('schedule')
    unique_names.remove('average')
    unique_names.remove('variance')
    num_runs_ = len(unique_names)
    if method == "ACED": num_runs_ = 5

    plt.errorbar(df["schedule"][start_index:end_index],
                 df[('average', 'train_set_accuracies')
                    ][start_index:end_index].astype(float),
                 label=f"train_set_accuracies {method} (average over {num_runs_} runs)",
                 color=color_,
                 linestyle='dashed',
                 capsize=5.0,
                 errorevery=3,
                 yerr=np.sqrt(df[('variance', 'train_set_accuracies')][start_index:end_index].astype(float)) / np.sqrt(num_runs_))
    if plot_test:
        plt.errorbar(df["schedule"][start_index:end_index],
                     df[('average', 'test_set_accuracies')
                        ][start_index:end_index].astype(float),
                     label=f"test_set_accuracies {method} (average over {num_runs_} runs)",
                     color=color_,
                     capsize=5.0,
                     errorevery=3,
                     yerr=np.sqrt(df[('variance', 'test_set_accuracies')][start_index:end_index].astype(float)) / np.sqrt(num_runs_))
    return


# syn_100 plot
dataset = "syn_100"
start_index = 0
end_index = 19
plot_test = True

# plot accuracies of ground truth model
plt.axhline(y=0.8415, color="black", linestyle='dashed',
            label=f"train_set_accuracy of the ground truth model")
plt.axhline(y=0.8429, color="black",
            label=f"test_set_accuracy of the ground truth model")

# plot accuracies of different methods
_plot_helper(results_path=f"outputs/ours/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="OURS", color_="red")

_plot_helper(results_path=f"outputs/passive/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="PASS", color_="green")

_plot_helper(results_path=f"outputs/aced/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="ACED", color_="blue")

_plot_helper(results_path=f"outputs/lev_score_samp/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="LSS", color_="orange")


plt.xlabel('Number of queries')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.title(f"Compare OURS, PASS, LSS and ACED on the {dataset} dataset")
plt.savefig(
    f"outputs/plots/compare_ours_passive_lss_aced_on_{dataset}.svg", bbox_inches='tight')
plt.close("all")


# musk_v2 plot
dataset = "musk_v2"
start_index = 0
end_index = 70
plot_test = True

_plot_helper(results_path=f"outputs/ours/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="OURS", color_="red")

_plot_helper(results_path=f"outputs/passive/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="PASS", color_="green")

_plot_helper(results_path=f"outputs/aced/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="ACED", color_="blue")

_plot_helper(results_path=f"outputs/lev_score_samp/{dataset}/results.csv", start_index=start_index,
             end_index=end_index, plot_test=plot_test, method="LSS", color_="orange")

plt.xlabel('Number of queries')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.title(f"Compare OURS, PASS, LSS and ACED on the {dataset} dataset")
plt.savefig(
    f"outputs/plots/compare_ours_passive_lss_aced_on_{dataset}.svg", bbox_inches='tight')
plt.close("all")
