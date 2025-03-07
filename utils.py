import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import signal
from typing import Literal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Evaluator:
    def __init__(self, schedule: list[int] | range):
        self.score_function = accuracy_score
        self.schedule = schedule
        REGULATION_STRENGTHS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-8, 1e-15]

        # Set up the parameter grid for hyperparameter tuning with valid combinations
        self.param_grid = [
            {'penalty': ["l2"],
             'C': [1./reg for reg in REGULATION_STRENGTHS],
             'solver': ["lbfgs", "liblinear"],
             'max_iter': [int(1e9),],
             'tol': [.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,],
             'fit_intercept': [True],
             },
        ]
        self.clf = LogisticRegression(
            max_iter=200, class_weight=None)
        self.time_limit = 300  # seconds

    def eval(self,
             query_data: np.ndarray,
             query_labels: np.ndarray,
             train_data: np.ndarray,
             train_labels: np.ndarray,
             test_data: np.ndarray,
             test_labels: np.ndarray,
             gt_weight: np.ndarray | None = None,
             gt_bias: np.ndarray | float | None = None,
             ) -> tuple[list[float], list[float], list[float], list[float]]:
        l2_to_gt = []  # l2 distance between best estimator to ground truth in case of synthetic data
        train_set_accuracies = []
        test_set_accuracies = []
        query_set_accuracies = []
        total_queries = query_data.shape[0]

        # Set the signal handler
        signal.signal(signal.SIGALRM, timeout_handler)

        for num_que in tqdm(self.schedule, "#num_queries gone through"):
            if num_que > total_queries:
                break
            tmp_query_data = query_data[:num_que]
            tmp_query_labels = query_labels[:num_que]
            if (tmp_query_labels.astype(int) == 1).all():
                # All labels are 1.
                tmp_y_pred_train = np.ones(train_labels.shape)
                tmp_y_pred_test = np.ones(test_labels.shape)
                tmp_y_pred_quer = np.ones(tmp_query_labels.shape)
            elif (tmp_query_labels.astype(int) == -1).all():
                # All labels are -1.
                tmp_y_pred_train = np.negative(np.ones(train_labels.shape))
                tmp_y_pred_test = np.negative(np.ones(test_labels.shape))
                tmp_y_pred_quer = np.negative(np.ones(tmp_query_labels.shape))
            else:
                # Mixed labels, train.
                cv_: int = min(5, int(num_que / 2))
                grid_search = GridSearchCV(
                    estimator=self.clf, param_grid=self.param_grid, cv=cv_, scoring="accuracy", n_jobs=-1, verbose=2)
                try:
                    # Set the alarm
                    signal.alarm(self.time_limit)
                    grid_search.fit(tmp_query_data, tmp_query_labels)
                except TimeoutException:
                    print("Grid search timed out!")
                    query_set_accuracies.append(float("nan"))
                    train_set_accuracies.append(float("nan"))
                    test_set_accuracies.append(float("nan"))
                    continue
                finally:
                    # Disable the alarm
                    signal.alarm(0)

                best_model = grid_search.best_estimator_
                if (gt_weight is not None) and (gt_bias is not None):
                    # ground truth probability corresponding to class 1
                    gt_probs = 1. / \
                        (1. + np.exp(- train_data @ gt_weight + gt_bias))
                    # predicted probability corresponding to class 1
                    pred_probs = best_model.predict_proba(train_data)[:, -1]
                    _l2 = np.linalg.norm(
                        gt_probs - pred_probs) / np.sqrt(train_data.shape[0])
                    l2_to_gt.append(_l2)

                tmp_y_pred_train = best_model.predict(train_data)
                tmp_y_pred_test = best_model.predict(test_data)
                tmp_y_pred_quer = best_model.predict(tmp_query_data)

            # query set accuracy
            tmp_acc = self.score_function(tmp_query_labels, tmp_y_pred_quer)
            query_set_accuracies.append(tmp_acc)
            # whole train set accuracy
            tmp_acc = self.score_function(train_labels, tmp_y_pred_train)
            train_set_accuracies.append(tmp_acc)
            # test set accuracy
            tmp_acc = self.score_function(test_labels, tmp_y_pred_test)
            test_set_accuracies.append(tmp_acc)
        return train_set_accuracies, test_set_accuracies, query_set_accuracies, l2_to_gt


def compute_mean_recur(prev_num_samp: int, latest_mean: float, curr_samp: float):
    return (prev_num_samp * latest_mean + curr_samp) / (prev_num_samp + 1.)


def compute_var_recur(prev_num_samp: int, latest_mean: float, latest_var: float, curr_samp: float) -> float:
    '''
    return updated var, which is std^2
    '''
    n = prev_num_samp
    return (n-1) * latest_var / n + (curr_samp - latest_mean)**2 / (n+1)


def update_avg_and_var(curr_list: list, avg_list: list, var_list: list, prev_num_samp: int) -> tuple[list, list]:
    # prev_num_samp is at least 0
    assert prev_num_samp >= 0
    if prev_num_samp == 0:
        assert len(avg_list) == 0
        assert len(var_list) == 0
        avg_list = curr_list
        var_list = [0.] * len(avg_list)
    else:
        # three lists should have same length
        assert len(avg_list) == len(curr_list)
        assert len(var_list) == len(curr_list)
        # recursively update avg and var
        for j in range(len(avg_list)):
            last_avg = avg_list[j]
            last_var = var_list[j]
            curr_samp = curr_list[j]
            var_list[j] = compute_var_recur(
                prev_num_samp=prev_num_samp, latest_mean=last_avg, latest_var=last_var, curr_samp=curr_samp)
            avg_list[j] = compute_mean_recur(
                prev_num_samp=prev_num_samp, latest_mean=last_avg, curr_samp=curr_samp)
    return avg_list, var_list


def gt_acc_syn_100():
    npzfile = np.load("./data/syn_100.npz")
    gt_weight = npzfile["weight"]
    gt_bias = npzfile["bias"]
    gt_model = LogisticRegression()
    gt_model.coef_ = gt_weight.reshape([1, -1])
    gt_model.intercept_ = gt_bias.reshape([1,])
    gt_model.classes_ = np.array([0, 1], dtype=int)
    train_data, train_labels, test_data, test_labels = npzfile["train_data"], npzfile[
        "train_labels"], npzfile["test_data"], npzfile["test_labels"]
    train_labels_pred = gt_model.predict(train_data)
    test_labels_pred = gt_model.predict(test_data)
    train_acc = accuracy_score(train_labels, train_labels_pred)
    test_acc = accuracy_score(test_labels, test_labels_pred)
    print("train accuracy of ground truth model in syn_100: ", train_acc)
    print("test accuracy of ground truth model in syn_100: ", test_acc)


if __name__ == "__main__":
    gt_acc_syn_100()
