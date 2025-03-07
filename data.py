import numpy as np
try:
    from ucimlrepo import fetch_ucirepo
except:
    print("Did not load ucimlrepo")
from sklearn.model_selection import train_test_split


def prep_data(train_data: np.ndarray, train_labels: np.ndarray,
              test_data: np.ndarray, test_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # preprocess data
    max_norm_train_data = np.max(np.linalg.norm(train_data, ord=2, axis=1))
    train_data = train_data / max_norm_train_data
    test_data = test_data / max_norm_train_data
    train_labels = train_labels * 2.0 - 1.0  # change labels from 0/1 to -1/1
    test_labels = test_labels * 2.0 - 1.0
    print("Preprocessed data #REQUIRE FOR BOTH TRAINING AND EVALUATING")
    return train_data, train_labels, test_data, test_labels


def create_syn_100() -> None:
    save_path = "./data/syn_100.npz"
    rng = np.random.default_rng()
    # Generate a random 10000x100-dimensional matrix
    n_train = 10000
    n_test = 10000
    train_data = rng.uniform(-1., 1., (n_train, 100))
    test_data = rng.uniform(-1., 1., (n_test, 100))
    weight = rng.uniform(-1., 1., 100)
    bias = rng.uniform(-1., 1., 1)
    train_data_probs = 1. / (1. + np.exp(- train_data @ weight + bias))
    train_labels = rng.binomial(n=1, p=train_data_probs)
    test_data_probs = 1. / (1. + np.exp(- test_data @ weight + bias))
    test_labels = rng.binomial(n=1, p=test_data_probs)
    np.savez(save_path, train_data=train_data, train_labels=train_labels,
             test_data=test_data, test_labels=test_labels, weight=weight, bias=bias)
    print("Saved synthetic dataset at: ", save_path)


def get_syn_100_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npzfile = np.load("./data/syn_100.npz")
    train_data, train_labels, test_data, test_labels = npzfile["train_data"], npzfile[
        "train_labels"], npzfile["test_data"], npzfile["test_labels"]
    print(f"Loaded dataset: Synthetic - {train_data.shape[1]} dim")
    print("Number of training points: ", train_data.shape[0])
    print("Number of testing points: ", test_data.shape[0])
    return train_data, train_labels, test_data, test_labels


def create_toy_syn_100() -> None:
    save_path = "./data/toy_syn_100.npz"
    rng = np.random.default_rng()
    # Generate a random 40x100-dimensional matrix
    n_train = 40
    n_test = 40
    train_data = rng.uniform(-1., 1., (n_train, 100))
    test_data = rng.uniform(-1., 1., (n_test, 100))
    weight = rng.uniform(-1., 1., 100)
    bias = rng.uniform(-1., 1., 1)
    train_data_probs = 1. / (1. + np.exp(- train_data @ weight + bias))
    train_labels = rng.binomial(n=1, p=train_data_probs)
    test_data_probs = 1. / (1. + np.exp(- test_data @ weight + bias))
    test_labels = rng.binomial(n=1, p=test_data_probs)
    np.savez(save_path, train_data=train_data, train_labels=train_labels,
             test_data=test_data, test_labels=test_labels, weight=weight, bias=bias)


def get_toy_syn_100_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npzfile = np.load("./data/toy_syn_100.npz")
    train_data, train_labels, test_data, test_labels = npzfile["train_data"], npzfile[
        "train_labels"], npzfile["test_data"], npzfile["test_labels"]
    print(f"Loaded dataset: Synthetic - {train_data.shape[1]} dim")
    print("Number of training points: ", train_data.shape[0])
    print("Number of testing points: ", test_data.shape[0])
    return train_data, train_labels, test_data, test_labels


def create_syn_100_gaussian() -> None:
    save_path = "./data/syn_100_gaussian.npz"
    rng = np.random.default_rng()
    # Generate a random 10000x100-dimensional matrix
    n_train = 10000
    n_test = 10000
    loc = 2.0
    scale = 5.0
    train_data = rng.normal(loc, scale, (n_train, 100))
    test_data = rng.normal(loc, scale, (n_test, 100))
    weight = rng.uniform(-1., 1., 100)
    bias = rng.uniform(-1., 1., 1)
    train_data_probs = 1. / (1. + np.exp(- train_data @ weight + bias))
    train_labels = rng.binomial(n=1, p=train_data_probs)
    test_data_probs = 1. / (1. + np.exp(- test_data @ weight + bias))
    test_labels = rng.binomial(n=1, p=test_data_probs)
    np.savez(save_path, train_data=train_data, train_labels=train_labels,
             test_data=test_data, test_labels=test_labels, weight=weight, bias=bias)


def get_syn_100_gaussian_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npzfile = np.load("./data/syn_100_gaussian.npz")
    train_data, train_labels, test_data, test_labels = npzfile["train_data"], npzfile[
        "train_labels"], npzfile["test_data"], npzfile["test_labels"]
    print(f"Loaded dataset: Synthetic - Gaussian - {train_data.shape[1]} dim")
    print("Number of training points: ", train_data.shape[0])
    print("Number of testing points: ", test_data.shape[0])
    return train_data, train_labels, test_data, test_labels


def create_musk_v2() -> None:
    """
    Create train and test sets from UCI musk v2 dataset (dataset id = 75)
    """
    # fetch dataset
    musk_version_2 = fetch_ucirepo(id=75)

    # data (as pandas dataframes)
    X = musk_version_2.data.features
    y = musk_version_2.data.targets

    # metadata
    print(musk_version_2.metadata)

    # variable information
    print(musk_version_2.variables)

    # create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values.squeeze(), test_size=0.33, random_state=42)
    np.savez(f"./data/musk_v2_166.npz", train_data=X_train,
             train_labels=y_train, test_data=X_test, test_labels=y_test)
    print("Saved Musk v2 dataset at: ./data/musk_v2_166.npz")
    return


def get_musk_v2_np() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npzfile = np.load("./data/musk_v2_166.npz")
    train_data, train_labels, test_data, test_labels = npzfile["train_data"], npzfile[
        "train_labels"], npzfile["test_data"], npzfile["test_labels"]
    print(f"Loaded dataset: Musk v2 - {train_data.shape[1]} dim")
    print("Number of training points: ", train_data.shape[0])
    print("Number of testing points: ", test_data.shape[0])
    return train_data, train_labels, test_data, test_labels


class Dataloader:
    @staticmethod
    def load_syn_100() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, train_labels, test_data, test_labels = get_syn_100_np()
        train_data, train_labels, test_data, test_labels = prep_data(train_data=train_data,
                                                                     train_labels=train_labels,
                                                                     test_data=test_data,
                                                                     test_labels=test_labels,)
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_musk_v2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, train_labels, test_data, test_labels = get_musk_v2_np()
        train_data, train_labels, test_data, test_labels = prep_data(train_data=train_data,
                                                                     train_labels=train_labels,
                                                                     test_data=test_data,
                                                                     test_labels=test_labels,)
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_syn_100_gaussian() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_data, train_labels, test_data, test_labels = get_syn_100_gaussian_np()
        train_data, train_labels, test_data, test_labels = prep_data(train_data=train_data,
                                                                     train_labels=train_labels,
                                                                     test_data=test_data,
                                                                     test_labels=test_labels,)
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_data(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if dataset == "syn_100":
            train_data, train_labels, test_data, test_labels = Dataloader.load_syn_100()
        elif dataset == "musk_v2":
            train_data, train_labels, test_data, test_labels = Dataloader.load_musk_v2()
        elif dataset == "syn_100_gaussian":
            train_data, train_labels, test_data, test_labels = Dataloader.load_syn_100_gaussian()
        else:
            raise Exception(f"Dataset {dataset} is not supported.")
        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def load_queries_from_file(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        npzfile = np.load(file_path)
        return npzfile["query_data"], npzfile["query_labels"]


if __name__ == "__main__":
    create_musk_v2()
    create_syn_100()
