from data import Dataloader
import numpy as np

dl = Dataloader()


def query(dataset: str):
    train_data, train_labels, _, _ = dl.load_data(
        dataset=dataset)
    for i in range(10):
        passive_rand_idxs = np.random.permutation(train_data.shape[0])
        que_data = train_data[passive_rand_idxs]
        que_labels = train_labels[passive_rand_idxs]
        queries_path = f"outputs/passive/{dataset}/run_{i}.npz"
        np.savez(queries_path, query_data=que_data,
                 query_labels=que_labels,
                 query_indices=passive_rand_idxs)
        print("Saved {} data points, labels, and their indices in the train set into {}".format(
            que_data.shape[0], queries_path))


query("syn_100")
query("musk_v2")
