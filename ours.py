import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm  # for progress bar
import numpy as np
import os
import argparse
from data import Dataloader
import sys


def get_data_tf(dataset: str):
    dl = Dataloader()
    _train_data, _train_labels, _test_data, _test_labels = dl.load_data(
        dataset=dataset)
    train_data = tf.convert_to_tensor(_train_data, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(_train_labels, dtype=tf.float32)
    test_data = tf.convert_to_tensor(_test_data, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(_test_labels, dtype=tf.float32)
    return train_data, train_labels, test_data, test_labels


def get_que_numpy(queries_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npzfile = np.load(queries_path)  # queries_path is a npz file
    query_data, query_labels, query_indices = npzfile[
        "query_data"], npzfile["query_labels"], npzfile["query_indices"]
    return query_data, query_labels, query_indices


def get_que_tf(queries_path: str) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    query_data_np, query_labels_np, query_indices_np = get_que_numpy(
        queries_path)
    query_data = tf.convert_to_tensor(query_data_np, dtype=tf.float32)
    query_labels = tf.convert_to_tensor(query_labels_np, dtype=tf.float32)
    query_indices = tf.convert_to_tensor(query_indices_np, dtype=tf.int32)
    return query_data, query_labels, query_indices


def f(W, X, Y, lammy):  # unnormalized log prob
    temp = tf.linalg.matmul(tf.linalg.diag(-Y), X)
    temp = tf.linalg.matmul(W, tf.transpose(temp))
    temp = tf.math.log(1. + tf.math.exp(temp))
    temp = tf.math.reduce_sum(temp, axis=1)
    return -temp-lammy*tf.norm(W, axis=1)


def fUniform(W, lammy):  # unnormalized log prob at iteration 0
    return -lammy*tf.norm(W, axis=1)


def run_mcmc(init_state: tf.Tensor,
             X: None | tf.Tensor,
             Y: None | tf.Tensor,
             firstIter: bool,
             lammy: float = 0.5,
             num_results_each_chain: int = 300,
             step_size: float = 1e-3,
             burning_steps: int = 25000):
    if firstIter == True:
        def target_fn(x): return fUniform(x, lammy)
    else:
        def target_fn(x): return f(x, X, Y, lammy)
    pass

    kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_fn,
        step_size=step_size,
    )

    states = tfp.mcmc.sample_chain(
        num_results=num_results_each_chain,
        current_state=init_state,
        kernel=kernel,
        num_burnin_steps=burning_steps,
        num_steps_between_results=2,
        trace_fn=None,
        seed=None)
    return states


def simple_kl(p: tf.Tensor,
              q: tf.Tensor,
              clip_val: float = 100):
    '''
        p: Bernoulli(p)
        q: Bernoulli(q)
    '''
    one_minus_p = tf.subtract(1., p)
    one_minus_q = tf.subtract(1., q)
    log_p_over_q = tf.math.log(tf.divide(p, q))
    log_one_minu_p_over_one_minu_q = tf.math.log(
        tf.divide(one_minus_p, one_minus_q))
    clip_log_p_over_q = tf.minimum(log_p_over_q, clip_val)
    clip_log_one_minu_p_over_one_minu_q = tf.minimum(
        log_one_minu_p_over_one_minu_q, clip_val)
    return p * clip_log_p_over_q + one_minus_p * clip_log_one_minu_p_over_one_minu_q


def samp_hypo_mcmc(query_data: tf.Tensor | None,
                   query_labels: tf.Tensor | None,
                   init_state: tf.Tensor,
                   firstIter: bool,
                   Lammy: float = 0.5,
                   num_chains: int = 100,
                   burning_steps: int = 25000,
                   num_results_each_chain: int = 300,
                   step_size: float = 1e-3
                   ) -> tuple[tf.Tensor, tf.Tensor]:
    states = run_mcmc(init_state, query_data, query_labels, firstIter=firstIter, lammy=Lammy,
                      num_results_each_chain=num_results_each_chain,
                      burning_steps=burning_steps, step_size=step_size)
    # samples is a matrix (2d tensor) shape [n, d+1]
    samples = tf.reshape(states, [num_results_each_chain*num_chains, -1])
    weights = samples[:, :-1]  # shape [n, d] -> n rows d columns.
    biases = samples[:, -1]  # shape [n] -> n rows -> column vector
    return weights, biases


def comp_que_crit(weights: tf.Tensor,
                  biases: tf.Tensor,
                  num_data_pt: int | tf.TensorShape,
                  num_samp_hypo: int | tf.TensorShape,
                  train_data: tf.Tensor,
                  clip_val: float = 100
                  ) -> tf.Tensor:
    # shape [n, m] where n = num_samp_hypo, m = num_data_pt
    w_dot_data = tf.matmul(weights, tf.transpose(train_data))
    probs = tf.math.sigmoid(w_dot_data + tf.transpose(tf.broadcast_to(
        tf.transpose(biases), [num_data_pt, num_samp_hypo])))  # shape [n,m]
    mean_probs = tf.reduce_mean(probs, 0)  # shape [m]
    bc_mean_probs = tf.broadcast_to(mean_probs, probs.shape)  # shape [n, m]
    all_kl = simple_kl(bc_mean_probs, probs, clip_val=clip_val)  # shape [n, m]
    return all_kl


def query(all_kl: tf.Tensor,
          train_data: tf.Tensor,
          train_labels: tf.Tensor,
          seed: tuple[tf.int32, tf.int32],
          topK: int = 1,
          ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    query_criteria = tf.reduce_mean(all_kl, 0)  # shape [m]
    # query from topK largest values
    # pick topK query candidates
    que_candidates = tf.math.top_k(query_criteria, k=topK)
    # randomly pick a candidate
    cand_index = tf.random.stateless_uniform(
        shape=[], seed=seed, minval=0, maxval=topK, dtype=tf.int32)
    query_index = que_candidates.indices[cand_index]
    curr_query_data = tf.expand_dims(train_data[query_index, :], 0)
    curr_query_labels = tf.expand_dims(train_labels[query_index], 0)
    # to make sure query_index has shape > 0.
    return curr_query_data, curr_query_labels, tf.expand_dims(query_index, 0)


@tf.function(jit_compile=True)
def train(init_que_data: None | tf.Tensor,
          init_que_labels: None | tf.Tensor,
          init_que_indices: None | tf.Tensor,
          train_data: tf.Tensor,
          train_labels: tf.Tensor,
          seed: tuple[tf.int32, tf.int32] = (2, 3),
          num_iters: int = 20,
          Lammy: float = 0.5,
          num_chains: int = 100,
          burning_steps: int = 25000,
          num_results_each_chain: int = 300,
          step_size: float = 1e-3,
          clip_val: float = 100,
          ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dim_data_point = train_data.shape[1]
    dim_data_point = tf.add(dim_data_point, 1)  # add additional pseudo-dim
    num_samp_hypo = num_results_each_chain*num_chains
    num_data_pt = train_data.shape[0]

    init_state = tf.ones([num_chains, dim_data_point], dtype=tf.float32)
    query_data = init_que_data
    query_labels = init_que_labels
    query_indices = init_que_indices
    for i in tqdm(range(num_iters)):
        if (not tf.is_tensor(query_data)) & (i == 0):
            # first iteration
            # sample hypotheses
            weights, biases = samp_hypo_mcmc(None, None, init_state, True, Lammy,
                                             num_chains, burning_steps, num_results_each_chain, step_size)  # with bias
            # compute r_bar
            all_kl = comp_que_crit(
                weights, biases, num_data_pt, num_samp_hypo, train_data, clip_val=clip_val)  # with bias
            # query
            query_data, query_labels, query_indices = query(
                all_kl, train_data, train_labels, seed=seed)
        else:
            # for each subsequent iteration:
            # sample hypotheses
            padd_query_data: tf.Tensor = tf.concat(
                [query_data, tf.ones([query_data.shape[0], 1])], 1)
            weights, biases = samp_hypo_mcmc(padd_query_data, query_labels, init_state, False, Lammy,
                                             num_chains, burning_steps, num_results_each_chain, step_size)  # with bias
            # compute r_bar
            all_kl = comp_que_crit(weights, biases, num_data_pt, num_samp_hypo, train_data,
                                   clip_val=clip_val,
                                   )
            # query
            curr_query_data, curr_query_labels, curr_query_index = query(
                all_kl, train_data, train_labels, seed=seed)
            # add query point to the set of queries
            query_data = tf.concat([query_data, curr_query_data], axis=0)
            query_labels = tf.concat([query_labels, curr_query_labels], axis=0)
            query_indices = tf.concat(
                [query_indices, curr_query_index], axis=0)
        pass
    return query_data, query_labels, query_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="syn_100",
                        help="dataset: syn_100, musk_v2", )
    parser.add_argument("--nque", type=int, default=20,
                        help="total number of queries",)
    parser.add_argument("--ord", type=int, default=0,
                        help="current run")
    parser.add_argument("--seed", nargs='+', type=int, default=(2, 3),
                        help="2-value seed",)
    args = parser.parse_args()
    dataset = args.data
    num_active_queries = args.nque
    curr_run = args.ord
    seed = args.seed
    # get data
    train_data, train_labels, _, _ = get_data_tf(dataset)
    # load queries
    queries_path = "outputs/ours/{}/run_{}.npz".format(dataset, curr_run)
    print("queries path: " + queries_path)
    if os.path.exists(queries_path):
        init_que_data, init_que_labels, init_query_indices = get_que_tf(
            queries_path)
        print("Loaded {} queries from the queries path".format(
            init_que_data.shape[0]))
    else:
        init_que_data, init_que_labels, init_query_indices = None, None, None
        print("No queries path found.")
        print("Loaded 0 queries from the queries path.")

    # train
    query_data, query_labels, query_indices = train(init_que_data,
                                                    init_que_labels,
                                                    init_query_indices,
                                                    train_data,
                                                    train_labels,
                                                    num_iters=num_active_queries,
                                                    seed=seed,)
    # save checkpoint of the query set
    try:
        np.savez(queries_path, query_data=query_data.numpy(),
                 query_labels=query_labels.numpy(),
                 query_indices=query_indices.numpy())
        print("Saved {} data points, labels, and their indices in the train set into {}".format(
            query_data.shape[0], queries_path))
    except:
        print(f"file {queries_path} is corrupted.")
        sys.exit(1)
