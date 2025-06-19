import warnings

import numpy as np
import numpy.typing as npt


def split_indices(
    num_cumsum: npt.NDArray, rand_perm: npt.NDArray[np.long]
) -> dict[int, npt.NDArray[np.long]]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    client_indices_pairs = [
        (cid, idxs) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])
    ]
    client_dict = dict(client_indices_pairs)
    return client_dict


def balance_split(num_clients: int, num_samples: int) -> npt.NDArray[np.int_]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    return client_sample_nums


def shards_partition(
    targets, num_clients: int, num_shards: int
) -> dict[int, npt.NDArray[np.int_]]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    size_shard = int(num_samples / num_shards)
    if num_samples % num_shards != 0:
        warnings.warn(
            "warning: length of dataset isn't divided exactly by num_shards. "
            "Some samples will be dropped.",
            stacklevel=1,
        )

    shards_per_client = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shards isn't divided exactly by num_clients. "
            "Some shards will be dropped.",
            stacklevel=1,
        )

    indices = np.arange(num_samples)
    # sort sample indices according to labels
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    sorted_indices = indices_targets[0, :]

    # permute shards idx, and slice shards_per_client shards for each client
    rand_perm = np.random.permutation(num_shards)
    num_client_shards = np.ones(num_clients) * shards_per_client
    # sample index must be int
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    # shard indices for each client
    client_shards_dict = split_indices(num_cumsum, rand_perm)

    # map shard idx to sample idx for each client
    client_dict = dict()
    for cid in range(num_clients):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * size_shard : (shard_id + 1) * size_shard]
            for shard_id in shards_set
        ]
        client_dict[cid] = np.concatenate(current_indices, axis=0)

    return client_dict


def client_inner_dirichlet_partition_faster(
    targets: list[int] | npt.NDArray[np.int_],
    num_clients: int,
    num_classes: int,
    dir_alpha: float,
    client_sample_nums: npt.NDArray[np.int_],
    verbose: bool = True,
) -> dict[int, npt.NDArray[np.int_]]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    class_priors = np.random.dirichlet(
        alpha=[dir_alpha] * num_classes, size=num_clients
    )
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print(f"Remaining Data: {format(np.sum(client_sample_nums))}")
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = int(np.argmax(np.random.uniform() <= curr_prior))
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                # Exception handling: If the current class has no samples left,
                # randomly select a non-zero class
                while True:
                    new_class = np.random.randint(num_classes)
                    if class_amount[new_class] > 0:
                        curr_class = new_class
                        break
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict
