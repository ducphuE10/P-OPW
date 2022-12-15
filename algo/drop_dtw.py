
import torch
import numpy as np
from os import path as osp
import ot

import torch.nn.functional as F


def compute_all_costs(x1, x2, gamma=1, keep_percentile=0.3, l2_nomalize=False):
    #convert x1, x2 to torch tensor
    if not isinstance(x1, torch.Tensor):
        x1 = torch.from_numpy(x1).float()
    if not isinstance(x2, torch.Tensor):
        x2 = torch.from_numpy(x2).float()

    # import ipdb; ipdb.set_trace()
    cost = ot.dist(x1, x2, metric='euclidean')
    cost = cost / cost.max()
    sim = 1 - cost

    k = int(torch.numel(sim) * keep_percentile)
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    # baseline_logits = baseline_logit.repeat([1, sim.shape[1]])
    sims_ext = torch.zeros([sim.shape[0]+1, sim.shape[1]+1]).to(sim.device)
    sims_ext[:sim.shape[0], :sim.shape[1]] = sim
    sims_ext[sim.shape[0], :] = baseline_logit
    sims_ext[:, sim.shape[1]] = baseline_logit

    softmax_sims = torch.nn.functional.softmax(sims_ext / gamma)
    softmax_sim, x1_drop_probs, x2_drop_probs = softmax_sims[:-1, :-1], softmax_sims[:-1, -1], softmax_sims[-1, :-1]
    zx_costs = -torch.log(softmax_sim + 1e-5)
    x1_drop_costs = -torch.log(x1_drop_probs + 1e-5)
    x2_drop_costs = -torch.log(x2_drop_probs + 1e-5)

    #convert to numpy
    zx_costs = zx_costs.cpu().detach().numpy()
    x1_drop_costs = x1_drop_costs.cpu().detach().numpy()
    x2_drop_costs = x2_drop_costs.cpu().detach().numpy()
    return zx_costs, x1_drop_costs, x2_drop_costs


def double_drop_dtw(
    pairwise_zx_costs,
    x_drop_costs,
    z_drop_costs,
    contiguous=True,
    one_to_many=True,
    many_to_one=True,
    return_labels=False,
):
    """Drop-DTW algorithm that allows drops from both sequences. See Algorithm 1 in Appendix.

    Parameters
    ----------
    pairwise_zx_costs: np.ndarray [K, N]
        pairwise match costs between K steps and N video clips
    x_drop_costs: np.ndarray [N]
        drop costs for each clip
    z_drop_costs: np.ndarray [N]
        drop costs for each step
    contiguous: bool
        if True, can only match a contiguous sequence of clips to a step
        (i.e. no drops in between the clips)
    """
    K, N = pairwise_zx_costs.shape

    # initialize solution matrices
    D = np.zeros([K + 1, N + 1, 4])  # the 4 dimensions are the following states: zx, z-, -x, --
    # no drops allowed in zx DP. Setting the same for all DPs to change later here.
    D[1:, 0, :] = 99999999
    D[0, 1:, :] = 99999999
    D[0, 0, 1:] = 99999999
    # Allow to drop x in z- and --
    D[0, 1:, 1], D[0, 1:, 3] = np.cumsum(x_drop_costs), np.cumsum(x_drop_costs)
    # Allow to drop z in -x and --
    D[1:, 0, 2], D[1:, 0, 3] = np.cumsum(z_drop_costs), np.cumsum(z_drop_costs)

    # initialize path tracking info for each of the 4 DP tables:
    P = np.zeros([K + 1, N + 1, 4, 3], dtype=int)  # (zi, xi, prev_state)
    for zi in range(1, K + 1):
        P[zi, 0, 2], P[zi, 0, 3] = (zi - 1, 0, 2), (zi - 1, 0, 3)
    for xi in range(1, N + 1):
        P[0, xi, 1], P[0, xi, 3] = (0, xi - 1, 1), (0, xi - 1, 3)

    # filling in the dynamic tables
    for zi in range(1, K + 1):
        for xi in range(1, N + 1):
            # define frequently met neighbors here
            diag_neigh_states = [0, 1, 2, 3]  # zx, z-, -x, --
            diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
            diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]

            left_pos_neigh_states = [0, 1]  # zx and z-
            left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
            left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]

            top_pos_neigh_states = [0, 2]  # zx and -x
            top_pos_neigh_coords = [(zi - 1, xi) for _ in top_pos_neigh_states]
            top_pos_neigh_costs = [D[zi - 1, xi, s] for s in top_pos_neigh_states]

            left_neg_neigh_states = [2, 3]  # -x and --
            left_neg_neigh_coords = [(zi, xi - 1) for _ in left_neg_neigh_states]
            left_neg_neigh_costs = [D[zi, xi - 1, s] for s in left_neg_neigh_states]

            top_neg_neigh_states = [1, 3]  # z- and --
            top_neg_neigh_coords = [(zi - 1, xi) for _ in top_neg_neigh_states]
            top_neg_neigh_costs = [D[zi - 1, xi, s] for s in top_neg_neigh_states]

            z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1

            # DP 0: coming to zx
            neigh_states_zx = diag_neigh_states
            neigh_coords_zx = diag_neigh_coords
            neigh_costs_zx = diag_neigh_costs
            if one_to_many:
                if contiguous:
                    neigh_states_zx.extend(left_pos_neigh_states[0:1])
                    neigh_coords_zx.extend(left_pos_neigh_coords[0:1])
                    neigh_costs_zx.extend(left_pos_neigh_costs[0:1])
                else:
                    neigh_states_zx.extend(left_pos_neigh_states)
                    neigh_coords_zx.extend(left_pos_neigh_coords)
                    neigh_costs_zx.extend(left_pos_neigh_costs)
            if many_to_one:
                neigh_states_zx.extend(top_pos_neigh_states)
                neigh_coords_zx.extend(top_pos_neigh_coords)
                neigh_costs_zx.extend(top_pos_neigh_costs)

            costs_zx = np.array(neigh_costs_zx) + pairwise_zx_costs[z_cost_ind, x_cost_ind]
            opt_ind_zx = np.argmin(costs_zx)
            P[zi, xi, 0] = *neigh_coords_zx[opt_ind_zx], neigh_states_zx[opt_ind_zx]
            D[zi, xi, 0] = costs_zx[opt_ind_zx]

            # DP 1: coming to z-
            neigh_states_z_ = left_pos_neigh_states
            neigh_coords_z_ = left_pos_neigh_coords
            neigh_costs_z_ = left_pos_neigh_costs
            costs_z_ = np.array(neigh_costs_z_) + x_drop_costs[x_cost_ind]
            opt_ind_z_ = np.argmin(costs_z_)
            P[zi, xi, 1] = *neigh_coords_z_[opt_ind_z_], neigh_states_z_[opt_ind_z_]
            D[zi, xi, 1] = costs_z_[opt_ind_z_]

            # DP 2: coming to -x
            neigh_states__x = top_pos_neigh_states
            neigh_coords__x = top_pos_neigh_coords
            neigh_costs__x = top_pos_neigh_costs
            costs__x = np.array(neigh_costs__x) + z_drop_costs[z_cost_ind]
            opt_ind__x = np.argmin(costs__x)
            P[zi, xi, 2] = *neigh_coords__x[opt_ind__x], neigh_states__x[opt_ind__x]
            D[zi, xi, 2] = costs__x[opt_ind__x]

            # DP 3: coming to --
            neigh_states___ = np.array(left_neg_neigh_states + top_neg_neigh_states)
            # adding negative left and top neighbors
            neigh_coords___ = np.array(left_neg_neigh_coords + top_neg_neigh_coords)
            costs___ = np.concatenate(
                [
                    left_neg_neigh_costs + x_drop_costs[x_cost_ind],
                    top_neg_neigh_costs + z_drop_costs[z_cost_ind],
                ],
                0,
            )

            opt_ind___ = costs___.argmin()
            P[zi, xi, 3] = *neigh_coords___[opt_ind___], neigh_states___[opt_ind___]
            D[zi, xi, 3] = costs___[opt_ind___]

    cur_state = D[K, N, :].argmin()
    min_cost = D[K, N, cur_state]

    # unroll path
    path = []
    zi, xi = K, N
    x_dropped = [N] if cur_state in [1, 3] else []
    z_dropped = [K] if cur_state in [2, 3] else []
    while not (zi == 0 and xi == 0):
        path.append((zi, xi))
        zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
        if prev_state in [1, 3]:
            x_dropped.append(xi_prev)
        if prev_state in [2, 3]:
            z_dropped.append(zi_prev)
        zi, xi, cur_state = zi_prev, xi_prev, prev_state

    if return_labels:
        labels = np.zeros(N)
        for zi, xi in path:
            if zi not in z_dropped and xi not in x_dropped:
                labels[xi - 1] = zi
        return labels
    else:
        return min_cost, path, x_dropped, z_dropped

def drop_dtw_cost(X, Y):
    xy_costs, x_drop_costs, y_drop_costs = compute_all_costs(X, Y, gamma=1, keep_percentile=0.2, l2_nomalize=False)
    result = double_drop_dtw(xy_costs, x_drop_costs, y_drop_costs)
    return result[0]