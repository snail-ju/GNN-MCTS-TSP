import numpy as np
from scipy.spatial import distance_matrix


def _calc_insert_cost(D, prv, nxt, ins):
    """
    Calculates insertion costs of inserting ins between prv and nxt
    :param D: distance matrix
    :param prv: node before inserted node, can be vector
    :param nxt: node after inserted node, can be vector
    :param ins: node to insert
    :return:
    """
    return (
            D[prv, ins]
            + D[ins, nxt]
            - D[prv, nxt]
    )


def run_insertion(loc, method, partial_tour):
    n = len(loc)
    D = distance_matrix(loc, loc)

    mask = np.zeros(n, dtype=bool)
    mask[partial_tour] = True
    tour = partial_tour  # np.empty((0, ), dtype=int)
    for i in range(len(partial_tour) + 1, n):
        feas = mask == 0
        feas_ind = np.flatnonzero(mask == 0)

        if method == 'random':
            # Order of instance is random so do in order for deterministic results
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0  # order does not matter so first is random
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()]  # node nearest to any in tour
        elif method == 'cheapest':
            assert False, "Not yet implemented"  # try all and find cheapest insertion cost

        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()  # Node with farthest distance to any other node
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]  # node which has closest node in tour farthest
        mask[a] = True

        if len(tour) == 0:
            tour = [a]
        else:
            # Find index with least insert cost
            ind_insert = np.argmin(
                _calc_insert_cost(
                    D,
                    tour,
                    np.roll(tour, -1),
                    a
                )
            )
            tour.insert(ind_insert + 1, a)

    cost = D[tour, np.roll(tour, -1)].sum()
    # return cost, tour
    return cost
