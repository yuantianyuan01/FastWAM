from itertools import chain

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line


def make_distance_transform(grid, grid_spacing=None, max_distance_to_obstacle=0.75):
    pad = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=grid.dtype)
    pad[1:-1, 1:-1] = grid
    dt = distance_transform_edt(pad)[1:-1, 1:-1]
    if grid_spacing is None:
        return dt
    assert grid_spacing >= 0, f"grid_spacing should be positive, if given (got {grid_spacing})"
    return np.clip(dt, 0.0, max_distance_to_obstacle / grid_spacing)


def cost_function(x, weight_exp, zeroish=1e-30):
    return np.power(np.maximum(x, zeroish), -weight_exp)


def get_pixel_cost(cost, p):
    return cost[round(p[0]), round(p[1])]


def get_segment_cost(cost, p1, p2):
    seg_cost = cost[line(round(p1[0]), round(p1[1]), round(p2[0]), round(p2[1]))]
    return seg_cost.sum(), seg_cost.max()


def make_grid_graph(
    grid,
    dt,
    weight_exp=2,
):
    def make_direction_edges(s0: tuple[slice, slice], s1: tuple[slice, slice]):
        locs = np.nonzero(grid[s0] * grid[s1])
        ws = np.maximum(cost[s0], cost[s1])[locs]
        labs0 = labels[s0][locs]
        labs1 = labels[s1][locs]
        return map(
            lambda labsw: (tuple(labsw[0]), tuple(labsw[1]), labsw[2]), zip(labs0, labs1, ws)
        )

    cost = cost_function(dt, weight_exp)
    labels = np.stack(
        np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij"), axis=2
    )
    edges = chain(
        make_direction_edges((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        make_direction_edges((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    )
    g = nx.Graph()
    g.add_weighted_edges_from(edges)

    return g


def simplify_path_greedy(waypoints, dt, weight_exp, grid_spacing, max_distance_to_obstacle):
    if len(waypoints) == 0:
        return [], np.inf

    cost = cost_function(dt, weight_exp)
    if weight_exp == 0:
        # Make obstacles effectively non-visitable
        cost[dt == 0] = 1e30

    if len(waypoints) == 1:
        return [waypoints[0]], get_pixel_cost(cost, waypoints[0])

    if len(waypoints) == 2:
        return list(waypoints), get_segment_cost(cost, waypoints[0], waypoints[1])

    # If shortcutting a section of the path doesn't increase the cost, that's what we should do!
    #  obviously, 4-connectivity won't bring us there (8-connectivity should? - but expensive)

    acceptable_cost = cost_function(max_distance_to_obstacle / grid_spacing, weight_exp)

    res = [waypoints[0]]
    path_cost = 0.0

    def add_to_path():
        ret = cur_cost - get_pixel_cost(cost, res[-1])
        res.append(cur_next)
        return ret

    cur_next = waypoints[1]
    cur_cost, cur_peak = get_segment_cost(cost, res[-1], cur_next)
    for it in range(2, len(waypoints)):
        pos_next = waypoints[it]

        # We add extra cost to each last step in each segment, which is incorrect, but works okay
        pos_cost, pos_peak = get_segment_cost(cost, res[-1], pos_next)
        pos_local_cost, local_peak = get_segment_cost(cost, cur_next, pos_next)

        if pos_cost <= cur_cost + pos_local_cost and (
            pos_peak <= max(cur_peak, local_peak) or pos_peak < acceptable_cost
        ):
            cur_cost = pos_cost
            cur_peak = pos_peak
        else:
            path_cost += add_to_path()

            cur_cost = pos_local_cost
            cur_peak = local_peak

        cur_next = pos_next

    path_cost += add_to_path()

    return res, path_cost


def make_discrete_path(
    graph,
    source_row,
    source_col,
    target_row,
    target_col,
    distance_transform,
    weight_exp,
    grid_spacing,
    max_distance_to_obstacle,
):
    locs = nx.astar_path(graph, (source_row, source_col), (target_row, target_col))
    waypoints, path_cost = simplify_path_greedy(
        locs, distance_transform, weight_exp, grid_spacing, max_distance_to_obstacle
    )
    # print(f"{len(locs)} locs to {len(waypoints)} waypoints")
    return waypoints, locs, path_cost
