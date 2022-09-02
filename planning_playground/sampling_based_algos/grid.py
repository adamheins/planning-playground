from .graph_planner import GraphPlanner
from pgraph import UGraph
import numpy as np


def grid_neighbour_indices(i, j, nx, ny):
    """Get the index of the current vertex and neighbouring vertices in a 2D
       grid but stored in a 1D array.

    Parameters:
       i: x-index
       j: y-index
       nx: length of x
       ny: length of y

    Returns: a tuple (my_idx, neighbour_idx), where
        my_idx is the index of the current cell
        neighbour_idx is a list of neighbour indices
    """
    my_idx = ny * i + j
    neighbour_idx = []

    # left
    if i > 0:
        neighbour_idx.append(ny * (i - 1) + j)

    # lower
    if j < ny - 1:
        neighbour_idx.append(ny * i + j + 1)

    # right
    if i < nx - 1:
        neighbour_idx.append(ny * (i + 1) + j)

    # upper
    if j > 0:
        neighbour_idx.append(ny * i + j - 1)

    return my_idx, neighbour_idx


class Grid(GraphPlanner):
    """2D grid-based planner.

    Discretizes the workspace and connects neighbouring points, then plans
    through the resulting graph.
    """

    def __init__(self, workspace, step):
        # NOTE: it is convenient to lean on the pgraph library for graph-based
        # planning, though for the grid case this is quite inefficient
        self.graph = UGraph()

        # add a vertex at the corner of every grid cell
        xs = np.arange(0, workspace.width + step, step) - 0.5 * workspace.width
        ys = np.arange(0, workspace.height + step, step) - 0.5 * workspace.height
        for x in xs:
            for y in ys:
                self.graph.add_vertex(coord=(x, y))

        # connect adjacent vertices
        nx = xs.shape[0]
        ny = ys.shape[0]
        for i in range(nx):
            for j in range(ny):
                my_idx, neighbour_idx = grid_neighbour_indices(i, j, nx, ny)
                v = self.graph[my_idx]
                for idx in neighbour_idx[:2]:
                    v.connect(self.graph[idx])

        # remove any vertices in collision
        # we do this afterward because it is much easier to compute adjacent
        # nodes for a regular grid rather than one with holes due to obstacles
        for i in reversed(range(self.graph.n)):
            if workspace.point_is_in_collision(self.graph[i].coord):
                self.graph.remove(self.graph[i])
