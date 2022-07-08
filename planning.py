import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pgraph import UGraph, DGraph
import time

import IPython


class Rectangle:
    """Rectangular obstacle."""

    def __init__(self, xy, width, height):
        self.xy = np.array(xy)
        self.width = width
        self.height = height

    def draw(self, ax, color="k"):
        ax.add_patch(patches.Rectangle(self.xy, self.width, self.height, color=color))

    def contains(self, xy):
        """Check if the point xy is contained in the rectangle.

        Returns True if the rectangle contains the point, False otherwise.
        """
        x, y = xy
        if x < self.xy[0] or x > self.xy[0] + self.width:
            return False
        if y < self.xy[1] or y > self.xy[1] + self.height:
            return False
        return True


class Circle:
    """Circular obstacle."""

    def __init__(self, xy, radius):
        self.xy = np.array(xy)
        self.radius = radius

    def draw(self, ax, color="k"):
        ax.add_patch(patches.Circle(self.xy, self.radius, color=color))

    def contains(self, xy):
        """Check if the point xy is contained in the circle.

        Returns True if the rectangle contains the point, False otherwise.
        """
        return np.linalg.norm(self.xy - xy) <= self.radius


class Workspace:
    """2D workspace for the robot."""

    def __init__(self, width, height):
        """Workspace is a rectangle with given width and height centered at (0, 0)."""
        self.width = width
        self.height = height
        self.obstacles = []

    def draw(self, ax):
        """Draw the workspace on the axes ax."""
        w2 = 0.5 * self.width
        h2 = 0.5 * self.height

        ax.set_xlim([-w2, w2])
        ax.set_ylim([-h2, h2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid()

        for obstacle in self.obstacles:
            obstacle.draw(ax)

    def point_is_in_collision(self, xy):
        """Check if the point xy is in collision with obstacles.

        Returns True if the point is in collision, False otherwise."""
        for obstacle in self.obstacles:
            if obstacle.contains(xy):
                return True
        return False

    def edge_is_in_collision(self, xy1, xy2, h=0.1):
        """Check if the straight-line edge between points xy1 and xy2 is collision-free.

        The edge is discretized in steps of length h."""
        d = np.linalg.norm(xy2 - xy1)
        r = (xy2 - xy1) / d

        steps = np.arange(0, d, h)
        points = xy1 + steps[:, None] * r
        for point in points:
            if self.point_is_in_collision(point):
                return True
        return False

    def sample(self, n=1, max_tries_factor=5):
        """Sample n points from the workspace that is collision-free.

        Will try generating up to max_tries_factor * n points in total, after
        which the samples are returned even if there are not n of them.
        """
        samples = []
        num_tries = 0
        while len(samples) < n and num_tries < max_tries_factor * n:
            # sample the entire workspace uniformly
            xy = (np.random.random(2) - 0.5) * [self.width, self.height]

            # add this point if it is collision-free
            if not self.point_is_in_collision(xy):
                samples.append(xy)

            num_tries += 1

        if n == 1:
            return samples[0]
        return np.array(samples)


class GraphPlanner:
    """Base class for all graph-based planners."""
    def __init__(self, graph):
        # graph may be directed or undirected, as selected by the subclass
        self.graph = graph

    def draw(self, ax, vertices=True, edges=True, rgb=(0, 0, 1)):
        """Draw the graph."""
        if vertices:
            for vertex in self.graph:
                x, y = vertex.coord
                ax.plot(x, y, "x", color=rgb)

        if edges:
            for edge in self.graph.edges():
                v1, v2 = edge.endpoints
                ax.plot(
                    [v1.coord[0], v2.coord[0]],
                    [v1.coord[1], v2.coord[1]],
                    color=(rgb[0], rgb[1], rgb[2], 0.25),
                )

    def k_nearest_neighbours(self, v, k, vertices=None):
        """Get k nearest neighbours to vertex v."""
        # get distance to each vertex in the graph
        if vertices is None:
            vertices = self.graph

        dists = []
        for vo in vertices:
            if v is vo:
                continue
            dists.append((vo.distance(v.coord), vo))

        # sort nearest to farthest
        dists.sort(key=lambda x: x[0])

        # return the k closest ones
        return [vo for _, vo in dists[:k]]

    def neighbours_within_dist(self, v, dist, vertices=None):
        """Get all vertices within dist of vertex v."""
        if vertices is None:
            vertices = self.graph

        dists = []
        for vo in vertices:
            if v is vo:
                continue
            dists.append((vo.distance(v.coord), vo))

        dists.sort(key=lambda x: x[0])

        # return all neighbours with distance dist
        return [vo for d, vo in dists if d <= dist]

    def closest_vertex(self, q, vertices=None):
        """Get closest vertex in the graph, or a subset of vertices."""
        # this generalized the pgraph implementation, which doesn't handle a
        # subset of vertices
        if vertices is None:
            vertices = self.graph

        min_dist = vertices[0].distance(q)
        v_closest = vertices[0]

        for v in vertices:
            d = v.distance(q)
            if d < min_dist:
                min_dist = d
                v_closest = v

        return v_closest, min_dist

    def query(self, start, goal):
        """Get the shortest path between the start and goal points, if one exists."""
        start_time = time.time()
        vs, _ = self.graph.closest(start)
        vg, _ = self.graph.closest(goal)
        path = self.graph.path_Astar(vs, vg)
        if path is None:
            return None
        idx = path[0]
        points = np.array([self.graph[i].coord for i in idx])
        self.query_time = time.time()-start_time
        return np.vstack((start, points, goal))


class PRM(GraphPlanner):
    """Probabilistic road-map for multi-query planning."""

    def __init__(self, workspace):
        super().__init__(UGraph())
        self.workspace = workspace

    def add_vertices(self, n, k):
        """Add n new vertices to the graph.

        Vertices are connected to their k nearest neighbours, if the edges
        between thm are collision-free.
        """
        new_size = self.graph.n + n
        start_time = time.time()
        while self.graph.n < new_size:
            # generate a new (collision-free) point
            point = self.workspace.sample()
            v = self.graph.add_vertex(point)

            # add edges between the new vertex and k nearest neighbours if the
            # edge between them is collision-free
            for vo in self.k_nearest_neighbours(v, k):
                if self.workspace.edge_is_in_collision(v.coord, vo.coord):
                    continue
                v.connect(vo)
        self.preprocessing_time = time.time()-start_time


class RRG(GraphPlanner):
    """Rapidly-exploring random graph.

    Equivalent to a rapidly-explored random tree (RRT) if all vertices are
    added with connect_multiple_vertices=False.
    """

    def __init__(self, workspace, q0, qf):
        super().__init__(UGraph())
        self.workspace = workspace
        self.root = self.graph.add_vertex(q0)

        # TODO include this properly, bias toward it, etc.
        # self.goal = self.graph.add_vertex(qf)

    def add_vertices(
        self,
        n,
        min_edge_len=0.5,
        max_edge_len=1,
        near_dist=1,
        connect_multiple_vertices=True,
    ):
        """Add n vertices to the RRG."""
        new_size = self.graph.n + n
        start_time = time.time()
        while self.graph.n < new_size:
            # TODO we can actually try to connect as close as possible to a
            # vertex in collision
            q = self.workspace.sample()

            v_nearest, dist = self.closest_vertex(q)
            if dist < min_edge_len:
                continue
            q_nearest = v_nearest.coord

            # move toward q as much as possible
            if dist > max_edge_len:
                q = q_nearest + (q - q_nearest) * max_edge_len / dist

            # don't add if edge is in collision
            if self.workspace.edge_is_in_collision(q_nearest, q):
                continue

            v = self.graph.add_vertex(q)
            v.connect(v_nearest)

            # find and add additional nearby vertices
            if connect_multiple_vertices:
                vs_near = self.neighbours_within_dist(v, near_dist)
                for vo in vs_near:
                    # don't make duplicate edges
                    if vo.isneighbour(v):
                        continue

                    if v.distance(vo.coord) < min_edge_len:
                        continue

                    # avoid collisions
                    if self.workspace.edge_is_in_collision(vo.coord, v.coord):
                        continue
                    vo.connect(v)
        self.preprocessing_time = time.time()-start_time



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


def main():
    # create the workspace with some obstacles
    workspace = Workspace(10, 10)
    workspace.obstacles = [
        Rectangle((0, -2), 1, 3),
        Rectangle((-2, -4), 6, 1),
        Rectangle((-2, 1), 3, 1),
        Rectangle((-4, 3), 1, 2),
    ]

    # start and goal locations
    start = (-4, -4)
    goal = (4, 4)

    # use an RRT planner
    planner = RRG(workspace, start, goal)
    planner.add_vertices(n=100, connect_multiple_vertices=False)

    # alternative: use a grid-based planner (only practical for small dimensions)
    # planner = Grid(workspace, 0.5)

    # alternative: use a PRM planner
    # planner = PRM(workspace)
    # planner.add_vertices(n=100)

    path = planner.query(start, goal)
    print("preprocessing time:", planner.preprocessing_time)
    print("query time:", planner.query_time)
    # plot the results
    plt.figure()
    ax = plt.gca()
    workspace.draw(ax)
    planner.draw(ax)
    ax.plot(start[0], start[1], "o", color="g")
    ax.plot(goal[0], goal[1], "o", color="r")

    if path is None:
        print("Could not find a path from start to goal!")
    else:
        ax.plot(path[:, 0], path[:, 1], color="r")

    plt.show()


if __name__ == "__main__":
    main()
