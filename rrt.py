import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import KDTree
from pgraph import UGraph

import IPython


class Rectangle:
    """Rectangular obstacle."""

    def __init__(self, xy, width, height):
        self.xy = xy
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
        self.xy = xy
        self.radius = radius

    def draw(self, ax, color="k"):
        ax.add_patch(patches.Circle(self.xy, self.radius, color=color))


class Workspace:
    """Workspace for the robot."""

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
        for obstacle in self.obstacles:
            if obstacle.contains(xy):
                return True
        return False

    def edge_is_in_collision(self, xy1, xy2, h=0.1):
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


class PRM:
    def __init__(self, workspace, n, k):
        self.graph = UGraph()

        # generate samples and add them to the graph
        self.samples = workspace.sample(n)
        for sample in self.samples:
            self.graph.add_vertex(sample)

        # we now want to connect nodes to their k nearest neighbours. We do
        # this using a KDTree for distance efficient distance computation.
        tree = KDTree(self.samples)
        for i, sample in enumerate(self.samples):
            _, idx = tree.query(sample, k)
            for j in idx:
                # don't connect the node to itself
                if i == j:
                    continue

                v1 = self.graph[i]
                v2 = self.graph[int(j)]

                # don't connect the nodes if the edge between them goes through
                # an obstacle
                if workspace.edge_is_in_collision(v1.coord, v2.coord):
                    continue

                # connect if these nodes are not already connected
                if not v1.isneighbour(v2):
                    v1.connect(v2)

    def draw(self, ax, vertices=True, edges=True):
        """Draw the PRM on the provided axes ax."""
        if vertices:
            for vertex in self.graph:
                x, y = vertex.coord
                ax.plot(x, y, "x", color="b")

        if edges:
            for edge in self.graph.edges():
                v1, v2 = edge.endpoints
                ax.plot(
                    [v1.coord[0], v2.coord[0]],
                    [v1.coord[1], v2.coord[1]],
                    color=(0, 0, 1, 0.25),
                )

    def query(self, start, goal):
        """Get the shortest path between the start and goal points, if one exists."""
        vs, _ = self.graph.closest(start)
        vg, _ = self.graph.closest(goal)
        idx, _, _ = self.graph.path_Astar(vs, vg)
        points = np.array([self.graph[i].coord for i in idx])
        return np.vstack((start, points, goal))


def main():
    # create the workspace
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

    # samples = workspace.sample(10)
    prm = PRM(workspace, n=100, k=10)
    path = prm.query(start, goal)

    plt.figure()
    ax = plt.gca()
    workspace.draw(ax)
    prm.draw(ax)
    ax.plot(start[0], start[1], "o", color="g")
    ax.plot(goal[0], goal[1], "o", color="r")
    ax.plot(path[:, 0], path[:, 1], color="r")
    plt.show()


if __name__ == "__main__":
    main()
