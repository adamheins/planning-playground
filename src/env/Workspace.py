import numpy as np
from matplotlib import patches

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
        if d ==0:
            return False
        r = (xy2 - xy1) / d

        steps = np.arange(0, d, h)
        points = xy1 + steps[:, None] * r
        for point in points:
            if self.point_is_in_collision(point):
                return True
        return False

    def sample(self, n=1, accept_point_in_collision=True):
        """Sample n points from the workspace that is collision-free.

        Will try generating up to max_tries_factor * n points in total, after
        which the samples are returned even if there are not n of them.
        """
        samples = []
        for _ in range(n):
            # sample the entire workspace uniformly
            xy = (np.random.random(2) - 0.5) * [self.width, self.height]
            collision = self.point_is_in_collision(xy)
            # add this point if it is collision-free
            if accept_point_in_collision or not collision:
                samples.append(xy)
            else:
                # TODO write later
                # Idea: while loop that moves point
                # or try generate n* max_tries_factor
                continue

        if n == 1:
            return samples[0]
        return np.array(samples)
