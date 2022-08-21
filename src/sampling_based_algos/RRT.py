import numpy as np
from src.sampling_based_algos.RRG import RRG


class RRT(RRG):
    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)
        self.v_start.parent = None

        # updated when we find a path to the goal node
        self.v_goal = None

    def has_path_to_goal(self):
        """Return True if a path to the goal exists, False otherwise."""
        return self.v_goal is not None

    def add_vertex(self, v_nearest, q):
        """Add a vertex located at q and connected to v_nearest."""
        v = self.graph.add_vertex(q)
        v.connect(v_nearest)
        v.parent = v_nearest
        return v

    def extend(
        self,
        goal,
        n=100,
        min_edge_len=0.5,
        max_edge_len=1,
        niu=1,
        divide_edges=True,
        stop_early=True,
    ):
        """Extend the tree with up to n nodes.

        If `stop_early` is passed, return as soon as a path to the goal is found.
        If `divide_edges` is passed, then instead of discarding samples farther
        than `max_edge_len` from the graph, multiples vertices are added such
        that the edges between each are less than `max_edge_len`.

        Returns True if a path to the goal exists, False otherwise.
        """
        count = 0
        while count < n:
            q = self.workspace.sample()
            v_nearest, dist = self.closest_vertex(q)

            # if new vertex would be too far away, split it so that each new
            # vertex is within max_edge_len of each other
            if divide_edges and dist > max_edge_len:
                num_samples = round(dist / max_edge_len)
                qs = []
                for i in range(num_samples):
                    step = (num_samples - i) / num_samples
                    qs.append(v_nearest.coord + step * (q - v_nearest.coord))
            else:
                qs = [q]

            # process all samples
            while len(qs) > 0:
                q = qs.pop()
                v_nearest, dist = self.closest_vertex(q)
                q = self.steer(q, v_nearest, dist, niu, min_edge_len)
                if q is None:
                    continue

                # add the new vertex
                # TODO rewire_radius should be a function of count
                v_nearest, _ = self.closest_vertex(q)
                v = self.add_vertex(v_nearest, q)
                count += 1

                # try to connect to the goal if we don't already have a path
                if not self.has_path_to_goal():
                    if v.distance(
                        goal
                    ) <= max_edge_len and not self.workspace.edge_is_in_collision(q, goal):
                        self.v_goal = self.add_vertex(v, goal)
                        count += 1

                # if we just want a path and don't want to iterate further, stop
                # now
                if self.has_path_to_goal() and stop_early:
                    return True

        return self.has_path_to_goal()

    def steer(self, q, v_nearest, dist, niu, min_edge_len):
        """Steer the point closer to the tree

        niu specifies how far to shift the sampled point (q) toward the
        closest point in the tree (q_nearest)."""
        q_nearest = v_nearest.coord
        q_new = q + (q_nearest - q) * niu / dist
        dist = dist - niu

        if dist < min_edge_len:
            return None

        if self.workspace.edge_is_in_collision(q_new, q_nearest):
            # TODO we would like to make this the closest point not in collision
            return None

        return q_new

    def get_path_to_goal(self):
        """Get the path to the goal node.

        Returns a 2D array of points representing the path or None if no path
        has been found."""
        if not self.has_path_to_goal():
            return None

        path = []
        v = self.v_goal
        while v is not None:
            path.append(v.coord)
            v = v.parent
        return np.array(path)