from .graph_planner import GraphPlanner
import time
from pgraph import UGraph


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
        self.preprocessing_time = time.time() - start_time
