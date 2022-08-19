

import numpy as np
from pgraph import UGraph
from src.sampling_based_algos.RRT import RRT



class RRT_star(RRT):
    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)

    def vertex_cost(self, v):
        """Determine the cost to reach vertex v from the start."""
        cost = 0
        while v.parent is not None:
            cost += self.edge_cost(v.coord, v.parent.coord)
            v = v.parent
        return cost

    def edge_cost(self, qa, qb):
        """Return the cost of connecting two vertices"""
        return np.linalg.norm(qa - qb)

    def add_vertex(self, v_nearest, q, neighborhood_radius, rewire=True):
        """Add a vertex located at q to the graph connected to existing vertex parent."""
        v = self.graph.add_vertex(q)

        # nominal cost of connecting to nearest neighbor
        cost = self.vertex_cost(v_nearest) + self.edge_cost(v_nearest.coord, q)
        neighborhood = self.neighbours_within_dist(q, neighborhood_radius)

        # see if cost is reduced by connecting to any of the vertices in the neighborhood
        for v_near in neighborhood:
            if not self.workspace.edge_is_in_collision(q, v_near.coord):
                new_cost = self.vertex_cost(v_near) + self.edge_cost(v_near.coord, q)
                if new_cost < cost:
                    cost = new_cost
                    v_nearest = v_near

        # connect
        v.parent = v_nearest
        v.edge = self.graph.add_edge(v_nearest, v)

        if rewire:
            self.rewire(v, neighborhood)

        return v


    def rewire(self, v, neighborhood):
        """Rewire edges in the neighborhood to minimize cost.

        In particular, check if the cost to reach each node would be lower if
        connected through the new node v."""
        cost_v = self.vertex_cost(v)
        for v_near in neighborhood:
            # check if we can connect at all
            if self.workspace.edge_is_in_collision(v.coord, v_near.coord):
                continue

            cost = cost_v + self.edge_cost(v.coord, v_near.coord)
            if self.vertex_cost(v_near) > cost:
                self.graph.remove(v_near.edge)
                v_near.parent = v
                v_near.edge = self.graph.add_edge(v, v_near)



