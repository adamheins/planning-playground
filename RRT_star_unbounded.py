import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pgraph import UGraph, DGraph, UVertex
import time
from planning import RRT, Workspace, Rectangle, Circle
from RRT_star import RRT_star
import math

class RRT_star_unbounded(RRT_star):

    def new_point_routine(self, q, v_nearest,dist,max_edge_len,min_edge_len):
        n =self.graph.n
        #print(2 * math.sqrt((math.log(n) / n)))
        #rn = min(100/n, 5) # change rn depending on how many nodes there are, look at formula on RTT star paper
        rn = 2*max_edge_len
        neighborhood = self.near_vertices(q,rn)
        if len(neighborhood) == 0:
            return
        cost = self.cost(v_nearest) + self.connection(v_nearest.coord, q)
        for vertex in neighborhood:
            point = vertex.coord
            if not self.workspace.edge_is_in_collision(q,point):
                c_prime = self.cost(vertex) + self.connection(point, q)
                if cost>c_prime:
                    cost=c_prime
                    v_nearest = vertex
                    dist = self.connection(point, q)

        # split distance into multiple edges
        if dist > max_edge_len:
            q_closest = v_nearest.coord
            vertices = round(dist / max_edge_len)
            for i in range(vertices - 1):
                cur_vertex = q_closest + (i + 1) * (q - q_closest) / np.linalg.norm(
                    q - q_closest
                )
                v_nearest, dist = self.closest_vertex(cur_vertex)
                if dist<min_edge_len:
                    continue
                v = self.new_point_routine(cur_vertex,v_nearest,max_edge_len,max_edge_len,min_edge_len)
                v_nearest = v

        v = self.graph.add_vertex(q)
        edge = v.connect(v_nearest)
        v.parent = v_nearest
        v.parent = v_nearest
        v.edge = edge
        q_min = v_nearest.coord

        for vertex in neighborhood:
            point = vertex.coord
            dist = np.linalg.norm(point -q_min)
            if dist== 0:
                continue
            cost = self.cost(v) + self.connection(q,point)
            if (not self.workspace.edge_is_in_collision(q,point)) and (self.cost(vertex)>cost):
                try:
                    self.graph.remove(vertex.edge)
                except:
                    print(vertex.coord)
                # # split distance into multiple edges
                # if dist > max_edge_len:
                #     q_closest = vertex.coord
                #     vertices = round(dist / max_edge_len)
                #     for i in range(vertices - 1):
                #         cur_vertex = q_closest + (i + 1) * (q - q_closest) / np.linalg.norm(
                #             q - q_closest
                #         )
                #         v_nearest, dist = self.closest_vertex(cur_vertex)
                #         if dist<min_edge_len:
                #             continue
                #         v = self.new_point_routine(cur_vertex,v_nearest,max_edge_len,max_edge_len,min_edge_len)
                #         v_nearest = v

                vertex.parent = v
                edge = vertex.connect(v)
                vertex.edge = edge
        return v



def main():
    # create the workspace with some obstacles
    workspace = Workspace(10, 10)
    workspace.obstacles = [
        Rectangle((0, -2), 1, 3),
        Rectangle((-2, -4), 6, 1),
        Rectangle((-2, 1), 3, 1),
        Rectangle((-4, 3), 1, 2),
        Circle((-3, 2), 0.5),
    ]

    # start and goal locations
    start = (-4, -4)
    goal = (4, 4)
    planner = RRT_star_unbounded(workspace,start)
    planner.query(start, goal,min_edge_len=0.4,max_edge_len=1.5)
    path = planner.find_path()

    print("preprocessing time:", planner.preprocessing_time)
    print("query time:", planner.query_time)
    # plot the results
    plt.figure()
    ax = plt.gca()
    workspace.draw(ax)
    planner.draw(ax)
    ax.plot(start[0], start[1], "o", color="g")
    ax.plot(goal[0], goal[1], "o", color="r")
    path.draw(ax,rgb=(0.5,1,0))
    plt.show()
    


if __name__ == "__main__":
    main()