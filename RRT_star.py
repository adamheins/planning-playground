import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pgraph import UGraph, DGraph, UVertex
import time
from planning import RRT, Workspace, Rectangle, Circle
import math

class RRT_star(RRT):
    def __init__(self, workspace, q0):
        super().__init__(workspace, q0)
        self.cost_dic = {}

    def cost(self,vb):
        """Determine the cost in between two points"""
        cost = 0
        vb = tuple(vb)
        for key in self.cost_dic.keys():
            if key == vb: 
                cost = self.cost_dic[vb]
            # elif key == va:
            #     cost_a = self.cost_dic[vb]
        return cost
        

    def near_vertices(self,x,rn):
        """Return all vertexes belonging to the graph in the ball"""
        X_near = []
        for vertex in self.graph:
            q = vertex.coord
            #print(np.linalg.norm(x-q),rn)
            if np.linalg.norm(x-q) <= rn:
                X_near.append(vertex)
        return X_near
    
    def connection(self, va, vb):
        """Return the cost of connecting two verices"""
        return np.linalg.norm(va -vb)

    def expand_graph(self, min_edge_len, max_edge_len, niu=0.8):
        """Add a vertex to the RRT graph"""
        q = self.workspace.sample()
        v_nearest, dist = self.closest_vertex(q)
        q_nearest = v_nearest.coord

        # steer the point closer to the tree
        q_new = q + (q_nearest - q) * niu / dist
        dist = dist - niu
        if self.workspace.edge_is_in_collision(q_new, q_nearest):
            #Idea: instead of simply returning, do something smarter?
            return
        q = q_new
        if dist < min_edge_len:
            return

        n =self.graph.n
        #print(2 * math.sqrt((math.log(n) / n)))
        #rn = min(100/n, 5) # change rn depending on how many nodes there are, look at formula on RTT star paper
        rn = 5
        neighborhood = self.near_vertices(q,rn)
        if len(neighborhood) == 0:
            return
        v_min = v_nearest
        cost = self.cost(v_nearest.coord) + self.connection(v_nearest.coord, q)
        for vertex in neighborhood:
            point = vertex.coord
            if not self.workspace.edge_is_in_collision(q,point):
                c_prime = self.cost(point) + self.connection(point, q)
                if cost>c_prime:
                    cost=c_prime
                    v_min = vertex
        self.cost_dic[tuple(q)] = cost
        v = self.graph.add_vertex(q)
        edge = v.connect(v_min)
        v.parent = v_min
        v.edge = edge
        q_min = v_min.coord

        for vertex in neighborhood:
            point = vertex.coord
            if np.linalg.norm(point -q_min)== 0:
                continue
            cost = self.cost(v.coord) + self.connection(q,point)
            if (not self.workspace.edge_is_in_collision(q,point)) and (self.cost(vertex.coord)>cost):
                self.graph.remove(vertex.edge)
                vertex.parent = v
                edge = vertex.connect(v)
                vertex.edge = edge
                self.cost_dic[tuple(point)] = cost


    def query(
        self,
        start,
        goal,
        n=1000,
        min_edge_len=0.5,
        max_edge_len=1,
        niu = 0.8
    ):
        new_size = self.graph.n + n
        start_time = time.time()
        solution = False
        for i in range(n):
            # TODO maybe change min and max edge len based on how many points have been taken?
            # add radius with minimum path
            if i%30 == 0 and not solution:
                plt.figure()
                ax = plt.gca()
                self.workspace.draw(ax)
                self.draw(ax)
                ax.plot(start[0], start[1], "o", color="g")
                ax.plot(goal[0], goal[1], "o", color="r")
                plt.show()
            self.expand_graph(min_edge_len, max_edge_len,niu)
            if self.end_condition(goal,goal_dist=1) and i%30 == 0:
                solution = True
                plt.figure()
                ax = plt.gca()
                self.workspace.draw(ax)
                self.draw(ax)
                ax.plot(start[0], start[1], "o", color="g")
                ax.plot(goal[0], goal[1], "o", color="r")
                path = self.find_path()
                path.draw(ax)
                plt.show()
                
                # self.preprocessing_time = 0
                # self.query_time = time.time() - start_time
                # break
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time


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
    planner = RRT_star(workspace,start)
    planner.query(start, goal)
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