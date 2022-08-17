import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from pgraph import UGraph, DGraph, UVertex
import time
from planning import RRT, Workspace, Rectangle, Circle
from RRT_star import RRT_star


class RRT_star_unbounded(RRT_star):

    def query(
        self,
        start,
        goal,
        n=1000,
        min_edge_len=0.5,
        max_edge_len=1,
        niu = 0.8
    ):
        new_size = self.graph.n 
        start_time = time.time()
        samples = []
        prev = 0
        while new_size<n:
            # if prev !=new_size:
            #     print(new_size)
            #     prev = new_size
            # TODO maybe change min and max edge len based on how many points have been taken?
            # add radius with minimum path
            if len(samples) == 0: 
                q = self.workspace.sample()
                v_nearest, dist = self.closest_vertex(q)
                if dist<min_edge_len:
                    continue
                if self.workspace.edge_is_in_collision(q, v_nearest.coord):
                    continue
                # split distance into multiple edges
                if dist > max_edge_len:
                    q_closest = v_nearest.coord
                    vertices = round(dist / max_edge_len)
                    for i in range(vertices - 1):
                        cur_vertex = q_closest + (i + 1) * (q - q_closest) / np.linalg.norm(
                            q - q_closest
                        )
                        samples.append(cur_vertex)
                samples.append(q)
            run = max(1,len(samples))
            while len(samples)>0:
                q = samples.pop()
                #n =self.graph.n
                #print(2 * math.sqrt((math.log(n) / n)))
                #rn = min(100/n, 5) # change rn depending on how many nodes there are, look at formula on RTT star paper

                rn = 2*max_edge_len
                neighborhood = self.near_vertices(q,rn)

                v_nearest = self.nearest(q, v_nearest, neighborhood)

                v = self.graph.add_vertex(q)
                edge = self.graph.add_edge(v,v_nearest)
                v.parent = v_nearest
                v.edge = edge

                self.re_route(v, v_nearest, neighborhood)
                new_size+=1
                if tuple(v.coord) == goal:
                    self.v_goal = v


            if self.end_condition(goal, goal_dist=1): #and i%30 == 0
                samples.append(goal)
        self.preprocessing_time = 0
        self.query_time = time.time() - start_time


def main():
    # create the workspace with some obstacles
    workspace = Workspace(20, 20)
    workspace.obstacles = [
        #Rectangle((0, -2), 1, 3),
        #Rectangle((-2, -4), 6, 1),
        #Rectangle((-2, 1), 3, 1),
        Rectangle((-5, -10), 1, 15),
        Rectangle((-1, -5), 1, 15),
        Rectangle((5, -10), 1, 15),

        #Rectangle((-4, 3), 1, 2),
        #Circle((-3, 2), 0.5),
    ]

    # start and goal locations
    start = (-8, -5)
    goal = (8, 9)
    planner = RRT_star_unbounded(workspace,start)
    planner.query(start, goal,n=300,min_edge_len=0.5 ,max_edge_len=1)
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
    if type(path) == type(np.array([])):
        ax.plot(path[:, 0], path[:, 1], color="g")
    plt.show()
    


if __name__ == "__main__":
    main()
