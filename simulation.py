import math
import os.path
import sys

import numpy
import numpy as np
import random
from IPython.display import clear_output
import torch.cuda
import torch as T
import matplotlib.pyplot as plt

from Agent import Agent
show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            # if show_animation:  # pragma: no cover
            #     plt.plot(self.calc_grid_position(current.x, self.min_x),
            #              self.calc_grid_position(current.y, self.min_y), "xc")
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect('key_release_event',
            #                                  lambda event: [exit(
            #                                      0) if event.key == 'escape' else None])
            #     if len(closed_set.keys()) % 10 == 0:
            #         plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry 

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # print("min_x:", self.min_x)
        # print("min_y:", self.min_y)
        # print("max_x:", self.max_x)
        # print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        # print("x_width:", self.x_width)
        # print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def normalDistribution(minValue, maxValue):
    mid = minValue + (maxValue - minValue) / 2
    std_dev = (maxValue - mid) / 2

    return round(np.random.normal(loc=mid, scale=std_dev))

class Simulation:
    def __init__(self, robot_radius, grid_size, obstacle_count, sx, sy, gx, gy):
        self.robot_radius = robot_radius
        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.gx = gx
        self.gy = gy
        self.sx = sx
        self.sy = sy
        self.pnt2 = 1
        self.theta = [math.pi / 2]
        self.dist_tolerance = 3*robot_radius
        self.pnt = 0
        self.steps = 0
        self.shapes = ['box']
        self.pathDistance = 50
        self.cone_angle = 15
        self.getMap()
        self.tx = [sx]
        self.ty = [sy]
        self.ix = [sx]
        self.iy = [sy]
        self.goal_magnitude = None
        angle = np.linspace( 0 , 2 * np.pi , 150 )
        radius = self.robot_radius - 0.5
        xpoints = sx + radius * np.cos( angle )
        ypoints = sy + radius * np.sin( angle )
        self.robot, = plt.plot(xpoints, ypoints, marker = ".", color = 'r')
        self.robotAngle, = plt.plot(sx + radius * np.cos( self.theta[-1] ), sy + radius * np.sin( self.theta[-1]  ), marker = ".", color = 'b')

    def __motion(
        self, v_left: float, v_right: float, x: float, y: float, theta: float, dt: float
    ):
        """Calculate the forward kinematics of the robot"""
        aveVel = (1 / 2) * (v_right + v_left)
        x_Dot = -aveVel * math.sin(theta - math.pi / 2)
        y_Dot = aveVel * math.cos(theta - math.pi / 2)
        theta_Dot = (v_right - v_left) / (self.robot_radius*2)

        x_new = x + x_Dot * dt
        y_new = y + y_Dot * dt
        theta_new = (theta + theta_Dot * dt) % (2*math.pi)

        return x_new, y_new, theta_new

    def getMap(self):

        while True:
            self.gx = random.randint(2, 68)  # [m]
            self.gy = random.randint(2, 68)  # [m]

            if (self.gx + self.pathDistance > 70 and self.gx - self.pathDistance < 1) and (self.gy + self.pathDistance > 70 and self.gy - self.pathDistance < 1):
                continue
            self.sx = -1
            self.sy = -1 

            while self.sx < 2 or self.sx > 68 or self.sy < 2 or self.sy > 68:
                theta = random.random() * math.pi * 2
                self.sx = self.gx + math.cos(theta) * self.pathDistance
                self.sy = self.gy + math.sin(theta) * self.pathDistance

            if self.gx < self.sx:
                min_obstacle_x = self.gx
                max_obstacle_x = self.sx
            else:
                min_obstacle_x = self.sx
                max_obstacle_x = self.gx

            if self.gy < self.sy:
                min_obstacle_y = self.gy
                max_obstacle_y = self.sy
            else:
                min_obstacle_y = self.sy
                max_obstacle_y = self.gy   

            map = np.zeros((71, 71), dtype=bool)

            # set obstacle positions
            ox, oy = [], []
            #border
            for i in range(0, 70):
                ox.append(i)
                oy.append(0)
                map[i, 0] = True
            for i in range(0, 70):
                ox.append(70)
                oy.append(i)
                map[70, i] = True
            for i in range(0, 71):
                ox.append(i)
                oy.append(70)
                map[i, 70] = True
            for i in range(0, 71):
                ox.append(0)
                oy.append(i)
                map[0, i] = True

            #obstacles
            for i in range(self.obstacle_count):
                shape_index = random.randint(0, len(self.shapes) - 1)
                shape = self.shapes[shape_index]
                if shape == 'box':
                    wall_length = random.randint(4,6)
                    while True:
                        x_diff = max_obstacle_x - min_obstacle_x
                        y_diff = max_obstacle_y - min_obstacle_y
                        x = random.randint(round(min_obstacle_x + x_diff*.1), round(max_obstacle_x - x_diff*.1))
                        y = random.randint(round(min_obstacle_y + y_diff*.1), round(max_obstacle_y - y_diff*.1))
                        if(x < 70 and x > 0 and y < 70 and y > 0):
                            break
                    ox.append(x)
                    oy.append(y)
                    map[x, y] = True
                    direction = random.randint(0, 3)
                    i = 0
                    tempY = y
                    while tempY < 70 and i < wall_length: #Left Wall
                        i += 1
                        tempY += 1
                        ox.append(x)
                        oy.append(tempY)
                        map[x, tempY] = True
                    tempX = x
                    i = 0
                    while tempX < 70 and i < wall_length: #Bottom Wall
                            i += 1
                            tempX += 1
                            ox.append(tempX)
                            oy.append(tempY)
                            map[tempX, tempY] = True
                    i = 0
                    while tempY > 0 and i < wall_length: # Right Wall
                        i += 1
                        tempY -= 1
                        ox.append(tempX)
                        oy.append(tempY)
                        map[tempX, tempY] = True
                    i = 0
                    while tempX > 0 and i < wall_length: # Top Wall
                        i += 1
                        tempX -= 1
                        ox.append(tempX)
                        oy.append(tempY)
                        map[tempX, tempY] = True
                        
            self.ox = ox
            self.oy = oy
            self.map = map

            a_star = AStarPlanner(ox, oy, self.grid_size, self.robot_radius)
            rx, ry = a_star.planning(self.sx, self.sy, self.gx, self.gy)
            if len(rx) > 1:
                break

        rx.reverse()
        ry.reverse()
        self.rx = rx
        self.ry = ry

    def getPath(self):
        # negative if left of path 
        minVal = self.distance(self.tx[-1], self.ty[-1], self.rx[0], self.ry[0])
        self.pnt = 0
        for i in range(1, len(self.rx) - 1):
            new = self.distance(self.tx[-1], self.ty[-1], self.rx[i], self.ry[i])
            if new < minVal:
                self.pnt = i
                minVal = new
        vector_to_path = (self.rx[self.pnt] - self.tx[-1], self.ry[self.pnt] - self.ty[-1])
        # print(f'path distance: {pathDistance}')
        return vector_to_path[0], vector_to_path[1]

    def getGoal(self):
        goal_vector = (self.gx - self.tx[-1], self.gy - self.ty[-1])
        # print(f'goal: {goalDistance}')
        return goal_vector

    def getObstacle(self):
        max_distance = 10
        obsDistanceforward = 10
        obsDistanceLeft = 10
        obsDistanceRight = 10
        obsDistanceBack = 10

        obsDistanceConeLeft = 10
        obsDistanceConeLeft_mid = 10
        obsDistanceConeRight_mid = 10
        obsDistanceConeRight = 10

        for i in range(10):
            if self.map[round(self.tx[-1] + i * math.cos(self.theta[-1])), round(self.ty[-1] + i * math.sin(self.theta[-1]))]:
                obsDistanceforward = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(self.theta[-1])), self.iy[-1] + round(i * math.sin(self.theta[-1])))
            coneTheta_mid = self.theta[-1] - (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeLeft_mid = self.distance(self.tx[-1], self.ty[-1],self.ix[-1] + round(i * math.cos(coneTheta_mid)),self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = self.theta[-1] - (self.cone_angle * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeLeft = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            coneTheta_mid = self.theta[-1] + (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeRight_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = self.theta[-1] + (self.cone_angle * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeRight = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            obsDistanceforward = min(obsDistanceforward, obsDistanceConeLeft, obsDistanceConeLeft_mid, obsDistanceConeRight, obsDistanceConeRight_mid)
            if obsDistanceforward < max_distance:
                break

        obsDistanceConeLeft = 10
        obsDistanceConeLeft_mid = 10
        obsDistanceConeRight_mid = 10
        obsDistanceConeRight = 10
        for i in range(10):
            leftTheta = self.theta[-1] + (90 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(leftTheta)), round(self.ty[-1] + i * math.sin(leftTheta))]:
                obsDistanceLeft = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(leftTheta)), self.iy[-1] + round(i * math.sin(leftTheta)))
            coneTheta_mid = leftTheta - (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeLeft_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = leftTheta - (self.cone_angle * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeLeft = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            coneTheta_mid = leftTheta + (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeRight_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = leftTheta + (self.cone_angle * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeRight = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            obsDistanceLeft = min(obsDistanceLeft, obsDistanceConeLeft, obsDistanceConeLeft_mid, obsDistanceConeRight, obsDistanceConeRight_mid)
            if obsDistanceLeft < max_distance:
                break

        obsDistanceConeLeft = 10
        obsDistanceConeLeft_mid = 10
        obsDistanceConeRight_mid = 10
        obsDistanceConeRight = 10
        for i in range(10):
            rightTheta = self.theta[-1] - (90 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(rightTheta)), round(self.ty[-1] + i * math.sin(rightTheta))]:
                obsDistanceRight = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(rightTheta)), self.iy[-1] + round(i * math.sin(rightTheta)))
            coneTheta_mid = rightTheta - (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeLeft_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = rightTheta - (self.cone_angle * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeLeft = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            coneTheta_mid = rightTheta + (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeRight_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = rightTheta + (self.cone_angle * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeRight = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            obsDistanceRight = min(obsDistanceRight, obsDistanceConeLeft, obsDistanceConeLeft_mid, obsDistanceConeRight, obsDistanceConeRight_mid)
            if obsDistanceRight < max_distance:
                break

        obsDistanceConeLeft = 10
        obsDistanceConeLeft_mid = 10
        obsDistanceConeRight_mid = 10
        obsDistanceConeRight = 10
        for i in range(10):
            backTheta = self.theta[-1] + (180 * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(backTheta)), round(self.ty[-1] + i * math.sin(backTheta))]:
                obsDistanceBack = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(backTheta)), self.iy[-1] + round(i * math.sin(backTheta)))
            coneTheta_mid = backTheta - (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeLeft_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = backTheta - (self.cone_angle * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeLeft = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            coneTheta_mid = backTheta + (self.cone_angle / 2 * math.pi / 180)
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta_mid)), round(self.ty[-1] + i * math.sin(coneTheta_mid))]:
                obsDistanceConeRight_mid = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta_mid)), self.iy[-1] + round(i * math.sin(coneTheta_mid)))
            coneTheta = backTheta + (self.cone_angle * math.pi / 180) 
            if self.map[round(self.tx[-1] + i * math.cos(coneTheta)), round(self.ty[-1] + i * math.sin(coneTheta))]:
                obsDistanceConeRight = self.distance(self.tx[-1], self.ty[-1], self.ix[-1] + round(i * math.cos(coneTheta)), self.iy[-1] + round(i * math.sin(coneTheta)))
            obsDistanceBack = min(obsDistanceBack, obsDistanceConeLeft,obsDistanceConeLeft_mid, obsDistanceConeRight, obsDistanceConeRight_mid)
            if obsDistanceBack < max_distance:
                break

        # print(f'obstacle Distance: {obsDistance}')
        return (obsDistanceforward, obsDistanceLeft,  obsDistanceRight, obsDistanceBack)

    def isHittingObstacle(self):
        bot_x = round(self.tx[-1])
        bot_y = round(self.ty[-1])
        on = self.map[bot_x, bot_y]

        return on


    def getTheta(self):
        #angle difference
        pathVector = (self.rx[self.pnt + 1] - self.rx[self.pnt], self.ry[self.pnt + 1] - self.ry[self.pnt])
        robotVector = (math.cos(self.theta[-1]), math.sin(self.theta[-1]))

        dot = pathVector[0] * robotVector[0] + pathVector[1] * robotVector[1]
        det = pathVector[0] * robotVector[1] - pathVector[1] * robotVector[0]


        d_theta = math.atan2(det, dot)
        if d_theta <= math.pi:
            d_theta *= -1
        else:
            d_theta = 2 * math.pi - d_theta

        return d_theta
    
    def getReward(self):
        x, y = self.getPath()
        gx, gy = self.getGoal()

        if self.goal_magnitude != None:
            goal = 125 * (self.goal_magnitude - numpy.sqrt(gx*gx + gy * gy))
        else:
            goal = 0
        self.goal_magnitude = numpy.sqrt(gx * gx + gy * gy)

        path_distance = numpy.sqrt(x*x + y*y)
        off_path = 2 * (self.robot_radius - path_distance)
        off_path_sparse = 0
        if path_distance > 7 * self.robot_radius:
            off_path_sparse -= 200

        obstacles = self.getObstacle()
        too_close = 0
        if obstacles[0] < 1 or obstacles[1] < 1 or obstacles[2] < 1 or obstacles[3] < 1:
            too_close = min(obstacles) - 2

        win = 1000 if self.distance(self.tx[-1], self.ty[-1], self.gx, self.gy) < self.dist_tolerance else 0
        hit = -200 if self.isHittingObstacle() else 0
        theta = -.5 * abs(self.getTheta())
        #print((hit, too_close, off_path, goal, theta, win))
        return hit + too_close + win + off_path_sparse + off_path + goal + theta

    
    def isDone(self):
        term = False
        if self.distance(self.tx[-1], self.ty[-1], self.gx, self.gy) < self.dist_tolerance:
            print("Made it to goal")
            term = True
        if self.distance(self.tx[-1], self.ty[-1], self.rx[self.pnt], self.ry[self.pnt]) > 5 * self.robot_radius:
            term = True
            print("Went too far from path")
        obstacles = self.getObstacle()
        if self.isHittingObstacle():
            term = True
            print("Hit an obstacle")
        return term

    def step(self, action):
        vLeft, vRight = action[0], action[1]
        dt = .1
        x, y, theta = self.__motion(vLeft, vRight, self.tx[-1], self.ty[-1], self.theta[-1], dt)

        ##replaced by finding the actual new point and going there
        self.tx.append(x)
        self.ty.append(y)
        self.theta.append(theta)
        self.ix.append(round(x))
        self.iy.append(round(y))
        self.pnt2 += 1
        self.steps += 1
        path_x, path_y = self.getPath()
        g_x, g_y = self.getGoal()
        goal_distance = np.sqrt(g_x**2 + g_y**2)
        obstacles = self.getObstacle()

        return [path_x, path_y, g_x/goal_distance, g_y/goal_distance, obstacles[0], obstacles[1], obstacles[2], obstacles[3], self.getTheta(),  self.tx[-1], self.ty[-1], self.theta[-1]], self.getReward(), self.isDone()


    def print(self, end=True):
        plt.plot(self.ox, self.oy, ".k")
        plt.plot(self.sx, self.sy, "og")
        plt.plot(self.gx, self.gy, "xb")
        plt.plot(self.rx, self.ry, '-r')
        plt.grid(True)
        plt.axis("equal")
        plt.plot(self.tx, self.ty, "-g")        

    def show(self, x, y, theta):
        try:
            self.robot.remove()
            self.robotAngle.remove()
            angle = np.linspace( 0 , 2 * np.pi , 150 )
            radius = self.robot_radius - 0.5
            xpoints = x + radius * np.cos( angle )
            ypoints = y + radius * np.sin( angle )
            self.robot, = plt.plot(xpoints, ypoints, marker = ".", color = 'r')
            self.robotAngle, = plt.plot(x + radius * math.cos( theta ), y + radius * math.sin( theta ), marker = ".", color = 'b')
        except KeyboardInterrupt:
            print("Visualization Exited")
            raise KeyboardInterrupt

    def showPath(self):
        self.print()
        try:
            for i in range(len(self.tx)):
                self.show(self.tx[i], self.ty[i], self.theta[i])
                plt.pause(0.005)
        except KeyboardInterrupt:
            print("Visualization exited")
            return


    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def reset(self):
        self.sx = random.randint(2, 68)  # [m]
        self.sy = random.randint(2, 68)  # [m]
        self.gx = random.randint(2, 68)  # [m]
        self.gy = random.randint(2, 68)  # [m]
        self.pnt2 = 1
        self.theta = [math.pi / 2]
        self.pnt = 0      
        self.getMap()
        self.tx = [self.sx]
        self.ty = [self.sy]
        self.ix = [self.sx]
        self.iy = [self.sy]
        self.steps = 0
        self.goal_magnitude = None


        path_x, path_y = self.getPath()
        g_x, g_y = self.getGoal()
        goal_distance = np.sqrt(g_x**2 + g_y**2)
        obs_1, obs_2, obs_3, obs_4 = self.getObstacle()
        return [path_x, path_y, g_x/goal_distance, g_y/goal_distance, obs_1, obs_2, obs_3, obs_4, self.getTheta(), self.ix[-1], self.iy[-1], self.theta[-1]]

    def plot_res(self, values, title='', goal=1000, run_number=0):
        ''' Plot the reward curve and histogram of results over time.'''
        # Update the window after each episode
        clear_output(wait=True)

        # Define the figure
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        f.suptitle(title)
        ax[0].plot(values, label='score per run')
        ax[0].axhline(goal, c='red', ls='--', label='goal')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(values))
        ax[0].legend()
        # Calculate the trend
        try:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax[0].plot(x, p(x), "--", label='trend')
        except:
            print('')

        # Plot the histogram of results
        ax[1].hist(values[-50:])
        ax[1].axvline(goal, c='red', label='goal')
        ax[1].set_xlabel('Scores per Last 50 Episodes')
        ax[1].set_ylabel('Frequency')
        ax[1].legend()

        dir_name = os.path.dirname(__file__)
        fig_name = "tmp/LearningPlot_" + str(run_number) + ".png"
        plt.savefig(
            os.path.join(dir_name, fig_name)
        )
        plt.close(f)




def main():
    print(__file__ + " start!!")
            # start and goal position
    sx = random.randint(3, 67)  # [m]
    sy = random.randint(3, 67)  # [m]
    gx = random.randint(3, 67)  # [m]
    gy = random.randint(3, 67)  # [m]
    grid_size = 2  # [m]
    robot_radius = 1.0  # [m]
    obstacle_count = 5

    sim = Simulation(robot_radius, grid_size, obstacle_count, sx, sy, gx, gy)

    agent = Agent(
        alpha=0.000025,
        beta=0.00025,
        input_dims=[12],
        tau=0.001,
        batch_size=64,
        fc1_dims=300,
        fc2_dims=200,
        fc3_dims=100,
        n_actions=2,
        action_range=1,
        run_name=sys.argv[1]
    )
    agent.load_models()
    print(T.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    np.random.seed(0)

    score_history = []
    for i in range(1000):
        done = False
        score = 0
        obs = sim.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done = sim.step(act)
            #agent.remember(obs, act, reward, new_state, int(done))
            #agent.learn()
            score += reward
            obs = new_state
            if score < -8000:
                break

        score_history.append(score)
        print(
            "episode",
            i,
            "score %.2f" % score,
            "100 game average %.2f" % np.mean(score_history[-100:]),
        )

        # if i % 25 == 0:
        #     agent.save_models()
        #     sim.plot_res(score_history, "Learning Rate Graphs", 2000, int(sys.argv[1]))
        if i % 1 == 0:
            sim.showPath()
            plt.close()





if __name__ == '__main__':
    main()
