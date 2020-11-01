import numpy as np
import math as m
from matplotlib import pyplot as plt
#from ZweiteMethodeDrehErkennung import x_g, y_g, z_g, goal_x, goal_y, wand_x, wand_y
from Points import x_g, y_g, z_g, goal_x, goal_y, wand
from Functions import Dreherkennung, plot_path, plot_path_turns, closest_point_func, plot_wand_ClosestPoint_p1_p2_modBahn, Bezier, angles_p1_p2 , plot_wand_ClosestPoint_p1_p2, plot_wand_ClosestPoint_p1, connect_points2,connect_points3, plot_path_wand, min_max,p_1_determination3, p_2_determination3, neighbors_check
################################################
x_original = []
y_original = []
wand_x = []
wand_y = []
x_original[:] = [l for l in x_g]
y_original[:] = [l for l in y_g]
wand_x[:] = [i[0] for i in wand]
wand_y[:] = [i[1] for i in wand]
#wand = list(zip(wand_x,wand_y))
################################################

#x_g = x_g[:700]
#y_g = y_g[:700]
#z_g = z_g[:700]

#del x_g[1::10]
#del y_g[1::10]
#del z_g[1::10]
#plot_path(x_g,y_g)
#################################################

points = Dreherkennung(x_g,y_g,z_g) # Bestimmung der Drehpunkte

points = neighbors_check(wand,1.5,points) # Check Neighbors of each turn and modify the points array
start_turn_x, start_turn_y, end_turn_x, end_turn_y = points[4:]
#print(start_turn_x)
min_max_value = min_max(x_g,y_g,z_g,points)
print(points)
#print(min_max_value)
y_g[:] = [-i for i in y_g]
end_turn_y[:] = [-i for i in end_turn_y]
start_turn_y[:] = [-i for i in start_turn_y]
plot_path_turns(y_g,x_g,start_turn_y, start_turn_x, end_turn_y, end_turn_x)
#plot_path_turns(x_g,y_g,start_turn_x, start_turn_y, end_turn_x, end_turn_y)


p_1_x, p_1_y, closest_point1_x, closest_point1_y = p_1_determination3(points, min_max_value, wand)
#plot_wand_ClosestPoint_p1(x_g,y_g,start_turn_x, start_turn_y, end_turn_x, end_turn_y,wand_x,wand_y, closest_point1_x, closest_point1_y,p_1_x, p_1_y)

p_2_x, p_2_y, closest_point2_x, closest_point2_y = p_2_determination3(points, min_max_value, wand)
#plot_wand_ClosestPoint_p1_p2(x_g,y_g,start_turn_x, start_turn_y, end_turn_x, end_turn_y,wand_x,wand_y, closest_point1_x, closest_point1_y,p_1_x, p_1_y,p_2_x,p_2_y)
#assert len(points[0]) == len(points[1]) == len(points[2]) == len(points[3])
#print(p_1_x,p_1_y)
#print(p_2_x,p_2_y)

x_m, y_m = connect_points3(points,x_g,y_g,p_1_x, p_1_y, p_2_x, p_2_y,goal_x,goal_y) 
#print(x_m[-3:])
#print(y_m[-3:])
#plot_path_turns(x_2,y_2,start_turn_x, start_turn_y, end_turn_x, end_turn_y)
#plot_wand_ClosestPoint_p1(x_g,y_g,start_turn_x, start_turn_y, end_turn_x, end_turn_y,wand_x,wand_y, closest_point1_x, closest_point1_y,p_1_x, p_1_y)
#plot_wand_ClosestPoint_p1_p2(x_g,y_g,start_turn_x, start_turn_y, end_turn_x, end_turn_y,wand_x,wand_y, closest_point1_x, closest_point1_y,p_1_x, p_1_y,p_2_x,p_2_y)
plot_wand_ClosestPoint_p1_p2_modBahn(x_original,y_original,start_turn_x,start_turn_y,end_turn_x,end_turn_y,wand_x,wand_y,closest_point1_x, closest_point1_y, p_1_x, p_1_y, p_2_x, p_2_y, x_m, y_m)

