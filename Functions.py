import numpy as np 
import math as m
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import linprog
from scipy import spatial
#####################################################
def Dreherkennung(x,y,z):
    global start_turn_x, start_turn_y, end_turn_x, end_turn_y
    start_turn_x = []
    start_turn_y = []
    start_turn_theta = []
    start_turn_indices = []
    end_turn_x = []
    end_turn_y = []
    end_turn_theta = []
    end_turn_indices = []
    #max_x = []
    #max_y = []
    ###############################
    diff = np.diff(z)
    find_end = False
    find_start = True
    pass_non_zero = True
    for n,i in enumerate(diff):
        if i != 0 and find_start and pass_non_zero : # Wenn die erste Aenderung in Winkelpfad erkannt wurde
            start_turn_x.append(x[n])
            start_turn_y.append(y[n])
            start_turn_theta.append(z[n])
            start_turn_indices.append(n)
            find_end = True
            find_start = False
            pass_non_zero = False
    ################################################## END Turn
        if (find_end or find_start) and i == 0 : #wait until the transient curve pass
            pass_non_zero = True

        if i != 0 and pass_non_zero and not any(diff[n+1:n+20]):
            
            end_turn_x.append(x[n])
            end_turn_y.append(y[n])
            end_turn_theta.append(z[n])
            end_turn_indices.append(n)
            find_end = False
            find_start = True
            pass_non_zero = False

    if len(start_turn_indices) - len(end_turn_indices) == 1 : # If the last start point has not any end point -> take the first element after transient curve as end point.
        for n,i in enumerate(diff[start_turn_indices[-1]:]) :
            if i == 0 :
                end_turn_x.append(x[n+start_turn_indices[-1]])
                end_turn_y.append(y[n+start_turn_indices[-1]])
                end_turn_theta.append(z[n+start_turn_indices[-1]])
                end_turn_indices.append(n+start_turn_indices[-1])
                break
            elif n == len(diff[start_turn_indices[-1]:])-1:
                end_turn_x.append(x[-1])
                end_turn_y.append(y[-1])
                end_turn_theta.append(z[-1])
                end_turn_indices.append(len(diff))
                break

    '''if len(start_turn_x) > 1: 
            n=0
            for i in (range(len(start_turn_x)-1)):
                Steigung = m.ceil(m.degrees(m.atan2(start_turn_y[n+1] - end_turn_y[n] , start_turn_x[n+1] -end_turn_x[n])))
                #print(Steigung)
                if (Steigung % 90) > 50 or (Steigung % 90) < -50 :
                    #print("Steigung zwischen "+str(n+1)+"'st Endpunkt"+ " und "+ str(n+2)+"'st Startpunkt is: " + str(Steigung%90))
                    #print("Bin Hier")
                    start_turn_x.pop(n+1)
                    start_turn_y.pop(n+1)
                    start_turn_theta.pop(n+1)
                    start_turn_indices.pop(n+1)
                    end_turn_x.pop(n)
                    end_turn_y.pop(n)
                    end_turn_theta.pop(n)
                    end_turn_indices.pop(n)
                    n-=1
                n+=1'''
    n=0     # if the angle difference between start and end point of each turn less than 10 grad is, delete that turn
    for i,j in  list(zip(start_turn_theta,end_turn_theta)):
        if -10 < abs(angle_modifier(j)) - abs(angle_modifier(i)) < 10:
            #print(abs(angle_modifier(j)) - abs(angle_modifier(i)))
            start_turn_x.pop(n)
            start_turn_y.pop(n)
            end_turn_x.pop(n)
            end_turn_y.pop(n)
            start_turn_theta.pop(n)
            end_turn_theta.pop(n)
            start_turn_indices.pop(n)
            end_turn_indices.pop(n)
            n-=1
        n+=1
    # Die Startpunkte und Endpunkte werden verglichen, wenn zu nah -> löschen
    n = 0
    for i,j in list(zip(start_turn_indices,end_turn_indices)):
        if j - i < 10 :
            
            start_turn_x.pop(n)
            start_turn_y.pop(n)
            end_turn_x.pop(n)
            end_turn_y.pop(n)
            start_turn_theta.pop(n)
            end_turn_theta.pop(n)
            start_turn_indices.pop(n)
            end_turn_indices.pop(n)
            n-=1
        n+=1
    '''
    n = 0
    for i in (range(len(start_turn_x)-1)):
        if start_turn_indices[n+1] - end_turn_indices[n] > 10 :
            start_turn_x.pop(n+1)
            start_turn_y.pop(n+1)
            start_turn_theta.pop(n+1)
            start_turn_indices.pop(n+1)
            end_turn_x.pop(n)
            end_turn_y.pop(n)
            end_turn_theta.pop(n)
            end_turn_indices.pop(n)
            n-=1
        n+=1'''
    #plot(x,y,start_turn_x,start_turn_y,end_turn_x,end_turn_y)
    start_points = list(zip(start_turn_x,start_turn_y))
    end_points =   list(zip(end_turn_x,end_turn_y))
    angles = list(zip(start_turn_theta,end_turn_theta))
    indices = list(zip(start_turn_indices,end_turn_indices))
    '''for i,j in end_points: # Maximum point of each turn = End punkt
        max_x.append(i)
        max_y.append(j)
    max_points = list(zip(max_x,max_y))'''
    return start_points, end_points, angles, indices, start_turn_x, start_turn_y, end_turn_x, end_turn_y

'''def Bahnanpassung1(x,y,z,points,goal_x,goal_y,goal_theta):

    length_factor = 1.0
    p_1_x = []
    p_1_y = []
    p_1_z = []
    p_2_x = []
    p_2_y = []
    p_2_z = []
    for n,i in enumerate(points[2]): # P1 berechnen
        p_1_x.append(points[0][n][0] + length_factor * m.cos(m.radians(i[0]))) # x[i] = x[i] + cos(z[i])
        p_1_y.append(points[0][n][1] + length_factor * m.sin(m.radians(i[0])))
        p_1_z.append(i[0])
    print('P_1 Points are= '  + str(list(zip(p_1_x,p_1_y))))
    # calculate p2 via Optimization linprog
    #print('Maximum points= '+str(points[4]))
    for n,i in enumerate(points[4]):
        c = [-1 , -1] # cX, (X = x , y) 
        A = [[1 , 0], [0,1]] # A X < b
        b = [(i[0] + 1.0) * m.cos(m.radians(points[2][n][0])) + ((2*(p_1_x[n] + points[1][n][0]))/4.0) * m.sin(m.radians(points[2][n][0])),
             (i[1] + 1.0) * m.sin(m.radians(points[2][n][0])) + ((2*(p_1_y[n] + points[1][n][1]))/4.0) * m.cos(m.radians(points[2][n][0]))] 
        #print(b)
        p_2 = linprog(c, A_ub=A, b_ub=b, # Ein Fehler bei der Berechnung von P2, wenn b einen negativen Wert hat.
                    options={"disp": False, "presolve":True})
        #print("New Average point is : " + str(p_2.x[0]) + ", "+str(p_2.x[1]) )
        p_2_x.append(p_2.x[0])
        p_2_y.append(p_2.x[1]) 
        p_2_x.append(b[0])
        p_2_y.append(b[1])
    print(p_2_x)
    print(p_2_y)
    for i in points[3] :
        p_2_z.append(sum(z[i[0]:i[1]])/float(len(z[i[0]:i[1]])))
    ################################### delete the elements started from start_turn_index
    indices_list = range(points[3][0][0],len(x))
    x = [i for n,i in enumerate(x) if n not in indices_list]
    y = [i for n,i in enumerate(y) if n not in indices_list]
    z = [i for n,i in enumerate(z) if n not in indices_list] 
    
    ##################################  add goal point to the list
    x.append(goal_x)
    y.append(goal_y) 
    z.append(goal_theta)
    ################################## reverse the lists
    p_1_x.reverse()
    p_1_y.reverse()
    p_1_z.reverse()
    p_2_x.reverse()
    p_2_y.reverse()
    p_2_z.reverse()
    ##################################  add avg points to the list before end point
    for i in range(len(points[3])): #insert avg point before end point
        x.insert(points[3][0][0],p_2_x[i])
        y.insert(points[3][0][0],p_2_y[i])
        z.insert(points[3][0][0],p_2_z[i])
        #print(str(i) +"'st  loop "+str(points[3][0][0]))
        #print(x[points[3][0][0]],y[points[3][0][0]])
        #print(x[points[3][0][0]+1],y[points[3][0][0]+1])
        #cover the gap between added avg point and added goal
        #x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],40)]    
        #y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],40)]
        z[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(z[points[3][0][0]],z[points[3][0][0]+1],40)]
        # NLI kommt hier
        result = NLI(x[points[3][0][0]],x[points[3][0][0]+1],y[points[3][0][0]],y[points[3][0][0]+1])
        x[points[3][0][0]+1:points[3][0][0]+1] = [round(l,3) for l in result[0] ]
        y[points[3][0][0]+1:points[3][0][0]+1] = [round(l,3) for l in result[1] ]
        
        #insert start turn point before added avg point
        x.insert(points[3][0][0],p_1_x[i])
        y.insert(points[3][0][0],p_1_y[i])
        z.insert(points[3][0][0],p_1_z[i])
        
        #cover the gap between added start turn point and avg point
        #x[points[3][i][0]+1:points[3][i][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][i][0]],x[points[3][i][0]+1],40)]    
        #y[points[3][i][0]+1:points[3][i][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][i][0]],y[points[3][i][0]+1],40)]
        z[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(z[points[3][0][0]],z[points[3][0][0]+1],40)]
        # NLI kommt hier
        result = NLI(x[points[3][0][0]],x[points[3][0][0]+1],y[points[3][0][0]],y[points[3][0][0]+1])
        x[points[3][0][0]+1:points[3][0][0]+1] = [round(l,3) for l in result[0] ]
        y[points[3][0][0]+1:points[3][0][0]+1] = [round(l,3) for l in result[1] ]
        
    #cover the gap between the point befor start turn point and start turn point
    #x[points[3][0][0]:points[3][0][0]] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]-1],x[points[3][0][0]],40)]
    #y[points[3][0][0]:points[3][0][0]] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]-1],y[points[3][0][0]],40)]
    z[points[3][0][0]:points[3][0][0]] =[round(float(l),3) for l in np.linspace(z[points[3][0][0]-1],z[points[3][0][0]],40)]
    
    # NLI kommt hier
    result = NLI(x[points[3][0][0]-1],x[points[3][0][0]],y[points[3][0][0]-1],y[points[3][0][0]])
    x[points[3][0][0]:points[3][0][0]] = [(l) for l in result[0] ]
    y[points[3][0][0]:points[3][0][0]] = [(l) for l in result[1] ]
    
    # hier kommt die Bezier Kurve
    #(x,y) = Bezier(x,y,5.0/len(x))

    #plot(x_g,y_g,start_turn_x,start_turn_y,end_turn_x,end_turn_y,x,y)'''

def angle_modifier(angle):
    if angle <= 180:
        return angle
    else:
        return (angle % 180) -180
def bernstaein_coef(n,i,u):
    ans = (m.factorial(n) / (m.factorial(i) * m.factorial(n-i))) * (((1-u)**(n-i)) * (u**i))
    return ans
def Bezier(x_points,y_points,iter_):
    n = len(x_points) -1
    u_,x,y = 0,0,0
    x_u,y_u = [],[]
    while u_ < 1 :
        for itrt in range(n+1):
            x += (bernstaein_coef(n,itrt,u_) * x_points[itrt])
            y += (bernstaein_coef(n,itrt,u_) * y_points[itrt])
        x_u.append(round(x,4))
        y_u.append(round(y,4))
        u_+=iter_
        x = 0
        y = 0
    return x_u , y_u
def NLI(point1X, point2X,point1Y,point2Y): # Nicht Lineare Interpolation
    #print(point1X, point2X,point1Y,point2Y)
    a = (point2Y - point1Y)/(np.cosh(point2X) - np.cosh(point1X))
    b = point1Y - a*np.cosh(point1X)
    x = np.linspace(point1X, point2X, 50)
    y = a*np.cosh(x) + b
    return list(x),list(y)
def plot_path_turns(x_g,y_g,start_x,start_y,end_x,end_y):

    plt.plot(x_g,y_g,'b',label='Global path')
    plt.plot(start_x,start_y,'r^',label='Start turn point')
    plt.plot(end_x,end_y,'r>',label='End turn point')
    plt.legend(loc="lower left")
    axes = plt.gca()
    axes.set_xlim([-2,1.5])
    axes.set_ylim([0,5])
    #matplotlib.axes.Axes.set_xlim(xmin=-2,xmax=5)
    #matplotlib.axes.Axes.set_ylim(ymin=-2,ymax=5)
    
    #set_xlim(-2,5)
    #set_ylim(-2,5)
    plt.grid()
    plt.show()

def plot_path_wand(x_g,y_g,wand_x,wand_y):
    plt.plot(x_g,y_g,'b',label='Roboterbahn')
    plt.plot(wand_x,wand_y,'rx',label='Wand')
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
def plot_path(x_g,y_g):
    plt.plot(x_g,y_g,'b',label='Roboterbahn')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
def plot_wand_ClosestPoint_p1_p2(x,y,start_x,start_y,end_x,end_y, wand_x, wand_y, closest_x, closest_y, p_1_x, p_1_y, p_2_x, p_2_y):
    plt.plot(x,y,'b')
    plt.plot(start_x,start_y,'rx',label='Startpunkt')
    plt.plot(end_x,end_y,'ro',label='Endpunkt')
    plt.plot(wand_x,wand_y,'ks',label='Wand')
    plt.plot(closest_x,closest_y,'kX')
    plt.plot(p_1_x,p_1_y,'kX')
    plt.plot(p_2_x,p_2_y,'kX')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
def plot_wand_ClosestPoint_p1(x,y,start_x,start_y,end_x,end_y, wand_x, wand_y, closest_x, closest_y, p_1_x, p_1_y):
    plt.plot(x,y,'b')
    plt.plot(start_x,start_y,'rx',label='Startpunkt')
    plt.plot(end_x,end_y,'ro',label='Endpunkt')
    plt.plot(wand_x,wand_y,'ks',label='Wand')
    plt.plot(closest_x,closest_y,'kX')
    plt.plot(p_1_x,p_1_y,'cX')
    #plt.plot(p_2_x,p_2_y,'kX')
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
def plot_wand_ClosestPoint_p1_p2_modBahn(x,y,start_x,start_y,end_x,end_y, wand_x, wand_y, closest_x1, closest_y1, p_1_x, p_1_y, p_2_x, p_2_y, x_m, y_m):
    plt.plot(x,y,'b',label='Globale Bahn')
    plt.plot(x_m,y_m,'m',label='Angepasste Bahn')
    plt.plot(wand_x,wand_y,'ks',label='Wand')
    plt.plot(start_x,start_y,'rx',label='Startpunkt')
    plt.plot(end_x,end_y,'ro',label='Endpunkt')
    plt.plot(closest_x1,closest_y1,'cX',label='Nearest Neighbor')
    #plt.plot(closest_x2,closest_y2,'cX')
    plt.plot(p_1_x,p_1_y,'cP',label='P1(Hilfspunkte)')
    plt.plot(p_2_x,p_2_y,'yP',label='P2(Hilfspunkte)')
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
def closest_point_func(point,wand_pos):
    tree = spatial.KDTree(wand_pos)
    dist, index = tree.query([point])
    return dist,index
def neighbors_check(wand,radius,points):# Checks the neighbors around the Start and End Points, if within the geiven radius a neighbors found, Performs the BahnAnpassung for that turn, otherwise no BahnAnpassung
    point_neighbors_list = [] 
    tree = spatial.KDTree(wand)
    Start_points = points[0]
    End_points = points[1]
    for n,Start_point in enumerate(Start_points):
        distances1, indices = tree.query(Start_point, p=2, distance_upper_bound=radius)
        distances2, indices = tree.query(End_points[n], p=2, distance_upper_bound=radius)
        if distances1 == m.inf and distances2 == m.inf:
            point_neighbors_list.append(False)
        else:
            point_neighbors_list.append(True)
    n = 0
    for i in point_neighbors_list:
        if not i :
            points[0].pop(n)
            points[1].pop(n)
            points[2].pop(n)
            points[3].pop(n)
            points[4].pop(n)
            points[5].pop(n)
            points[6].pop(n)
            points[7].pop(n)
            n-=1
        n+=1
    return points
def p_1_determination(start_points_coordinate,start_points_angles,angles_p1,wand): # as a list [(x1,y1), (x2,y2), ... ]
    Epsilon = 0.01
    p_1_x = []
    p_1_y = []
    closest_point_x = []
    closest_point_y = []
    n = 0
    for i,j in list(zip(start_points_coordinate,angles_p1)):
        p1_x = i[0] + m.cos(m.radians(start_points_angles[n][0] - 90))
        p1_y = i[1] + m.sin(m.radians(start_points_angles[n][0] - 90))
        #print(p1_x)
        #print(p1_y)
        dist , index = closest_point_func((p1_x,p1_y),wand) # Bestimmung der Closest point#
        while dist < 0.5: # if distance of shifted point (p1) to wall less than 0.5 m is, perform the while loop to increase the distance!
            p1_x -= Epsilon * m.cos(m.radians(start_points_angles[n][0] - 90))
            p1_y -= Epsilon * m.sin(m.radians(start_points_angles[n][0] - 90))
            dist , index = closest_point_func((p1_x,p1_y),wand) 
        if p1_y != i[1] : # wenn p1_y verschoben wurde, berechne die entsprechende p1_x
            p1_x = ((p1_y - i[1])/m.tan(m.radians(j))) + i[0]
        else:
            p1_y = ((p1_x - i[0])*m.tan(m.radians(j))) + i[1]
        n += 1
        p_1_x.append(p1_x)
        p_1_y.append(p1_y)
        closest_point_x.append(wand[index[0]][0])
        closest_point_y.append(wand[index[0]][1])
    #print(p_1_x)
    #print(p_1_y)
    return p_1_x, p_1_y, closest_point_x, closest_point_y

def p_1_determination2(start_points_coordinate,start_points_angles,angles_p1,wand): # as a list [(x1,y1), (x2,y2), ... ]

    Epsilon = 0.1
    p_1_x = []
    p_1_y = []
    closest_point_x = []
    closest_point_y = []
    n = 0
    for i,j in list(zip(start_points_coordinate,angles_p1)):
        p1_x = i[0] + m.cos(m.radians(start_points_angles[n][0] - 90))
        p1_y = i[1] + m.sin(m.radians(start_points_angles[n][0] - 90))
        
        angle = start_points_angles[n][0] - 90
        print(angle)
        print(start_points_angles[n][0] - 20)
        print(i)
        while angle < start_points_angles[n][0] - 20 :
            p1_x += Epsilon * m.cos(m.radians(start_points_angles[n][0]))
            p1_y += Epsilon * m.sin(m.radians(start_points_angles[n][0]))
            
            delta_x = p1_x - i[0]
            delta_y = p1_y - i[1]
            angle = m.degrees(m.atan(delta_y/delta_x))
            #print(angle)
            dist , index = closest_point_func((p1_x,p1_y),wand) # Bestimmung der Closest point#
            #print(dist)
            if dist < 0.5:
                print("Break- Too Close to the Wall!!!") 
                break   
        #print(dist)
        '''dist , index = closest_point_func((p1_x,p1_y),wand) # Bestimmung der Closest point#
        while dist < 0.5: # if distance of shifted point (p1) to wall less than 0.5 m is, perform the while loop to increase the distance!
            p1_x -= Epsilon * m.cos(m.radians(start_points_angles[n][0] - 90))
            p1_y -= Epsilon * m.sin(m.radians(start_points_angles[n][0] - 90))
            dist , index = closest_point_func((p1_x,p1_y),wand)
            print(dist)
            if dist > 1 :
                break''' 
        '''if p1_y != i[1] : # wenn p1_y verschoben wurde, berechne die entsprechende p1_x
            p1_x = ((p1_y - i[1])/m.tan(m.radians(j))) + i[0]
        else:
            p1_y = ((p1_x - i[0])*m.tan(m.radians(j))) + i[1]'''
        n += 1
        p_1_x.append(p1_x)
        p_1_y.append(p1_y)
        closest_point_x.append(wand[index[0]][0])
        closest_point_y.append(wand[index[0]][1])
    #print(p_1_x)
    #print(p_1_y)
    return p_1_x, p_1_y, closest_point_x, closest_point_y

def p_1_determination3(points,minmax,wand):
    global h
    Epsilon = 0.1
    start_point_coordinate = points[0]
    start_point_angle = points[2]
    h = []
    p_1_x = []
    p_1_y = []
    closest_point_x = []
    closest_point_y = []
    # Berechnung vom Abstand
    for i,j,k in list(zip(start_point_coordinate,minmax,start_point_angle)): # Diese For Schleife berechnet nur den Abstand von Start Punkt bis Max Point der Kurve
        x = i[0]
        y = i[1]
        delta_x = j[0] - x
        delta_y = j[1] - y
        angle = m.degrees(m.atan2(delta_y,delta_x))
        #if angle < 0:
        #    angle += 360
        #print(angle)
        #print(j[2])
        while True :# and angle < j[2]+5:    # j[2] = Angle of maximum Point
            if angle > 0 and angle > j[2]:
                #print("iffff While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                #print(angle)
                #print(j[2])
                #print(angle,j[2])
                if  angle < j[2]:
                    #print("Break")
                    break
            if angle > 0 and angle < j[2]:
                #print("if While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                #print(angle)
                #print(j[2])
                #print(angle,j[2])
                if  angle > j[2]:
                    #print("Break")
                    break
            if angle < 0 and angle > angle_modifier(j[2]):
                #print("else1 While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                if  angle < angle_modifier(j[2]):
                    #print("Break")
                    break   
            if angle < 0 and angle < angle_modifier(j[2]):
                #print(angle, angle_modifier(j[2]))
                #print("else2 While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                if  angle > angle_modifier(j[2]):
                    #print("Break")
                    break 
            #if angle < 0:
            #    angle += 360
            
        #print(x,y)
        height = m.sqrt( (x-i[0])**2 + (y-i[1])**2 )
        h.append(height)
        #print(h)
    #h.append(2.3)
    #h.append(3.46)
    n = 0
    for i,j in list(zip(start_point_coordinate,start_point_angle)): # Diese For Schleife bewegt den StartPunkt Schritt für Schritt mit 90 grad Versatz um 1 M mit wall Check
        p1_x = i[0]
        p1_y = i[1]
        dist_to_wall , index = closest_point_func((p1_x,p1_y),wand) # Distance to wall
        dist_to_StartPunkt = 0 # Distance of shifted point to start point is at first 0
        # Bestimmung der Drehung um -90 grad oder +90
        if -10 < j[0] < 10 and j[1] > 180: # if the difference between startangle and endangle of each turn is too large (0 -> 270) this "0" is actually 360
            #j[0] += 360
            if j[1] - (j[0]+360) > 0 : # Turn is CW 
                DrehRichtung = -90
            else: # Turn is CCW
                DrehRichtung = +90
        elif j[1] - j[0] > 0 : # Turn is CW 
            DrehRichtung = -90
        else: # Turn is CCW 
            DrehRichtung = +90
        #print("Drehrichtung ist: " + str(DrehRichtung))
        while dist_to_wall > 0.5 and dist_to_StartPunkt < 0.5:
            p1_x += Epsilon * m.cos(m.radians(j[0] + DrehRichtung))
            p1_y += Epsilon * m.sin(m.radians(j[0] + DrehRichtung))
            dist_to_StartPunkt = m.sqrt((p1_x - i[0])**2 + (p1_y - i[1])**2 ) # Distance to Start Point
            dist_to_wall , index = closest_point_func((p1_x,p1_y),wand) # Distance to wall
        #print(p1_x,p1_y)
        p1_end_x = p1_x + (h[n]/1) * m.cos(m.radians(j[0]))
        p1_end_y = p1_y + (h[n]/1) * m.sin(m.radians(j[0]))
        #print(p1_end_x,p1_end_y)
        while m.sqrt((p1_x - p1_end_x)**2 + (p1_y - p1_end_y)**2 ) > 0.1 and dist_to_wall > 0.4 : # Beweg den verschobenen Start Punkt nach endgueltigen Position von P1 und BREAK wenn der End Punkt erreicht ist ODER too close to wall
            p1_x += Epsilon * (h[n]/1) * m.cos(m.radians(j[0]))
            p1_y += Epsilon * (h[n]/1) * m.sin(m.radians(j[0]))
            dist_to_wall , index = closest_point_func((p1_x,p1_y),wand) # Distance to wall
        p_1_x.append(p1_x)
        p_1_y.append(p1_y)
        closest_point_x.append(wand[index[0]][0])
        closest_point_y.append(wand[index[0]][1])
        n+=1
    return p_1_x, p_1_y, closest_point_x, closest_point_y

def p_2_determination(start_points_coordinate,start_points_angles,angles_p2,wand):
    Epsilon = 0.01
    p_2_x = []
    p_2_y = []
    closest_point_x = []
    closest_point_y = []
    n = 0
    for i,j in list(zip(start_points_coordinate,angles_p2)):
        print(start_points_coordinate)
        p2_x = i[0] + 2.5*m.cos(m.radians(start_points_angles[n][0]))
        p2_y = i[1] + 2.5*m.sin(m.radians(start_points_angles[n][0]))
        print(p2_x,p2_y)
        dist , index = closest_point_func((p2_x,p2_y),wand) # Bestimmung der Closest point#
        while dist < 0.25: # if distance of shifted point (p1) to wall less than 0.5 m is, perform the while loop to increase the distance!
            p2_x -= Epsilon * m.cos(m.radians(start_points_angles[n][0]))
            p2_y -= Epsilon * m.sin(m.radians(start_points_angles[n][0]))
            dist , index = closest_point_func((p2_x,p2_y),wand)
        if p2_y != i[1]: # wenn p2_y verschoben wurde, berechne die entsprechende p2_x
            p2_x = ((p2_y - i[1])/m.tan(m.radians(j))) + i[0]
        else:
            p2_y = ((p2_x - i[0])*m.tan(m.radians(j))) + i[1]
        n += 1
        p_2_x.append(p2_x)
        p_2_y.append(p2_y)
        closest_point_x.append(wand[index[0]][0])
        closest_point_y.append(wand[index[0]][1])
    return p_2_x, p_2_y, closest_point_x, closest_point_y
def p_2_determination3(points,minmax,wand):
    Epsilon = 0.1
    start_point_coordinate = points[0]
    start_point_angle = points[2]
    #h = []
    p_2_x = []
    p_2_y = []
    closest_point_x = []
    closest_point_y = []
    '''for i,j,k in list(zip(start_point_coordinate,minmax,start_point_angle)): # Diese For Schleife berechnet den Abstand bis Max Punkt der Kurve
        x = i[0]
        y = i[1]
        delta_x = j[0] - x
        delta_y = j[1] - y
        angle = m.degrees(m.atan2(delta_y,delta_x))
        #if angle < 0:
        #    angle += 360
        #print(j[2])
        while angle < j[2]:# and angle < j[2]+5:    # j[2] = Angle of maximum Point
            if angle > 0 :
                #print("While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                if  angle > j[2]:
                    #print("Break")
                    break
            else:
                #print("While Schleife...")
                x += Epsilon * m.cos(m.radians(k[0]))
                y += Epsilon * m.sin(m.radians(k[0]))
                delta_x = j[0] - x
                delta_y = j[1] - y
                angle = m.degrees(m.atan2(delta_y,delta_x))
                if  angle < angle_modifier(j[2]):
                    #print("Break")
                    break
        #    if angle < 0:
        #       angle += 360
        #print(x,y)
        height = m.sqrt( (x-i[0])**2 + (y-i[1])**2 )
        h.append(height)'''
    n = 0
    #print(h)
    for i,j in list(zip(start_point_coordinate,start_point_angle)):
        p2_end_x = i[0] + ((h[n] + 1.5) * m.cos(m.radians(j[0])))
        p2_end_y = i[1] + ((h[n] + 1.5) * m.sin(m.radians(j[0])))
        dist_to_wall = 1
        p2_x = i[0]
        p2_y = i[1]
        while dist_to_wall > 0.5 and m.sqrt((p2_x - p2_end_x)**2 + (p2_y - p2_end_y)**2 ) > 0.1:  # Beweg den verschobenen Start Punkt nach endgueltigen Position von P2 und BREAK wenn der End Punkt erreicht ist ODER too close to wall
            p2_x += Epsilon * m.cos(m.radians(start_point_angle[n][0]))
            p2_y += Epsilon * m.sin(m.radians(start_point_angle[n][0]))
            dist_to_wall , index = closest_point_func((p2_x,p2_y),wand)
        p_2_x.append(p2_x)
        p_2_y.append(p2_y)
        closest_point_x.append(wand[index[0]][0])
        closest_point_y.append(wand[index[0]][1])
        n+=1
    return p_2_x, p_2_y, closest_point_x, closest_point_y
def angles_p1_p2(angles_of_turns, p1_angle, p2_angle): # berechnet die Winkel für p1 und p2 für jede Drehung
    angles_p1 = []
    angles_p2 = []
    for i in angles_of_turns:
        diff_angle1 = i[0] - p1_angle # Für P1 wird aktueller Winkel vom gewünschte p1-Winkel abgezogen, weil dieser Punkt immer kleinere Winkel hat.(In Richtung Gegenuhrzeigersinn ). 
        diff_angle2 = i[0] + p2_angle # Für P2 wird aktueller Winkel mit gewünschte p1-Winkel addiert, weil dieser Punkt immer vor dem Start Punkt steht.(In Richtung Uhrzeigersinn) 
        if diff_angle1 < 0:           # ----------------> P1
            angles_p1.append(p1_angle + 270)
        elif 90 > diff_angle1 > 0:
            angles_p1.append(p1_angle)
        elif 180 > diff_angle1 > 90:    
            angles_p1.append(p1_angle + 90)
        elif 270 > diff_angle1 > 180:    
            angles_p1.append(p1_angle + 180)
        if 90 > diff_angle2 > 0 :     # ----------------> P2
            angles_p2.append(p2_angle)
        elif 180 > diff_angle2 > 90:   
            angles_p2.append(p2_angle + 90)
        elif 270 > diff_angle2 > 180:    
            angles_p2.append(p2_angle + 180)
        elif 360 > diff_angle2 > 270:    
            angles_p2.append(p2_angle + 270)
    return angles_p1, angles_p2
def connect_points(points,x,y,z,p_1_x,p_1_y,p_2_x,p_2_y): # connects all points of p1 and p2 to the rest of the path via LINEAR Interpolation and refine via Bezier
    indices_list = range(points[3][0][0],len(x))
    x = [i for n,i in enumerate(x) if n not in indices_list]
    y = [i for n,i in enumerate(y) if n not in indices_list]
    #z = [i for n,i in enumerate(z) if n not in indices_list]
    ##################################  add goal point to the list
    x.append(1)
    y.append(2.25) 
    z.append(0)
    ################################## reverse the lists
    p_1_x.reverse()
    p_1_y.reverse()
    p_2_x.reverse()
    p_2_y.reverse()
    ##################################  add avg points to the list before end point
    for i in range(len(points[3])): 
        x.insert(points[3][0][0],p_2_x[i])  # Insert P2 x cordinate, startet from last turn 
        y.insert(points[3][0][0],p_2_y[i])  # Insert P2 y cordinate, startet from last turn
        #z.insert(points[3][0][0],p_2_z[i])
        #cover the gap between added P2 point and added goal
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],50)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],50)]
        #insert P1 point before added P2 point
        x.insert(points[3][0][0],p_1_x[i])
        y.insert(points[3][0][0],p_1_y[i])
        #z.insert(points[3][0][0],p_1_z[i])
        #cover the gap between added P1 and P2
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],50)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],50)]
    #cover the gap between the point befor start turn point and P1 of first turn point
    x[points[3][0][0]:points[3][0][0]] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]-1],x[points[3][0][0]],50)]
    y[points[3][0][0]:points[3][0][0]] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]-1],y[points[3][0][0]],50)]
    del x[1::2]
    del y[1::2]
    print(len(x),len(y))
    (x,y) = Bezier(x,y,30.0/len(x))
    return x, y
def connect_points2(points,x,y,p_1_x, p_1_y, p_2_x, p_2_y,goal_x,goal_y):
    if len(points[0]) > 1:
        #print(len(x),len(y))
        points[0].append((goal_x,goal_y))  # goal point will added to the start points array
        points[3].append((len(x),len(x)))
        start_points = points[0]
        end_points = points[1]
        angles = points[2]
        indices = points[3]
        start_points.reverse()
        end_points.reverse()
        angles.reverse()
        indices.reverse()
        p_1_x.reverse()
        p_1_y.reverse()
        p_2_x.reverse()
        p_2_y.reverse()
        #print(indices)
        #print(x)
        ################################### 
    
        for n,i in enumerate(end_points):
            dist = m.sqrt((i[0] - start_points[n][0])**2 + (i[1] - start_points[n][1])**2)
            print(dist)
            if dist > 3 :
                '''# Abstand zwischen jedem Endpunkt und nächstem Startpunkt wird in 3 Teilen aufgeteilt.
                schritt = int((indices[n][0]-indices[n+1][1])/3)
                #print(schritt)
                #del x_g[indices[n][0]- 1 * schritt : (indices[n][0] - 0*schritt) - 1 ] #Letzte Drittel (Alte Methode für das Löschen von Elementen) 
                del x[indices[n+1][1] + 2 * schritt : (indices[n+1][1] + 3*schritt) - 1 ] #Letzte Drittel
                del y[indices[n+1][1] + 2 * schritt : (indices[n+1][1] + 3*schritt) - 1 ] #Letzte Drittel
                #print(x)
                x[(indices[n+1][1] + 2 * schritt)-1: indices[n+1][1] + 2*schritt] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][1] + 2 * schritt)-1],x[indices[n+1][1] + 2*schritt],5,endpoint=False)]
                y[(indices[n+1][1] + 2 * schritt)-1: indices[n+1][1] + 2*schritt] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][1] + 2 * schritt)-1],y[indices[n+1][1] + 2*schritt],5,endpoint=False)]
                del x[indices[n+1][1] + 1 * schritt : (indices[n+1][1] + 2*schritt) - 1 ] # Mitte Drittel
                del y[indices[n+1][1] + 1 * schritt : (indices[n+1][1] + 2*schritt) - 1 ] # Mitte Drittel
                #print(x)
                x[(indices[n+1][1] + 1 * schritt)-1: indices[n+1][1] + 1*schritt] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][1] + 1 * schritt)-1],x[indices[n+1][1] + 1*schritt],5,endpoint=False)]
                y[(indices[n+1][1] + 1 * schritt)-1: indices[n+1][1] + 1*schritt] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][1] + 1 * schritt)-1],y[indices[n+1][1] + 1*schritt],5,endpoint=False)]
                #print(x)
                del x[indices[n+1][0] + 0 * schritt : (indices[n+1][1] + 1*schritt) - 1 ] # Erste Drittel
                del y[indices[n+1][0] + 0 * schritt : (indices[n+1][1] + 1*schritt) - 1 ] # Erste Drittel
                #print(x)
                x.insert(indices[n+1][0],p_2_x[n])
                y.insert(indices[n+1][0],p_2_y[n])
                #print(x)
                x[(indices[n+1][0] + 0 * schritt)-0: (indices[n+1][0] + 0*schritt)+1] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][0] + 0 * schritt)-0],x[(indices[n+1][0] + 0*schritt)+1],5,endpoint=False)] # connect the added point (Hier p_2_x) to the next point
                y[(indices[n+1][0] + 0 * schritt)-0: (indices[n+1][0] + 0*schritt)+1] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][0] + 0 * schritt)-0],y[(indices[n+1][0] + 0*schritt)+1],5,endpoint=False)] # connect the added point (Hier p_2_x) to the next point
                
                x.insert(indices[n+1][0],p_1_x[n]) # P_1_x will added before the p_2_x point
                y.insert(indices[n+1][0],p_1_y[n]) # P_1_y will added before the p_2_y point

                x[(indices[n+1][0] + 0 * schritt)-0: (indices[n+1][0] + 0*schritt)+1] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][0] + 0 * schritt)-0],x[(indices[n+1][0] + 0*schritt)+1],5,endpoint=False)]
                y[(indices[n+1][0] + 0 * schritt)-0: (indices[n+1][0] + 0*schritt)+1] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][0] + 0 * schritt)-0],y[(indices[n+1][0] + 0*schritt)+1],5,endpoint=False)]
                x[(indices[n+1][0] + 0 * schritt)-1: indices[n+1][0] + 0*schritt] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][0] + 0 * schritt)-1],x[indices[n+1][0] + 0*schritt],5,endpoint=False)]
                y[(indices[n+1][0] + 0 * schritt)-1: indices[n+1][0] + 0*schritt] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][0] + 0 * schritt)-1],y[indices[n+1][0] + 0*schritt],5,endpoint=False)]
                #if n == len(end_points) -2 :
                #    break
                '''
                # Abstand muss halbiert werden
                #print("For "+str(n)+"'st turn dist is > 3")
                del x[indices[n+1][0]:int((indices[n+1][1]+indices[n][1])/2)]
                del y[indices[n+1][0]:int((indices[n+1][1]+indices[n][1])/2)]
                x.insert(indices[n+1][0],p_2_x[n])
                y.insert(indices[n+1][0],p_2_y[n])
                x[indices[n+1][0]+0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]+0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]],y[indices[n+1][0]+1],150,endpoint=False)]
                
                x.insert(indices[n+1][0],p_1_x[n])
                y.insert(indices[n+1][0],p_1_y[n])
                x[indices[n+1][0]+0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]+0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]],y[indices[n+1][0]+1],150,endpoint=False)]
                
            else:
                print("For "+str(n)+"'st turn dist is < 3")
                #print(n)
                del x[indices[n+1][0]:indices[n][0]]
                del y[indices[n+1][0]:indices[n][0]]
                if n == 0 :
                    x.append(goal_x)
                    y.append(goal_y)
                    #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                    #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
                x.insert(indices[n+1][0],p_2_x[n])
                y.insert(indices[n+1][0],p_2_y[n])
                #print(x[indices[n+1][0]])
                #print(x[indices[n+1][0]+1])
                x[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-0],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-0],y[indices[n+1][0]+1],150,endpoint=False)]
                #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
                x.insert(indices[n+1][0],p_1_x[n])
                y.insert(indices[n+1][0],p_1_y[n])
                x[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-0],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-0],y[indices[n+1][0]+1],150,endpoint=False)]
                #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
            #print(x)
            #if n==0:
            #    break
    else:
        indices_list = range(points[3][0][0],len(x))
        x = [i for n,i in enumerate(x) if n not in indices_list]
        y = [i for n,i in enumerate(y) if n not in indices_list]
        x.append(goal_x)
        y.append(goal_y) 
        x.insert(points[3][0][0],p_2_x[0])  # Insert P2 x cordinate, startet from last turn 
        y.insert(points[3][0][0],p_2_y[0])  # Insert P2 y cordinate, startet from last turn
        #cover the gap between added P2 point and added goal
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],150,endpoint=False)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],150,endpoint=False)]
        x.insert(points[3][0][0],p_1_x[0])
        y.insert(points[3][0][0],p_1_y[0])
        #z.insert(points[3][0][0],p_1_z[i])
        #cover the gap between added P1 and P2
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],150)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],150)]
    #print(x)
    x[points[3][-1][0]-1:points[3][-1][0]+0] =[round(float(l),3) for l in np.linspace(x[points[3][-1][0]-1],x[points[3][-1][0]],150,endpoint=False)]
    y[points[3][-1][0]-1:points[3][-1][0]+0] =[round(float(l),3) for l in np.linspace(y[points[3][-1][0]-1],y[points[3][-1][0]],150,endpoint=False)]
    #print(len(x),len(y))
    del x[1::2]
    del y[1::2]
    del x[1::2]
    del y[1::2]
    print(len(x),len(y))
    #(x,y) = Bezier(x,y,8.0/len(x)) 
    return x, y
def connect_points3(points,x,y,p_1_x, p_1_y, p_2_x, p_2_y,goal_x,goal_y):
    if len(points[0]) > 1:
        #print(len(x),len(y))
        points[0].append((goal_x,goal_y))  # goal point will added to the start points array
        points[3].append((len(x),len(x)))
        start_points = points[0]
        end_points = points[1]
        angles = points[2]
        indices = points[3]
        start_points.reverse()
        end_points.reverse()
        angles.reverse()
        indices.reverse()
        p_1_x.reverse()
        p_1_y.reverse()
        p_2_x.reverse()
        p_2_y.reverse()
        #print(indices)
        #print(x)
        ################################### 
    
        for n,i in enumerate(indices):
            dist = m.sqrt((start_points[n][0] - end_points[n][0])**2 + (start_points[n][1] - end_points[n][1])**2)
            #print(dist)
            if dist > 3 :
                schritt = int(len(x[indices[n+1][1]:i[0]])/3)
                del x[indices[n+1][1] + 2 * schritt : i[0]]
                del y[indices[n+1][1] + 2 * schritt : i[0]]
                if i[0] == i[1]: # letzte Drehung zu Ende der Bahn
                    x.append(goal_x) # NUR fuer letzte Drehung zu Ende der Bahn
                    y.append(goal_y) # NUR fuer letzte Drehung zu Ende der Bahn
                x[(indices[n+1][1] + 2 * schritt)-1 :(indices[n+1][1] + 2 * schritt)+0 ] =[round(float(l),3) for l in np.linspace(x[(indices[n+1][1] + 2 * schritt)-1],x[(indices[n+1][1] + 2 * schritt)+0],150,endpoint=False)]
                y[(indices[n+1][1] + 2 * schritt)-1 :(indices[n+1][1] + 2 * schritt)+0 ] =[round(float(l),3) for l in np.linspace(y[(indices[n+1][1] + 2 * schritt)-1],y[(indices[n+1][1] + 2 * schritt)+0],150,endpoint=False)]

                del x[indices[n+1][0] : indices[n+1][1] + schritt]
                del y[indices[n+1][0] : indices[n+1][1] + schritt]

                x.insert(indices[n+1][0] , p_2_x[n])
                y.insert(indices[n+1][0] , p_2_y[n])
                x[indices[n+1][0] :indices[n+1][0]+1 ] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0] :indices[n+1][0]+1 ] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]],y[indices[n+1][0]+1],150,endpoint=False)]

                x.insert(indices[n+1][0] , p_1_x[n])
                y.insert(indices[n+1][0] , p_1_y[n])
                x[indices[n+1][0] :indices[n+1][0]+1 ] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0] :indices[n+1][0]+1 ] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]],y[indices[n+1][0]+1],150,endpoint=False)]
                
                
            else:
                print("For "+str(n)+"'st turn dist is < 3")
                #print(n)
                del x[indices[n+1][0]:indices[n][0]]
                del y[indices[n+1][0]:indices[n][0]]
                if n == 0 :
                    x.append(goal_x)
                    y.append(goal_y)
                    #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                    #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
                x.insert(indices[n+1][0],p_2_x[n])
                y.insert(indices[n+1][0],p_2_y[n])
                #print(x[indices[n+1][0]])
                #print(x[indices[n+1][0]+1])
                x[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-0],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-0],y[indices[n+1][0]+1],150,endpoint=False)]
                #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
                x.insert(indices[n+1][0],p_1_x[n])
                y.insert(indices[n+1][0],p_1_y[n])
                x[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-0],x[indices[n+1][0]+1],150,endpoint=False)]
                y[indices[n+1][0]-0:indices[n+1][0]+1] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-0],y[indices[n+1][0]+1],150,endpoint=False)]
                #x[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(x[indices[n+1][0]-1],x[indices[n+1][0]+0],150,endpoint=False)]
                #y[indices[n+1][0]-1:indices[n+1][0]+0] =[round(float(l),3) for l in np.linspace(y[indices[n+1][0]-1],y[indices[n+1][0]+0],150,endpoint=False)]
            #print(x)
            #if n==0:
            #    break
            if n == len(indices) -2 :
                break
    else:
        indices_list = range(points[3][0][0],len(x))
        x = [i for n,i in enumerate(x) if n not in indices_list]
        y = [i for n,i in enumerate(y) if n not in indices_list]
        x.append(goal_x)
        y.append(goal_y) 
        x.insert(points[3][0][0],p_2_x[0])  # Insert P2 x cordinate, startet from last turn 
        y.insert(points[3][0][0],p_2_y[0])  # Insert P2 y cordinate, startet from last turn
        #cover the gap between added P2 point and added goal
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],150,endpoint=False)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],150,endpoint=False)]
        x.insert(points[3][0][0],p_1_x[0])
        y.insert(points[3][0][0],p_1_y[0])
        #z.insert(points[3][0][0],p_1_z[i])
        #cover the gap between added P1 and P2
        x[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(x[points[3][0][0]],x[points[3][0][0]+1],150)]    
        y[points[3][0][0]+1:points[3][0][0]+1] =[round(float(l),3) for l in np.linspace(y[points[3][0][0]],y[points[3][0][0]+1],150)]
    #print(x)
    #print(points[3][-1][0]-1,points[3][-1][0]+0)
    x[points[3][-1][0]-1:points[3][-1][0]+0] =[round(float(l),3) for l in np.linspace(x[points[3][-1][0]-1],x[points[3][-1][0]],150,endpoint=False)]
    y[points[3][-1][0]-1:points[3][-1][0]+0] =[round(float(l),3) for l in np.linspace(y[points[3][-1][0]-1],y[points[3][-1][0]],150,endpoint=False)]
    #print(len(x),len(y))
    #del x[1::2]
    #del y[1::2]
    del x[1::2]
    del y[1::2]
    print(len(x),len(y))
    #(x,y) = Bezier(x,y,20.0/len(x)) 
    return x, y
def min_max(x_g, y_g,z_g, points):
    min_max_value = []
    for n,i in enumerate(points[2]):
        if 0 <=  i[0] < 90:
            #print("Quadrant I")
            index = np.argmax(x_g[points[3][n][0]:points[3][n][1]+1])
            x_wert = x_g[points[3][n][0] + index]
            y_wert = y_g[points[3][n][0] + index]
            z_wert = z_g[points[3][n][0] + index]
            min_max_value.append((x_wert,y_wert,z_wert))
        elif 90 <=  i[0] < 180:
            #print("Quadrant II")
            index = np.argmax(y_g[points[3][n][0]:points[3][n][1]])
            x_wert = x_g[points[3][n][0] + index]
            y_wert = y_g[points[3][n][0] + index]
            z_wert = z_g[points[3][n][0] + index]
            min_max_value.append((x_wert,y_wert,z_wert))
        elif 180 <=  i[0] < 270:
            #print("Quadrant III")
            index = np.argmin(x_g[points[3][n][0]:points[3][n][1]])
            x_wert = x_g[points[3][n][0] + index]
            y_wert = y_g[points[3][n][0] + index]
            z_wert = z_g[points[3][n][0] + index]
            min_max_value.append((x_wert,y_wert,z_wert))
        elif 270 <=  i[0] < 360:
            #print("Quadrant IV")
            index = np.argmin(y_g[points[3][n][0]:points[3][n][1]])
            x_wert = x_g[points[3][n][0] + index]
            y_wert = y_g[points[3][n][0] + index]
            z_wert = z_g[points[3][n][0] + index]
            min_max_value.append((x_wert,y_wert,z_wert))
    return min_max_value


