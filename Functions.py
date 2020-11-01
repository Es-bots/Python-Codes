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

    '''if len(start_turn_x) > 1: # pich haye poshte sare ham ruye shibe namozoun yeki mishavand.
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
    # noghate shoru va payane har pich moghayese mishavand, agar kheili nazdik be ham budan pak mishan.
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
    # Dar in ravesh bad az inke be andazeye 1 vahed az start punkt fasele gereftim, ba favasele 0.1 be samte bala harkat mikonim( movazi ba masire harkate robot dar noghte pich)
    # hamzaman ham faseleye noghte jadid ba divar check mishavad, agar fasele kamtar az 0.5M shod break, agar ham zavie noghte jadid ba Start punkt be mizan dade shode resid,
    # ke die while loop ham tamam mishavad, khubie in ravesh ineke noghteye P1 diege az divar rad nemishe, chon mogheyiatesh hamvare check mishe.
    # vali bayad ruye nahveye dadane zavaya kar shavad, chon alan faghat baraye piche 2 kar mikone, Irad mitune tuye Function angles_p1_p2 bashad.
    # Be ehtemale ziad in ravesh baraye halati ke masiere harkat zavayaye 0,90,180,... nadashte bashad ham kar mikonad, chon ruye khate movazie masir harkat mikonim.
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


# First part of path
#x_1 = [0.05, 0.056, 0.061, 0.067, 0.072, 0.078, 0.083, 0.089, 0.094, 0.1, 0.106, 0.111, 0.117, 0.122, 0.128, 0.133, 0.139, 0.144, 0.15, 0.156, 0.161, 0.167, 0.172, 0.178, 0.183, 0.189, 0.194, 0.2, 0.206, 0.211, 0.217, 0.222, 0.228, 0.233, 0.239, 0.244, 0.25, 0.256, 0.261, 0.267, 0.272, 0.278, 0.283, 0.289, 0.294, 0.3, 0.306, 0.311, 0.317, 0.322, 0.328, 0.333, 0.339, 0.344, 0.35, 0.356, 0.361, 0.367, 0.372, 0.378, 0.383, 0.389, 0.394, 0.4, 0.406, 0.411, 0.417, 0.422, 0.428, 0.433, 0.439, 0.444, 0.45, 0.456, 0.461, 0.467, 0.472, 0.478, 0.483, 0.489, 0.494, 0.5, 0.506, 0.511, 0.517, 0.522, 0.528, 0.533, 0.539, 0.544, 0.55, 0.556, 0.561, 0.567, 0.572, 0.578, 0.583, 0.589, 0.594, 0.6, 0.606, 0.611, 0.617, 0.622, 0.628, 0.633, 0.639, 0.644, 0.65, 0.656, 0.661, 0.667, 0.672, 0.678, 0.683, 0.689, 0.694, 0.7, 0.706, 0.711, 0.717, 0.722, 0.728, 0.733, 0.739, 0.744, 0.75, 0.756, 0.761, 0.767, 0.772, 0.778, 0.783, 0.789, 0.794, 0.8, 0.806, 0.811, 0.817, 0.822, 0.828, 0.833, 0.839, 0.844, 0.85, 0.856, 0.861, 0.867, 0.872, 0.878, 0.883, 0.889, 0.894, 0.9, 0.906, 0.911, 0.917, 0.922, 0.928, 0.933, 0.939, 0.944, 0.95, 0.956, 0.961, 0.967, 0.972, 0.978, 0.983, 0.989, 0.994, 1.0, 1.006, 1.011, 1.017, 1.022, 1.028, 1.033, 1.039, 1.044, 1.05, 1.056, 1.061, 1.067, 1.072, 1.078, 1.083, 1.089, 1.094, 1.1, 1.106, 1.111, 1.117, 1.122, 1.128, 1.133, 1.139, 1.144, 1.15, 1.156, 1.161, 1.167, 1.172, 1.178, 1.183, 1.189, 1.194, 1.2, 1.206, 1.211, 1.217, 1.222, 1.228, 1.233, 1.239, 1.244, 1.25, 1.256, 1.261, 1.267, 1.272, 1.278, 1.283, 1.289, 1.294, 1.3, 1.306, 1.311, 1.317, 1.322, 1.328, 1.333, 1.339, 1.344, 1.35, 1.356, 1.361, 1.367, 1.372, 1.378, 1.383, 1.389, 1.394, 1.4, 1.406, 1.411, 1.417, 1.422, 1.428, 1.433, 1.439, 1.444, 1.45, 1.456, 1.461, 1.467, 1.472, 1.478, 1.483, 1.489, 1.494, 1.5, 1.506, 1.511, 1.517, 1.522, 1.528, 1.533, 1.539, 1.544, 1.55, 1.556, 1.561, 1.567, 1.572, 1.578, 1.583, 1.589, 1.594, 1.6, 1.606, 1.611, 1.617, 1.622, 1.628, 1.633, 1.639, 1.644, 1.65, 1.656, 1.661, 1.667, 1.672, 1.678, 1.683, 1.689, 1.694, 1.7, 1.706, 1.711, 1.717, 1.722, 1.728, 1.733, 1.739, 1.744, 1.75, 1.756, 1.761, 1.767, 1.772, 1.778, 1.783, 1.789, 1.794, 1.8, 1.806, 1.811, 1.817, 1.822, 1.828, 1.833, 1.839, 1.844, 1.85, 1.856, 1.861, 1.867, 1.872, 1.878, 1.883, 1.889, 1.894, 1.9, 1.906, 1.911, 1.917, 1.922, 1.928, 1.933, 1.939, 1.944, 1.95, 1.956, 1.961, 1.967, 1.972, 1.978, 1.983, 1.989, 1.994, 2.0, 2.006, 2.011, 2.017, 2.022, 2.028, 2.033, 2.039, 2.044, 2.05, 2.056, 2.061, 2.067, 2.072, 2.078, 2.083, 2.089, 2.094, 2.1, 2.106, 2.111, 2.117, 2.122, 2.128, 2.133, 2.139, 2.144, 2.15, 2.156, 2.161, 2.167, 2.172, 2.178, 2.183, 2.189, 2.194, 2.2, 2.206, 2.211, 2.217, 2.222, 2.228, 2.233, 2.239, 2.244, 2.25, 2.295, 2.34, 2.386, 2.431, 2.476, 2.52, 2.564, 2.608, 2.65, 2.661, 2.672, 2.683, 2.694, 2.706, 2.717, 2.728, 2.739, 2.75, 2.761, 2.772, 2.783, 2.794, 2.806, 2.817, 2.828, 2.839, 2.85, 2.861, 2.872, 2.883, 2.894, 2.906, 2.917, 2.928, 2.939, 2.95, 2.961, 2.972, 2.983, 2.994, 3.006, 3.017, 3.028, 3.039, 3.05, 3.061, 3.072, 3.083, 3.094, 3.106, 3.117, 3.128, 3.139, 3.15, 3.161, 3.172, 3.183, 3.194, 3.206, 3.217, 3.228, 3.239, 3.25, 3.261, 3.272, 3.283, 3.294, 3.306, 3.317, 3.328, 3.339, 3.35, 3.361, 3.372, 3.383, 3.394, 3.406, 3.417, 3.428, 3.439, 3.45, 3.461, 3.472, 3.483, 3.494, 3.506, 3.517, 3.528, 3.539, 3.55, 3.561, 3.572, 3.583, 3.594, 3.606, 3.617, 3.628, 3.639, 3.65, 3.661, 3.672, 3.683, 3.694, 3.706, 3.717, 3.728, 3.739, 3.75, 3.761, 3.772, 3.783, 3.794, 3.806, 3.817, 3.828, 3.839, 3.85, 3.861, 3.872, 3.883, 3.894, 3.906, 3.917, 3.928, 3.939, 3.95, 3.961, 3.972, 3.983, 3.994, 4.006, 4.017, 4.028, 4.039, 4.05, 4.061, 4.072, 4.083, 4.094, 4.106, 4.117, 4.128, 4.139, 4.15, 4.161, 4.172, 4.183, 4.194, 4.206, 4.217, 4.228, 4.239, 4.25, 4.261, 4.272, 4.283, 4.294, 4.306, 4.317, 4.328, 4.339, 4.35, 4.361, 4.372, 4.383, 4.394, 4.406, 4.417, 4.428, 4.439, 4.45, 4.461, 4.472, 4.483, 4.494, 4.506, 4.517, 4.528, 4.539, 4.55, 4.561, 4.572, 4.583, 4.594, 4.606, 4.617, 4.628, 4.639, 4.65, 4.661, 4.672, 4.683, 4.694, 4.706, 4.717, 4.728, 4.739, 4.75, 4.788, 4.825, 4.863, 4.901, 4.939, 4.979, 5.019, 5.059, 5.1, 5.106, 5.111, 5.117, 5.122, 5.128, 5.133, 5.2, 5.3, 6.0]
#y_1 = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.049, 0.046, 0.039, 0.029, 0.016, 0.0, -0.006, -0.011, -0.017, -0.022, -0.028, -0.033, -0.039, -0.044, -0.05, -0.056, -0.061, -0.067, -0.072, -0.078, -0.083, -0.089, -0.094, -0.1, -0.106, -0.111, -0.117, -0.122, -0.128, -0.133, -0.139, -0.144, -0.15, -0.156, -0.161, -0.167, -0.172, -0.178, -0.183, -0.189, -0.194, -0.2, -0.206, -0.211, -0.217, -0.222, -0.228, -0.233, -0.239, -0.244, -0.25, -0.256, -0.261, -0.267, -0.272, -0.278, -0.283, -0.289, -0.294, -0.3, -0.306, -0.311, -0.317, -0.322, -0.328, -0.333, -0.339, -0.344, -0.35, -0.356, -0.361, -0.367, -0.372, -0.378, -0.383, -0.389, -0.394, -0.4, -0.406, -0.411, -0.417, -0.422, -0.428, -0.433, -0.439, -0.444, -0.45, -0.456, -0.461, -0.467, -0.472, -0.478, -0.483, -0.489, -0.494, -0.5, -0.506, -0.511, -0.517, -0.522, -0.528, -0.533, -0.539, -0.544, -0.55, -0.556, -0.561, -0.567, -0.572, -0.578, -0.583, -0.589, -0.594, -0.6, -0.606, -0.611, -0.617, -0.622, -0.628, -0.633, -0.639, -0.644, -0.65, -0.656, -0.661, -0.667, -0.672, -0.678, -0.683, -0.689, -0.694, -0.7, -0.706, -0.711, -0.717, -0.722, -0.728, -0.733, -0.739, -0.744, -0.75, -0.756, -0.761, -0.767, -0.772, -0.778, -0.783, -0.789, -0.794, -0.8, -0.806, -0.811, -0.817, -0.822, -0.828, -0.833, -0.839, -0.844, -0.85, -0.856, -0.861, -0.867, -0.872, -0.878, -0.883, -0.889, -0.894, -0.9, -0.906, -0.911, -0.917, -0.922, -0.928, -0.933, -0.939, -0.944, -0.95, -0.956, -0.961, -0.967, -0.972, -0.978, -0.983, -0.989, -0.994, -1.0, -1.006, -1.011, -1.017, -1.022, -1.028, -1.033, -1.039, -1.044, -1.05, -1.066, -1.081, -1.097, -1.112, -1.126, -1.136, -1.144, -1.148, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15]
#z_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 357.0, 353.0, 349.0, 345.0, 341.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 339.0, 343.0, 347.0, 351.0, 356.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Second part of path
#x_2 = x_g[:800]
#y_2 = y_g[:800]
#z_2 = z_g[:800]
# Third part of path
#x_3 = x_g[900:1500]
#y_3 = y_g[900:1500]
#z_3 = z_g[900:1500]
#fifth
#x_5 = x_g[2400:2800] 
#y_5 = y_g[2400:2800]
#z_5 = z_g[2400:2800]
#sixth
#x_6 = x_g[2750:3100] 
#y_6 = y_g[2750:3100]
#z_6 = z_g[2750:3100]

#verkuerzte Bahn
#x_g = [0.05, 0.061, 0.067, 0.078, 0.083, 0.094, 0.1, 0.111, 0.117, 0.128, 0.133, 0.144, 0.15, 0.161, 0.167, 0.178, 0.183, 0.194, 0.2, 0.211, 0.217, 0.228, 0.233, 0.244, 0.25, 0.261, 0.267, 0.278, 0.283, 0.294, 0.3, 0.311, 0.317, 0.328, 0.333, 0.344, 0.35, 0.361, 0.367, 0.378, 0.383, 0.394, 0.4, 0.411, 0.417, 0.428, 0.433, 0.444, 0.45, 0.461, 0.467, 0.478, 0.483, 0.494, 0.5, 0.511, 0.517, 0.528, 0.533, 0.544, 0.55, 0.561, 0.567, 0.578, 0.583, 0.594, 0.6, 0.611, 0.617, 0.628, 0.633, 0.644, 0.65, 0.661, 0.667, 0.678, 0.683, 0.694, 0.7, 0.711, 0.717, 0.728, 0.733, 0.744, 0.75, 0.761, 0.767, 0.778, 0.783, 0.794, 0.8, 0.811, 0.817, 0.828, 0.833, 0.844, 0.85, 0.861, 0.867, 0.878, 0.883, 0.894, 0.9, 0.911, 0.917, 0.928, 0.933, 0.944, 0.95, 0.961, 0.967, 0.978, 0.983, 0.994, 1.0, 1.011, 1.017, 1.028, 1.033, 1.044, 1.05, 1.061, 1.067, 1.078, 1.083, 1.094, 1.1, 1.111, 1.117, 1.128, 1.133, 1.144, 1.15, 1.161, 1.167, 1.178, 1.183, 1.194, 1.2, 1.211, 1.217, 1.228, 1.233, 1.244, 1.25, 1.261, 1.267, 1.278, 1.283, 1.294, 1.3, 1.311, 1.317, 1.328, 1.333, 1.344, 1.35, 1.361, 1.367, 1.378, 1.383, 1.394, 1.4, 1.411, 1.417, 1.428, 1.433, 1.444, 1.45, 1.461, 1.467, 1.478, 1.483, 1.494, 1.5, 1.511, 1.517, 1.528, 1.533, 1.544, 1.55, 1.561, 1.567, 1.578, 1.583, 1.594, 1.6, 1.611, 1.617, 1.628, 1.633, 1.644, 1.65, 1.661, 1.667, 1.678, 1.683, 1.694, 1.7, 1.711, 1.717, 1.728, 1.733, 1.744, 1.75, 1.761, 1.767, 1.778, 1.783, 1.794, 1.8, 1.811, 1.817, 1.828, 1.833, 1.844, 1.85, 1.861, 1.867, 1.878, 1.883, 1.894, 1.9, 1.911, 1.917, 1.928, 1.933, 1.944, 1.95, 1.961, 1.967, 1.978, 1.983, 1.994, 2.0, 2.011, 2.017, 2.028, 2.033, 2.044, 2.05, 2.061, 2.067, 2.078, 2.083, 2.094, 2.1, 2.111, 2.117, 2.128, 2.133, 2.144, 2.15, 2.161, 2.167, 2.178, 2.183, 2.194, 2.2, 2.211, 2.217, 2.228, 2.233, 2.244, 2.25, 2.34, 2.386, 2.476, 2.52, 2.608, 2.65, 2.672, 2.683, 2.706, 2.717, 2.739, 2.75, 2.772, 2.783, 2.806, 2.817, 2.839, 2.85, 2.872, 2.883, 2.906, 2.917, 2.939, 2.95, 2.972, 2.983, 3.006, 3.017, 3.039, 3.05, 3.072, 3.083, 3.106, 3.117, 3.139, 3.15, 3.172, 3.183, 3.206, 3.217, 3.239, 3.25, 3.272, 3.283, 3.306, 3.317, 3.339, 3.35, 3.372, 3.383, 3.406, 3.417, 3.439, 3.45, 3.472, 3.483, 3.506, 3.517, 3.539, 3.55, 3.572, 3.583, 3.606, 3.617, 3.639, 3.65, 3.672, 3.683, 3.706, 3.717, 3.739, 3.75, 3.772, 3.783, 3.806, 3.817, 3.839, 3.85, 3.872, 3.883, 3.906, 3.917, 3.939, 3.95, 3.972, 3.983, 4.006, 4.017, 4.039, 4.05, 4.072, 4.083, 4.106, 4.117, 4.139, 4.15, 4.172, 4.183, 4.206, 4.217, 4.239, 4.25, 4.272, 4.283, 4.306, 4.317, 4.339, 4.35, 4.372, 4.383, 4.406, 4.417, 4.439, 4.45, 4.472, 4.483, 4.506, 4.517, 4.539, 4.55, 4.572, 4.583, 4.606, 4.617, 4.639, 4.65, 4.672, 4.683, 4.706, 4.717, 4.739, 4.75, 4.825, 4.863, 4.939, 4.979, 5.059, 5.1, 5.111, 5.117, 5.128, 5.133, 5.144, 5.15, 5.161, 5.167, 5.178, 5.183, 5.194, 5.2, 5.211, 5.217, 5.228, 5.233, 5.244, 5.25, 5.34, 5.386, 5.476, 5.52, 5.608, 5.65, 5.709, 5.738, 5.794, 5.822, 5.875, 5.9, 5.968, 6.0, 6.058, 6.085, 6.13, 6.15, 6.181, 6.197, 6.226, 6.236, 6.249, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.249, 6.239, 6.216, 6.194, 6.183, 6.172, 6.161, 6.15, 6.112, 6.07, 6.025, 5.976, 5.944, 5.933, 5.922, 5.911, 5.9, 5.832, 5.758, 5.679, 5.594, 5.512, 5.437, 5.361, 5.281, 5.2, 5.189, 5.178, 5.167, 5.156, 5.144, 5.133, 5.122, 5.111, 5.1, 5.089, 5.078, 5.067, 5.056, 5.044, 5.033, 5.022, 5.011, 5.0, 4.989, 4.978, 4.967, 4.956, 4.944, 4.933, 4.922, 4.911, 4.9, 4.889, 4.878, 4.867, 4.856, 4.844, 4.833, 4.822, 4.811, 4.8, 4.789, 4.778, 4.767, 4.756, 4.744, 4.733, 4.722, 4.711, 4.7, 4.689, 4.678, 4.667, 4.656, 4.644, 4.633, 4.622, 4.611, 4.6, 4.589, 4.578, 4.567, 4.556, 4.544, 4.533, 4.522, 4.511, 4.5, 4.489, 4.478, 4.467, 4.456, 4.444, 4.433, 4.422, 4.411, 4.4, 4.389, 4.378, 4.367, 4.356, 4.344, 4.333, 4.322, 4.311, 4.3, 4.289, 4.278, 4.267, 4.256, 4.244, 4.233, 4.222, 4.211, 4.2, 4.189, 4.178, 4.167, 4.156, 4.144, 4.133, 4.122, 4.111, 4.1, 4.089, 4.078, 4.067, 4.056, 4.044, 4.033, 4.022, 4.011, 4.0, 3.989, 3.978, 3.967, 3.956, 3.944, 3.933, 3.922, 3.911, 3.9, 3.889, 3.878, 3.867, 3.856, 3.844, 3.833, 3.822, 3.811, 3.8, 3.789, 3.778, 3.767, 3.756, 3.744, 3.733, 3.722, 3.711, 3.7, 3.689, 3.678, 3.667, 3.656, 3.644, 3.633, 3.622, 3.611, 3.6, 3.589, 3.578, 3.567, 3.556, 3.544, 3.533, 3.522, 3.511, 3.5, 3.489, 3.478, 3.467, 3.456, 3.444, 3.433, 3.422, 3.411, 3.4, 3.389, 3.378, 3.367, 3.356, 3.344, 3.333, 3.322, 3.311, 3.3, 3.289, 3.278, 3.267, 3.256, 3.244, 3.233, 3.222, 3.211, 3.2, 3.189, 3.178, 3.167, 3.156, 3.144, 3.133, 3.122, 3.111, 3.1, 3.089, 3.078, 3.067, 3.056, 3.044, 3.033, 3.022, 3.011, 3.0, 2.989, 2.978, 2.967, 2.956, 2.944, 2.933, 2.922, 2.911, 2.9, 2.889, 2.878, 2.867, 2.856, 2.844, 2.833, 2.822, 2.811, 2.8, 2.789, 2.778, 2.767, 2.756, 2.744, 2.733, 2.722, 2.711, 2.7, 2.689, 2.678, 2.667, 2.656, 2.644, 2.633, 2.622, 2.611, 2.6, 2.589, 2.578, 2.567, 2.556, 2.544, 2.533, 2.522, 2.511, 2.5, 2.489, 2.478, 2.467, 2.456, 2.444, 2.433, 2.422, 2.411, 2.4, 2.389, 2.378, 2.367, 2.356, 2.344, 2.333, 2.322, 2.311, 2.3, 2.289, 2.278, 2.267, 2.256, 2.244, 2.233, 2.222, 2.211, 2.2, 2.189, 2.178, 2.167, 2.156, 2.144, 2.133, 2.122, 2.111, 2.1, 2.089, 2.078, 2.067, 2.056, 2.005, 1.915, 1.824, 1.736, 1.65, 1.628, 1.606, 1.583, 1.561, 1.539, 1.517, 1.494, 1.472, 1.45, 1.428, 1.406, 1.383, 1.361, 1.339, 1.317, 1.294, 1.272, 1.25, 1.228, 1.206, 1.183, 1.161, 1.139, 1.117, 1.094, 1.072, 1.05, 1.028, 1.006, 0.983, 0.961, 0.939, 0.917, 0.894, 0.872, 0.85, 0.828, 0.806, 0.783, 0.761, 0.739, 0.717, 0.694, 0.672, 0.65, 0.628, 0.606, 0.583, 0.561, 0.539, 0.517, 0.494, 0.472, 0.45, 0.428, 0.406, 0.383, 0.361, 0.339, 0.317, 0.294, 0.272, 0.25, 0.228, 0.206, 0.183, 0.161, 0.139, 0.117, 0.094, 0.072, 0.05, 0.028, 0.006, -0.017, -0.039, -0.061, -0.083, -0.106, -0.128, -0.15, -0.172, -0.194, -0.217, -0.239, -0.261, -0.283, -0.306, -0.328, -0.35, -0.372, -0.394, -0.417, -0.439, -0.461, -0.483, -0.506, -0.528, -0.55, -0.572, -0.594, -0.617, -0.639, -0.661, -0.683, -0.706, -0.728, -0.75, -0.772, -0.794, -0.817, -0.839, -0.888, -0.963, -1.039, -1.119, -1.2, -1.211, -1.222, -1.233, -1.244, -1.256, -1.267, -1.278, -1.289, -1.3, -1.311, -1.322, -1.333, -1.344, -1.356, -1.367, -1.378, -1.389, -1.4, -1.411, -1.422, -1.433, -1.444, -1.456, -1.467, -1.478, -1.489, -1.5, -1.511, -1.522, -1.533, -1.544, -1.556, -1.567, -1.578, -1.589, -1.6, -1.611, -1.622, -1.633, -1.644, -1.656, -1.667, -1.678, -1.689, -1.7, -1.711, -1.722, -1.733, -1.744, -1.756, -1.767, -1.778, -1.789, -1.8, -1.811, -1.822, -1.833, -1.844, -1.856, -1.867, -1.878, -1.889, -1.9, -1.911, -1.922, -1.933, -1.944, -1.956, -1.967, -1.978, -1.989, -2.0, -2.011, -2.022, -2.033, -2.044, -2.056, -2.067, -2.078, -2.089, -2.1, -2.111, -2.122, -2.133, -2.144, -2.156, -2.167, -2.178, -2.189, -2.2, -2.211, -2.222, -2.233, -2.244, -2.256, -2.267, -2.278, -2.289, -2.3, -2.311, -2.322, -2.333, -2.344, -2.356, -2.367, -2.378, -2.389, -2.4, -2.411, -2.422, -2.433, -2.444, -2.456, -2.467, -2.478, -2.489, -2.5, -2.511, -2.522, -2.533, -2.544, -2.556, -2.567, -2.578, -2.589, -2.6, -2.611, -2.622, -2.633, -2.644, -2.656, -2.667, -2.678, -2.689, -2.7, -2.711, -2.722, -2.733, -2.744, -2.756, -2.767, -2.778, -2.789, -2.8, -2.811, -2.822, -2.833, -2.844, -2.856, -2.867, -2.878, -2.889, -2.9, -2.911, -2.922, -2.933, -2.944, -2.956, -2.967, -2.978, -2.989, -3.0, -3.011, -3.022, -3.033, -3.044, -3.056, -3.067, -3.078, -3.089, -3.1, -3.111, -3.122, -3.133, -3.144, -3.156, -3.167, -3.178, -3.189, -3.2, -3.211, -3.222, -3.233, -3.244, -3.256, -3.267, -3.278, -3.289, -3.3, -3.311, -3.322, -3.333, -3.344, -3.356, -3.367, -3.378, -3.389, -3.4, -3.411, -3.422, -3.433, -3.444, -3.456, -3.467, -3.478, -3.489, -3.5, -3.511, -3.522, -3.533, -3.544, -3.556, -3.567, -3.578, -3.589, -3.6, -3.611, -3.622, -3.633, -3.644, -3.656, -3.667, -3.678, -3.689, -3.7, -3.711, -3.722, -3.733, -3.744, -3.756, -3.767, -3.778, -3.789, -3.8, -3.811, -3.822, -3.833, -3.844, -3.856, -3.867, -3.878, -3.889, -3.9, -3.911, -3.922, -3.933, -3.944, -3.956, -3.967, -3.978, -3.989, -4.0, -4.011, -4.022, -4.033, -4.044, -4.056, -4.067, -4.078, -4.089, -4.1, -4.111, -4.122, -4.133, -4.144, -4.156, -4.167, -4.178, -4.189, -4.2, -4.29, -4.381, -4.47, -4.558, -4.63, -4.688, -4.744, -4.798, -4.85, -4.861, -4.872, -4.883, -4.894, -4.934, -5.0, -5.058, -5.109, -5.15, -5.181, -5.212, -5.236, -5.248, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.25, -5.245, -5.229, -5.2, -5.162, -5.12, -5.075, -5.026, -4.994, -4.983, -4.972, -4.961, -4.95, -4.939, -4.928, -4.917, -4.906, -4.866, -4.796, -4.719, -4.637, -4.55, -4.528, -4.506, -4.483, -4.461, -4.439, -4.417, -4.394, -4.372, -4.35, -4.328, -4.306, -4.283, -4.261, -4.239, -4.217, -4.194, -4.172, -4.15, -4.128, -4.106, -4.083, -4.061, -4.039, -4.017, -3.994, -3.972, -3.95, -3.928, -3.906, -3.883, -3.861, -3.839, -3.817, -3.794, -3.772, -3.75, -3.728, -3.706, -3.683, -3.661, -3.639, -3.617, -3.594, -3.572, -3.55, -3.475, -3.399, -3.321, -3.241, -3.194, -3.183, -3.172, -3.161, -3.15, -3.139, -3.128, -3.117, -3.106, -3.094, -3.083, -3.072, -3.061, -3.05, -3.039, -3.028, -3.017, -3.006, -2.994, -2.983, -2.972, -2.961, -2.95, -2.939, -2.928, -2.917, -2.906, -2.894, -2.883, -2.872, -2.861, -2.85, -2.839, -2.828, -2.817, -2.806, -2.794, -2.783, -2.772, -2.761, -2.75, -2.739, -2.728, -2.717, -2.706, -2.694, -2.683, -2.672, -2.661, -2.65, -2.639, -2.628, -2.617, -2.606, -2.594, -2.583, -2.572, -2.561, -2.55, -2.539, -2.528, -2.517, -2.506, -2.494, -2.483, -2.472, -2.461, -2.45, -2.439, -2.428, -2.417, -2.406, -2.394, -2.383, -2.372, -2.361, -2.35, -2.339, -2.328, -2.317, -2.306, -2.294, -2.283, -2.272, -2.261, -2.25, -2.239, -2.228, -2.217, -2.206, -2.194, -2.183, -2.172, -2.161, -2.15, -2.139, -2.128, -2.117, -2.106, -2.094, -2.083, -2.072, -2.061, -2.05, -2.039, -2.028, -2.017, -2.006, -1.994, -1.983, -1.972, -1.961, -1.95, -1.939, -1.928, -1.917, -1.906, -1.894, -1.883, -1.872, -1.861, -1.85, -1.839, -1.828, -1.817, -1.806, -1.794, -1.783, -1.772, -1.761, -1.75, -1.739, -1.728, -1.717, -1.706, -1.694, -1.683, -1.672, -1.661, -1.65, -1.639, -1.628, -1.617, -1.606, -1.594, -1.583, -1.572, -1.561, -1.55, -1.539, -1.528, -1.517, -1.506, -1.494, -1.483, -1.472, -1.461, -1.45, -1.439, -1.428, -1.417, -1.406, -1.394, -1.383, -1.372, -1.361, -1.35, -1.339, -1.328, -1.317, -1.306, -1.294, -1.283, -1.272, -1.261, -1.25, -1.239, -1.228, -1.217, -1.206, -1.194, -1.183, -1.172, -1.161, -1.15, -1.139, -1.128, -1.117, -1.106, -1.094, -1.083, -1.072, -1.061, -1.05, -1.039, -1.028, -1.017, -1.006, -0.994, -0.983, -0.972, -0.961, -0.95, -0.939, -0.928, -0.917, -0.906, -0.894, -0.883, -0.872, -0.861, -0.85, -0.839, -0.828, -0.817, -0.806, -0.794, -0.783, -0.772, -0.761, -0.75, -0.739, -0.728, -0.717, -0.706, -0.694, -0.683, -0.672, -0.661, -0.65, -0.639, -0.628, -0.617, -0.606, -0.594, -0.583, -0.572, -0.561, -0.55, -0.539, -0.528, -0.517, -0.506, -0.494, -0.483, -0.472, -0.461, -0.45, -0.439, -0.428, -0.417, -0.406, -0.394, -0.383, -0.372, -0.361, -0.35, -0.339, -0.328, -0.317, -0.306, -0.294, -0.283, -0.272, -0.261, -0.25, -0.239, -0.228, -0.217, -0.206, -0.194, -0.183, -0.172, -0.161, -0.15, -0.139, -0.128, -0.117, -0.106, -0.094, -0.083, -0.072, -0.061, -0.05, -0.039, -0.028, -0.017, -0.006, 0.006, 0.017, 0.028, 0.039, 0.05, 0.061, 0.072, 0.083, 0.094]
#y_g = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.046, 0.039, 0.016, 0.0, -0.011, -0.017, -0.028, -0.033, -0.044, -0.05, -0.061, -0.067, -0.078, -0.083, -0.094, -0.1, -0.111, -0.117, -0.128, -0.133, -0.144, -0.15, -0.161, -0.167, -0.178, -0.183, -0.194, -0.2, -0.211, -0.217, -0.228, -0.233, -0.244, -0.25, -0.261, -0.267, -0.278, -0.283, -0.294, -0.3, -0.311, -0.317, -0.328, -0.333, -0.344, -0.35, -0.361, -0.367, -0.378, -0.383, -0.394, -0.4, -0.411, -0.417, -0.428, -0.433, -0.444, -0.45, -0.461, -0.467, -0.478, -0.483, -0.494, -0.5, -0.511, -0.517, -0.528, -0.533, -0.544, -0.55, -0.561, -0.567, -0.578, -0.583, -0.594, -0.6, -0.611, -0.617, -0.628, -0.633, -0.644, -0.65, -0.661, -0.667, -0.678, -0.683, -0.694, -0.7, -0.711, -0.717, -0.728, -0.733, -0.744, -0.75, -0.761, -0.767, -0.778, -0.783, -0.794, -0.8, -0.811, -0.817, -0.828, -0.833, -0.844, -0.85, -0.861, -0.867, -0.878, -0.883, -0.894, -0.9, -0.911, -0.917, -0.928, -0.933, -0.944, -0.95, -0.961, -0.967, -0.978, -0.983, -0.994, -1.0, -1.011, -1.017, -1.028, -1.033, -1.044, -1.05, -1.081, -1.097, -1.126, -1.136, -1.148, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.15, -1.145, -1.139, -1.116, -1.1, -1.062, -1.042, -0.998, -0.975, -0.926, -0.9, -0.832, -0.796, -0.719, -0.679, -0.594, -0.55, -0.475, -0.437, -0.361, -0.321, -0.241, -0.2, -0.189, -0.183, -0.172, -0.167, -0.156, -0.15, -0.139, -0.133, -0.122, -0.117, -0.106, -0.1, -0.089, -0.083, -0.072, -0.067, -0.056, -0.05, -0.039, -0.033, -0.022, -0.017, -0.011, 0.0, 0.011, 0.022, 0.033, 0.044, 0.056, 0.067, 0.078, 0.089, 0.1, 0.111, 0.122, 0.133, 0.144, 0.156, 0.167, 0.178, 0.189, 0.2, 0.211, 0.222, 0.233, 0.244, 0.256, 0.267, 0.278, 0.289, 0.3, 0.311, 0.322, 0.333, 0.344, 0.356, 0.367, 0.378, 0.389, 0.4, 0.411, 0.422, 0.433, 0.444, 0.456, 0.467, 0.478, 0.489, 0.5, 0.511, 0.522, 0.533, 0.544, 0.556, 0.567, 0.578, 0.589, 0.6, 0.611, 0.622, 0.633, 0.644, 0.656, 0.667, 0.678, 0.689, 0.7, 0.711, 0.722, 0.733, 0.744, 0.756, 0.767, 0.778, 0.789, 0.8, 0.811, 0.822, 0.833, 0.844, 0.856, 0.867, 0.878, 0.889, 0.9, 0.911, 0.922, 0.933, 0.944, 0.956, 0.967, 0.978, 0.989, 1.0, 1.011, 1.022, 1.033, 1.044, 1.056, 1.067, 1.078, 1.089, 1.1, 1.111, 1.122, 1.133, 1.144, 1.156, 1.167, 1.178, 1.189, 1.2, 1.211, 1.222, 1.233, 1.244, 1.256, 1.267, 1.278, 1.289, 1.3, 1.311, 1.322, 1.333, 1.344, 1.356, 1.367, 1.378, 1.389, 1.4, 1.411, 1.422, 1.433, 1.444, 1.456, 1.467, 1.478, 1.489, 1.5, 1.511, 1.522, 1.533, 1.544, 1.556, 1.567, 1.578, 1.589, 1.6, 1.611, 1.622, 1.633, 1.644, 1.656, 1.667, 1.678, 1.689, 1.7, 1.711, 1.722, 1.733, 1.744, 1.756, 1.767, 1.778, 1.789, 1.8, 1.811, 1.822, 1.833, 1.844, 1.856, 1.867, 1.878, 1.889, 1.9, 1.911, 1.922, 1.933, 1.944, 1.956, 1.967, 1.978, 1.989, 2.0, 2.011, 2.022, 2.033, 2.044, 2.056, 2.067, 2.078, 2.089, 2.1, 2.111, 2.122, 2.133, 2.144, 2.156, 2.167, 2.178, 2.189, 2.2, 2.211, 2.222, 2.233, 2.244, 2.256, 2.267, 2.278, 2.289, 2.3, 2.311, 2.322, 2.333, 2.344, 2.356, 2.367, 2.378, 2.389, 2.4, 2.411, 2.422, 2.433, 2.444, 2.456, 2.467, 2.478, 2.489, 2.5, 2.511, 2.522, 2.533, 2.544, 2.556, 2.567, 2.578, 2.589, 2.6, 2.611, 2.622, 2.633, 2.644, 2.656, 2.667, 2.678, 2.689, 2.7, 2.711, 2.722, 2.733, 2.744, 2.756, 2.767, 2.778, 2.789, 2.8, 2.811, 2.822, 2.833, 2.844, 2.856, 2.867, 2.878, 2.889, 2.9, 2.911, 2.922, 2.933, 2.944, 2.956, 2.967, 2.978, 2.989, 3.0, 3.09, 3.181, 3.27, 3.358, 3.411, 3.433, 3.456, 3.478, 3.5, 3.559, 3.617, 3.672, 3.725, 3.756, 3.767, 3.778, 3.789, 3.8, 3.868, 3.93, 3.985, 4.03, 4.066, 4.097, 4.126, 4.144, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.15, 4.155, 4.171, 4.2, 4.211, 4.222, 4.233, 4.244, 4.256, 4.267, 4.278, 4.289, 4.3, 4.311, 4.322, 4.333, 4.344, 4.356, 4.367, 4.378, 4.389, 4.4, 4.411, 4.422, 4.433, 4.444, 4.456, 4.467, 4.478, 4.489, 4.5, 4.511, 4.522, 4.533, 4.544, 4.556, 4.567, 4.578, 4.589, 4.6, 4.611, 4.622, 4.633, 4.644, 4.656, 4.667, 4.678, 4.689, 4.7, 4.711, 4.722, 4.733, 4.744, 4.756, 4.767, 4.778, 4.789, 4.8, 4.811, 4.822, 4.833, 4.844, 4.856, 4.867, 4.878, 4.889, 4.9, 4.911, 4.922, 4.933, 4.944, 4.956, 4.967, 4.978, 4.989, 5.0, 5.011, 5.022, 5.033, 5.044, 5.056, 5.067, 5.078, 5.089, 5.1, 5.111, 5.122, 5.133, 5.144, 5.156, 5.167, 5.178, 5.189, 5.2, 5.211, 5.222, 5.233, 5.244, 5.256, 5.267, 5.278, 5.289, 5.3, 5.311, 5.322, 5.333, 5.344, 5.356, 5.367, 5.378, 5.389, 5.4, 5.411, 5.422, 5.433, 5.444, 5.466, 5.497, 5.526, 5.544, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.55, 5.549, 5.539, 5.516, 5.482, 5.442, 5.398, 5.351, 5.3, 5.289, 5.278, 5.267, 5.256, 5.216, 5.146, 5.069, 4.987, 4.9, 4.825, 4.749, 4.671, 4.591, 4.544, 4.533, 4.522, 4.511, 4.5, 4.489, 4.478, 4.467, 4.456, 4.444, 4.433, 4.422, 4.411, 4.4, 4.389, 4.378, 4.367, 4.356, 4.344, 4.333, 4.322, 4.311, 4.3, 4.289, 4.278, 4.267, 4.256, 4.244, 4.233, 4.222, 4.211, 4.2, 4.189, 4.178, 4.167, 4.156, 4.144, 4.133, 4.122, 4.111, 4.1, 4.089, 4.078, 4.067, 4.056, 4.044, 4.033, 4.022, 4.011, 4.0, 3.989, 3.978, 3.967, 3.956, 3.944, 3.933, 3.922, 3.911, 3.9, 3.889, 3.878, 3.867, 3.856, 3.805, 3.715, 3.624, 3.536, 3.45, 3.391, 3.334, 3.278, 3.226, 3.194, 3.183, 3.172, 3.161, 3.15, 3.139, 3.128, 3.117, 3.106, 3.066, 3.0, 2.942, 2.891, 2.85, 2.839, 2.828, 2.817, 2.806, 2.794, 2.783, 2.772, 2.761, 2.75, 2.739, 2.728, 2.717, 2.706, 2.694, 2.683, 2.672, 2.661, 2.65, 2.639, 2.628, 2.617, 2.606, 2.594, 2.583, 2.572, 2.561, 2.55, 2.539, 2.528, 2.517, 2.506, 2.494, 2.483, 2.472, 2.461, 2.45, 2.439, 2.428, 2.417, 2.406, 2.394, 2.383, 2.372, 2.361, 2.35, 2.319, 2.288, 2.264, 2.252, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25, 2.25]
#z_g = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 353.0, 349.0, 341.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 343.0, 347.0, 356.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 11.0, 19.0, 23.0, 26.0, 28.0, 31.0, 33.0, 36.0, 45.0, 47.0, 50.0, 56.0, 59.0, 65.0, 68.0, 68.0, 68.0, 73.0, 77.0, 86.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 93.0, 101.0, 109.0, 113.0, 113.0, 113.0, 113.0, 113.0, 116.0, 119.0, 123.0, 126.0, 135.0, 135.0, 135.0, 135.0, 135.0, 137.0, 143.0, 149.0, 155.0, 158.0, 158.0, 163.0, 171.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 173.0, 165.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 158.0, 163.0, 171.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 183.0, 191.0, 199.0, 204.0, 208.0, 211.0, 215.0, 225.0, 225.0, 225.0, 225.0, 225.0, 225.0, 230.0, 236.0, 242.0, 248.0, 248.0, 249.0, 257.0, 266.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 270.0, 277.0, 285.0, 293.0, 296.0, 299.0, 303.0, 306.0, 315.0, 315.0, 315.0, 315.0, 315.0, 315.0, 315.0, 315.0, 315.0, 315.0, 320.0, 326.0, 332.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 338.0, 339.0, 347.0, 356.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#print(len(x_g),len(y_g))