#!/usr/bin/env python
import rospy 
import math
from turtlesim.msg import Pose
from turtlesim.msg import Color
from geometry_msgs.msg import Twist
from std_msgs.msg import String
##################################################### Global Variables
pos_x = 0.0
pos_y = 0.0
pos_theta = 0
##################################################### callback function
def callback_pose(msg):
    global pos_x,pos_y,pos_theta
    pos_x= msg.x
    pos_y= msg.y
    pos_theta=  msg.theta
    vel_x = msg.linear_velocity
    vel_y = msg.angular_velocity
##################################################### Publisher
def publisher():
    global pos_x,pos_y,pos_theta
    pub_msg = Twist()
    pub = rospy.Publisher("turtle1/cmd_vel",Twist,queue_size=1 )
    rate = rospy.Rate(10)
    rospy.sleep(0.1)
    test_list = [1,2,3,0]
    winkel_list = []
    ##################################################################### Referenzwinkellist erstellen
    for i in test_list:
        var =  pos_theta + (i * (math.pi/2.0))
        if var > (2 * math.pi):
            var = var - (2 * math.pi)
        if var < 0 :
            var = var + (2 * math.pi)
        winkel_list.append(var)
    initial_x = pos_x
    initial_y = pos_y
    initial_theta = pos_theta
    print(initial_x,initial_y,initial_theta)
    #####################################################################
    for j in xrange(6):
        rospy.sleep(0.1)
        print(str(j+1)+"st turn started") 
        for i in winkel_list:
            ref_x = pos_x + (2 * math.cos(pos_theta))
            ref_y = pos_y + (2 * math.sin(pos_theta))
            
            while not rospy.is_shutdown():

                pub_msg.linear.x =1
                pub_msg.angular.z = 0.0
                rate.sleep()
                
                distance_ = abs(math.sqrt(((ref_x - pos_x)**2) +((ref_y - pos_y)**2)))
                if distance_ < 0.05 :
                    break
                pub.publish(pub_msg)
            #rospy.sleep(0.5)
            while not rospy.is_shutdown():
                #pub_msg.data = ("actual x position is= "+ str(pos_x)+"actual y position is= "+str(pos_y))
                
                pub_msg = Twist()
                pub_msg.linear.x =0.0
                pub_msg.angular.z = 0.3
                rate.sleep()
                angle_diff =abs( i  - abs(pos_theta))
                
                if angle_diff <0.02:
                    pub_msg.linear.x =0.0
                    pub_msg.angular.z = 0.0
                    pub.publish(pub_msg)
                    break 
                pub.publish(pub_msg) 
        print(str(j+1)+"st turn finished") 
    print("Rotation endet")
    print("Difference on x axis= " + str(abs(initial_x-pos_x)))
    print("Difference on y axis= " + str(abs(initial_y-pos_y)))
    print("Angle Difference= " + str(abs(initial_theta-pos_theta)))
##################################################### MAIN

if __name__ == '__main__':
    rospy.init_node('es_turtle_move', anonymous=False)
    sub = rospy.Subscriber("/turtle1/pose",Pose,callback_pose)
    publisher()
    rospy.spin()