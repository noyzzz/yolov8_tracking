#a ros node that subscribes to gazebo/model_states and publishes to gazebo/set_model_state
#this node is used to move the object in the gazebo world
import rospy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from std_srvs.srv import Empty
import numpy as np
import math
import time

import rospy
from std_msgs.msg import String

person_names = ['person_standing_spehere', 'person_standing_spehere_clone', 'person_standing_spehere_clone_0']
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
t = 0
person2pose = {}

def callback(data):

    #get pose of the objects that have "person" in their name
    #create dictionary name:pose
    for i in range(len(data.name)):
        if "person" in data.name[i]:
            pose = data.pose[i]
            person2pose[data.name[i]] = pose
    #print(person2pose[person_names[0]].position.x)


    
    

    # print the actt = 0ual message in its raw format
    # rospy.loginfo("Here's what was subscribed: %s", data)
      
    # otherwise simply print a convenient message on the terminal
    # print('Data from /topic_name received')
  
class RandomWalk:
    def __init__(self, dt=0.01):
        self.mu = 0.#np.random.uniform(-10.0, 10.0)
        self.sigma = 1.# (np.random.uniform(0.0, 0.1))
        self.dt = dt
        self.ddot = [0.0]
        self.dot = [0.0]
    
    def update(self):
        this_u = np.random.normal(self.mu, self.sigma)
        self.ddot.append(self.ddot[-1] + this_u * self.dt)
        self.dot.append(self.dot[-1] + self.ddot[-1] * self.dt)
        return self.dot[-1]
    
  
def main():
      
    # initialize a node by the name 'listener'.
    # you may choose to name it however you like,
    # since you don't have to use it ahead
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)
      
    # spin() simply keeps python from
    # exiting until this node is stopped
    # rospy.spin()
    t = 0
    #define a publisher for the /gazebo/set_model_state e topic with gazebo_msgs/ModelStat message type
    pub_gazebo = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
    #generate mu and sigma for each person in person2pose array
    dt = 0.01
    random_walks_x = []
    random_walks_y = []
    for i in range(3):
        random_walks_x.append(RandomWalk(dt))
        random_walks_y.append(RandomWalk(dt))

    while not rospy.is_shutdown():
        t = t + dt
        #print(" len(person2pose)" , len(person2pose))
        if len(person2pose)  == 3:
            for i in range(len(person2pose)):
                state_msg = ModelState()
                state_msg.model_name = person_names[i]
                state_msg.pose.position.x = person2pose[person_names[i]].position.x +  random_walks_x[i].update()
                state_msg.pose.position.y = person2pose[person_names[i]].position.y +   random_walks_y[i].update()
                state_msg.pose.position.z = person2pose[person_names[i]].position.z
                state_msg.pose.orientation.x = 0
                state_msg.pose.orientation.y = 0
                state_msg.pose.orientation.z = 0.7512804
                state_msg.pose.orientation.w = 0.6599831
                pub_gazebo.publish(state_msg)
        if t > 1:
            t = 0
            random_walks_x = []
            random_walks_y = []
            for i in range(3):
                random_walks_x.append(RandomWalk(dt))
                random_walks_y.append(RandomWalk(dt))
        #     reset_simulation()
            
        #     while rospy.is_shutdown() == True:
        #         print(rospy.is_shutdown())
        #     print(rospy.is_shutdown())

            
        #     t = 0
        #     rate = rospy.Rate(1) 
        #     rate.sleep()

            
        
        rate = rospy.Rate(30) # 10hz
        rate.sleep()
  
if __name__ == '__main__':
      
    # you could name this function
    try:
        main()
    except rospy.ROSInterruptException:
        print("Shutting down")
        pass