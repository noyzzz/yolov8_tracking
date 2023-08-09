from typing import Dict, List, Optional, Tuple, Union
from tracker_centerer import track_centerizer

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rospy
from my_tracker.msg import ImageDetectionMessage
from std_msgs.msg import String

from stable_baselines3.common.type_aliases import GymStepReturn

from stable_baselines3 import SAC
ENV_IMAGE_LEN = 1

class SimpleMultiObsEnv(gym.Env):

    def __init__(
        self,
        random_start: bool = True,
        discrete_actions: bool = False,
        channel_last: bool = True,
        track_centerizer: track_centerizer = None,
    ):
        super().__init__()
        self.track_centerizer = track_centerizer
        self.vector_size = 5
        if channel_last:
            self.img_size = [84, 84, 1]
        else:
            self.img_size = [1, 84, 84]
        
        self.random_start = random_start
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(0, 1, (4,))

        self.observation_space = spaces.Dict(
            spaces={
                "img1": spaces.Box(0, 255, self.img_size, dtype=np.uint8)
                # ,
                # "img2": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
                # "img3": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
                # "img4": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
            }
        )
        self.last_reset = rospy.get_time()
        # Timeout
        self.max_count = 300
        self.log = ""
        self.state = 0
        self.action2str = ["left", "down", "right", "up"]

    def get_last_obs(self):
        while len(track_centerizer.annotated_frames) < 1:
            #sleep for 0.1 seconds
            rospy.sleep(0.1)
        return track_centerizer.resize_frames(track_centerizer.annotated_frames[-4:])[0].reshape(self.img_size)

    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).

        :param action:
        :return: tuple (observation, reward, terminated, truncated, info).
        """
        while len(track_centerizer.annotated_frames) < ENV_IMAGE_LEN:
            #sleep for 0.1 seconds
            rospy.sleep(0.1)
        if not self.discrete_actions:
            action = np.argmax(action)  # type: ignore[assignment]


        prev_state = self.state

        reward = -1*track_centerizer.get_track_centering_penalty(track_centerizer.tracks_data_parser(track_centerizer.tracks_data), track_centerizer.image_width)
        #print the reward
        # print("reward is : " ,reward)

        resized_frames = track_centerizer.resize_frames(track_centerizer.annotated_frames[-4:])
        # print("resized_frames shape is : ", resized_frames.shape)
        # print("action is : ", action)
        track_centerizer.move_robot(action)

        # print("count is : ", self.count)
        truncated = False
        terminated = False
        if self.last_reset + 15 < rospy.get_time():
            #print resetting the sim
            # #print count and a series of stars
            # print("**********")
            # print("**********")
            # print("resetting the sim")
            # print("**********")
            # print("**********")
            truncated = True
            terminated = True
        # #print the state
        # print("state is : ", self.state)
        # # define state transition
        # if self.state in self.left_possible and action == 0:  # left
        #     self.state -= 1
        # elif self.state in self.down_possible and action == 1:  # down
        #     self.state += self.num_col
        # elif self.state in self.right_possible and action == 2:  # right
        #     self.state += 1
        # elif self.state in self.up_possible and action == 3:  # up
        #     self.state -= self.num_col

        # got_to_end = self.state == self.max_state
        # reward = 1 if got_to_end else reward
        # truncated = self.count > self.max_count
        # terminated = got_to_end

        self.log = f"Went {self.action2str[action]} in state {prev_state}, got to state {self.state}"
        while len(track_centerizer.annotated_frames) < ENV_IMAGE_LEN:
            #sleep for 0.1 seconds
            rospy.sleep(0.1)
        resized_frames = track_centerizer.resize_frames(track_centerizer.annotated_frames[-4:])
        track_centerizer.annotated_frames = track_centerizer.annotated_frames[-4:]
        #resized_frames is a list of 4 frames, convert it to a dict of 4 frames with keys: img1, img2, img3, img4, but reshape the frames to (84, 84,1)
        resized_frames_dict = {"img1": resized_frames[0].reshape(self.img_size)
                            #    ,
                            #     "img2": resized_frames[1].reshape(self.img_size),
                            #       "img3": resized_frames[2].reshape(self.img_size),
                            #         "img4": resized_frames[3].reshape(self.img_size)
                                    }
        return resized_frames_dict, reward, truncated, terminated, {"got_to_end": False}

    def render(self, mode: str = "human") -> None:
        """
        Prints the log of the environment.

        :param mode:
        """
        # print(self.log)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Resets the environment state and step count and returns reset observation.

        :param seed:
        :return: observation dict {'vec': ..., 'img': ...}
        """
        # if seed is not None:
        super().reset(seed=seed)
        #set self.last_reset to the current time
        #print the time difference between the last reset and the current time
        print("time difference is : ", rospy.get_time() - self.last_reset)
        self.last_reset = rospy.get_time()
        # if not self.random_start:
        #     self.state = 0
        # else:
        #     self.state = np.random.randint(0, self.max_state)
        # print(self.get_last_obs().shape)
        # print("self.state_mapping[self.state]", (self.state_mapping[self.state]["img1"]).shape)
        track_centerizer.reset_sim()
        # rospy.sleep(0.3)
        return {"img1": self.get_last_obs()}, {}
    
track_centerizer = track_centerizer()
rospy.init_node('track_centerizer', anonymous=True)
env = SimpleMultiObsEnv(track_centerizer=track_centerizer)
model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log="./sac_tra11cker_tensorboard/")
model = SAC.load("sac_pendulum")
model.set_env(env)
model.learn(total_timesteps=20000, log_interval=1)
model.save("sac_pendulum")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs, info = env.reset()
while not rospy.is_shutdown():
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
# while not rospy.is_shutdown():
#     rospy.spin()