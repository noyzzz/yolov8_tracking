from tracker_centerer import *
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.type_aliases import GymStepReturn

class SimpleMultiObsEnv(gym.Env):
    """
    Base class for GridWorld-based MultiObs Environments 4x4  grid world.

    .. code-block:: text

        ____________
       | 0  1  2   3|
       | 4|¯5¯¯6¯| 7|
       | 8|_9_10_|11|
       |12 13  14 15|
       ¯¯¯¯¯¯¯¯¯¯¯¯¯¯

    start is 0
    states 5, 6, 9, and 10 are blocked
    goal is 15
    actions are = [left, down, right, up]

    simple linear state env of 15 states but encoded with a vector and an image observation:
    each column is represented by a random vector and each row is
    represented by a random image, both sampled once at creation time.

    :param num_col: Number of columns in the grid
    :param num_row: Number of rows in the grid
    :param random_start: If true, agent starts in random position
    :param channel_last: If true, the image will be channel last, else it will be channel first
    """

    def __init__(
        self,
        num_col: int = 4,
        num_row: int = 4,
        random_start: bool = True,
        discrete_actions: bool = True,
        channel_last: bool = True,
    ):
        super().__init__()

        self.vector_size = 4
        self.img_size = [1, 84, 84]
        # if channel_last:
        #     self.img_size = [64, 64, 1]
        # else:
        #     self.img_size = [1, 64, 64]

        self.random_start = random_start
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(0, 1, (4,))

        self.observation_space = spaces.Dict(
            spaces={
                # "vec1": spaces.Box(0, 1, (self.vector_size,), dtype=np.float64),
                # "vec2": spaces.Box(0, 1, (self.vector_size,), dtype=np.float64),
                # "vec3": spaces.Box(0, 1, (self.vector_size,), dtype=np.float64),
                # "vec4": spaces.Box(0, 1, (self.vector_size,), dtype=np.float64),
                "img1": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
                "img2": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
                "img3": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
                "img4": spaces.Box(0, 255, self.img_size, dtype=np.uint8),
            }
        )
        self.count = 0
        # Timeout
        self.max_count = 100
        self.log = ""
        self.state = 0
        self.action2str = ["left", "down", "right", "up"]
        self.init_possible_transitions()

        self.num_col = num_col
        self.state_mapping: List[Dict[str, np.ndarray]] = []
        self.init_state_mapping(num_col, num_row)

        self.max_state = len(self.state_mapping) - 1


    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, terminated, truncated, info).

        :param action:
        :return: tuple (observation, reward, terminated, truncated, info).
        """
        if not self.discrete_actions:
            action = np.argmax(action)  # type: ignore[assignment]

        self.count += 1

        prev_state = self.state

        reward = -0.1
        # define state transition
        if self.state in self.left_possible and action == 0:  # left
            self.state -= 1
        elif self.state in self.down_possible and action == 1:  # down
            self.state += self.num_col
        elif self.state in self.right_possible and action == 2:  # right
            self.state += 1
        elif self.state in self.up_possible and action == 3:  # up
            self.state -= self.num_col

        got_to_end = self.state == self.max_state
        reward = 1 if got_to_end else reward
        truncated = self.count > self.max_count
        terminated = got_to_end

        self.log = f"Went {self.action2str[action]} in state {prev_state}, got to state {self.state}"

        return self.get_state_mapping(), reward, terminated, truncated, {"got_to_end": got_to_end}


    def render(self, mode: str = "human") -> None:
        """
        Prints the log of the environment.

        :param mode:
        """
        print(self.log)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Resets the environment state and step count and returns reset observation.

        :param seed:
        :return: observation dict {'vec': ..., 'img': ...}
        """
        if seed is not None:
            super().reset(seed=seed)
        self.count = 0
        if not self.random_start:
            self.state = 0
        else:
            self.state = np.random.randint(0, self.max_state)
        return self.state_mapping[self.state], {}
