import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class Hallway(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    at the end of a hallway
    """

    def __init__(self, length=12, **kwargs):

        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 0.5)
        params.set('turn_step', 30)

        assert length >= 2
        self.length = length

        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=-1, max_x=-1 + self.length,
            min_z=-2, max_z=2
        )

        # Place the box at the end of the hallway
        self.box = self.place_entity(
            Box(color='red'),
            min_x=room.max_x - 2
        )

        # Place the agent a random distance away from the goal
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            max_x=room.max_x - 2
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
