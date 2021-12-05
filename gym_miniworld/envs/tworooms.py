import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS

class TwoRooms(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, **kwargs):
        super().__init__(
            max_episode_steps=250,
            **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-4, max_x=-0.5,
            min_z=1 , max_z=7
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=-0.25, max_x=10,
            min_z=1, max_z=10
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=1, max_z=3, max_y=1.74)

        self.box = self.place_entity(Box(color='red'))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class TwoRoomsS6NoTask(MiniWorldEnv):
    """
    Classic four rooms environment.
    The agent must reach the red box to get a reward.
    """

    def __init__(self, forward_step=0.5, turn_step=30, env_kwargs=None):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        _config = {
            # 'size': 6,
            'max_episode_steps': 500,
            # 'simple_env': False, 
            # 'place_box': False, 
            # 'randomize_start_pos': True, 
            # 'box_size': 0.8, 

            'domain_rand': False,
            'params': params,
            'obs_width': 80,
            'obs_height': 80, 
        }
        _config.update(env_kwargs or {})

        self.possible_start_pos_x_room_1 = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
        # self.possible_start_pos_x_room_1 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.possible_start_pos_z_room_1 = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]

        self.possible_start_pos_x_room_2 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
        # self.possible_start_pos_x_room_2 = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]
        self.possible_start_pos_z_room_2 = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]

        self.possible_dir_radians = [i * math.pi / 180 for i in range(0, 370, 30)]

        super().__init__(
            **_config,
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(
            min_x=-4, max_x=-0.5,
            # min_x=0, max_x=4,
            min_z=1 , max_z=7
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=0, max_x=10,
            # min_x=4.5, max_x=15,
            min_z=1, max_z=10
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=1.5, max_z=3, max_y=1.74)

        starting_room = np.random.choice(2)
        if starting_room == 0:
            start_x = np.random.choice(self.possible_start_pos_x_room_1)
            start_z = np.random.choice(self.possible_start_pos_z_room_1)
        elif starting_room == 1:
            start_x = np.random.choice(self.possible_start_pos_x_room_2)
            start_z = np.random.choice(self.possible_start_pos_z_room_2)

        start_dir = np.random.choice(self.possible_dir_radians)
        start_pos = np.array(
            [start_x, 0., start_z]
        )
        self.place_agent(
            dir=start_dir, pos=start_pos
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == 2:  # if move forward, then move forward twice
            obs, reward, done, info = super().step(action)

        return obs, reward, done, info
