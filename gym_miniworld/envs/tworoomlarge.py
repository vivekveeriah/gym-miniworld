import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame, MeshEnt
from ..params import DEFAULT_PARAMS

class TwoRoomLarge(MiniWorldEnv):
    """
    Outside environment with two rooms connected by a gap in a wall
    """

    def __init__(self, env_kwargs=None):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', 0.5)
        # params.set('forward_step', 0.7)
        params.set('turn_step', 30)

        _config = {
            'max_episode_steps': 500, 
            'obs_width': 80, 
            'obs_height': 80, 
            'params': params,

            'randomize_start_pos': True, 
        }
        _config.update(env_kwargs or {})

        self.possible_start_pos_x_room_1 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        self.possible_start_pos_z_room_1 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        self.possible_start_pos_x_room_2 = [
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
        self.possible_start_pos_z_room_2 = [
            -7.5, -7.0, -6.5, -6.0,
            -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5]

        self.possible_dir_radians = [i * math.pi / 180 for i in range(0, 370, 30)]

        self.randomize_start_pos = _config['randomize_start_pos']
        self.agent_pos = np.array(
            [5, 0.0, -3.0]
        )

        _config.pop('randomize_start_pos', None)

        super().__init__(
            **_config,
            # max_episode_steps=300,
            # **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # Top
        room0 = self.add_rect_room(
            min_x=0, max_x=5,
            min_z=0, max_z=5,
            # wall_tex='brick_wall',
            wall_tex='door_doom',
            # floor_tex='asphalt',
            floor_tex='floor_tiles_white',
            no_ceiling=True
        )
        # Bottom
        room1 = self.add_rect_room(
            min_x=0, max_x=10,
            # min_z=-6, max_z=-1,
            min_z=-8, max_z=-1,
            wall_tex='brick_wall',
            # floor_tex='asphalt',
            floor_tex='floor_tiles_white',
            no_ceiling=True
        )
        self.connect_rooms(room0, room1, min_x=2, max_x=3.5)

        # Decorative building in the background
        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=30
            ),
            pos = np.array([30, 0, 30]),
            dir = -math.pi
        )

        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=35
            ),
            pos = np.array([-30, 0, -30]),
            dir = math.pi / 2
        )

        if self.randomize_start_pos:
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
        else:
            start_pos = self.agent_pos
            start_dir = 0

        self.place_agent(room=room0, dir=start_dir, pos=start_pos)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if action == 2:
            obs, reward, done, info = super().step(action)

        # if self.near(self.box):
        #     reward += self._reward()
        #     done = True

        return obs, reward, done, info
