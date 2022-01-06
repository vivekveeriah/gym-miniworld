import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class OneRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.box = self.place_entity(Box(color='red'))
        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

class OneRoomNoTask(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, simple_env=False, place_box=False, randomize_start_pos=True, box_size=0.8, **kwargs):
        assert size % 2 == 0

        self.size = size
        self.simple_env = simple_env
        self.place_box = place_box
        self.randomize_start_pos = randomize_start_pos
        self.box_size = box_size

        # Create variables that are going to be used later on
        self.possible_start_pos = [0.5 * i for i in range(1, self.size * 2)]

        if self.simple_env:
            self.possible_dir_radians = [i * math.pi / 180 for i in range(0, 370, 30)]
        else:
            self.possible_dir_radians = [i * math.pi / 180 for i in range(0, 370, 45)]
        
        self.box_pos = np.array(
            [self.size // 2, 0, self.size // 2]
        )
        if not self.place_box:
            self.agent_pos = np.array(
                [self.size // 2, 0, self.size // 2]
            )
        else:
            self.agent_pos = np.array(
                [self.size // 2 + 1, 0, self.size // 2 + 1]
            )

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)
        
    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        if self.place_box:
            self.box = self.place_entity(
                Box(color='red', size=self.box_size), pos=self.box_pos, dir=0
            )

        # self.box = self.place_entity(Box(color='red'))
        if self.randomize_start_pos:
            while True:
                random_pos_x = np.random.choice(self.possible_start_pos)
                random_pos_y = np.random.choice(self.possible_start_pos)
                random_dir = np.random.choice(self.possible_dir_radians)

                random_start_pos = np.array(
                    [random_pos_x, 0., random_pos_y]
                )

                if self.place_box: 
                    if not np.array_equal(random_start_pos, self.box_pos):
                        break
                else:
                    break
                
        else:
            random_start_pos = self.agent_pos
            random_dir = 0

        self.place_agent(
            dir=random_dir, pos=random_start_pos
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == 2:  # if move forward, then move forward twice
            obs, reward, done, info = super().step(action)

        return obs, reward, done, info

class OneRoomS6(OneRoom):
    def __init__(self, max_episode_steps=100, **kwargs):
        super().__init__(size=6, max_episode_steps=max_episode_steps, **kwargs)

class OneRoomS6NoTask(OneRoomNoTask):
    def __init__(self, turn_step=30, forward_step=0.5, env_kwargs=None):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        _config = {
            'size': 6,
            # 'max_episode_steps': 200,
            'max_episode_steps': 10_000,
            'simple_env': False, 
            'place_box': False, 
            'randomize_start_pos': True, 
            'box_size': 0.8, 

            'domain_rand': False,
            'params': params,
            'obs_width': 80,
            'obs_height': 80, 
        }
        _config.update(env_kwargs or {})

        super().__init__(
            **_config,
        )

# class OneRoomS6NoTaskHighRes(OneRoomNoTask):
#     def __init__(self, max_episode_steps=200, turn_step=30, forward_step=0.5, size=6, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=size,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=160, obs_height=160,
#             **kwargs
#         )

# class OneRoomS6NoTaskSimpleHighRes(OneRoomNoTask):
#     def __init__(self, max_episode_steps=200, turn_step=45, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=6,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=160, obs_height=160,
#             simple_env=True,
#             **kwargs
#         )

# class OneRoomS6Fast(OneRoomS6):
#     def __init__(self, forward_step=0.5, turn_step=45):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             max_episode_steps=50,
#             params=params,
#             domain_rand=False
#         )

# class OneLargeRoomNoTask(MiniWorldEnv):
#     """
#     Environment in which the goal is to go to a red box
#     placed randomly in one big room.
#     """

#     def __init__(self, size=10, max_episode_steps=180, simple_env=False, **kwargs):
#         assert size == 10
#         self.size = size
#         self._simple_env = simple_env

#         super().__init__(
#             max_episode_steps=max_episode_steps,
#             **kwargs
#         )

#         # Allow only movement actions (left/right/forward)
#         self.action_space = spaces.Discrete(self.actions.move_forward+1)

#     def _gen_world(self):
#         room = self.add_rect_room(
#             min_x=0,
#             max_x=self.size,
#             min_z=0,
#             max_z=self.size
#         )

#         # self.box = self.place_entity(Box(color='red'))
        
#         # possible_pos = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
#         # possible_pos = [0.5, 5.5]

#         possible_pos = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]

#         if not self._simple_env:
#             possible_dir_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
#             possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#         else:
#             possible_dir_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#             possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#         random_pos_x = np.random.choice(possible_pos)
#         random_pos_y = np.random.choice(possible_pos)
#         random_pos = [random_pos_x, 0., random_pos_y]

#         random_dir = np.random.choice(possible_dir_radians)

#         self.place_agent(
#             dir=random_dir, pos=np.array(random_pos)
#         )

#     def step(self, action):
#         obs, reward, done, info = super().step(action)

#         if action == 2:  # if move forward, then move forward twice
#             obs, reward, done, info = super().step(action)
            
#         # if self.near(self.box):
#         #     reward += self._reward()
#         #     done = True

#         return obs, reward, done, info

# class OneLargeRoomS6NoTask(OneLargeRoomNoTask):
#     def __init__(self, max_episode_steps=500, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=10,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=80, obs_height=80,
#             **kwargs
#         )

# class OneLargeRoomS6NoTaskHighRes(OneLargeRoomNoTask):
#     def __init__(self, max_episode_steps=500, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=10,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=160, obs_height=160,
#             **kwargs
#         )

# class OneRoomWithObstacleNoTask(MiniWorldEnv):
#     """
#     Environment in which the goal is to go to a red box
#     placed randomly in one big room.
#     """

#     def __init__(self, size=10, max_episode_steps=180, simple_env=False, **kwargs):
#         assert size == 6
#         self.size = size
#         self._simple_env = simple_env

#         super().__init__(
#             max_episode_steps=max_episode_steps,
#             **kwargs
#         )

#         # Allow only movement actions (left/right/forward)
#         self.action_space = spaces.Discrete(self.actions.move_forward+1)

#     def _gen_world(self):
#         room = self.add_rect_room(
#             min_x=0,
#             max_x=self.size,
#             min_z=0,
#             max_z=self.size
#         )

#         self.box = self.place_entity(
#             Box(color='red'), pos=np.array([3., 0., 3.]), dir=0)
        
#         random_pos = np.array([3., 0., 3.])        

#         while np.array_equal(random_pos, np.array([3., 0., 3.])):

#             possible_pos = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
#             # possible_pos = [0.5, 5.5]

#             if not self._simple_env:
#                 possible_dir_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
#                 possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#             else:
#                 possible_dir_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#                 possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#             random_pos_x = np.random.choice(possible_pos)
#             random_pos_y = np.random.choice(possible_pos)
#             random_pos = [random_pos_x, 0., random_pos_y]

#             random_dir = np.random.choice(possible_dir_radians)

#         self.place_agent(
#             dir=random_dir, pos=np.array(random_pos)
#         )

#     def step(self, action):
#         obs, reward, done, info = super().step(action)

#         if action == 2:  # if move forward, then move forward twice
#             obs, reward, done, info = super().step(action)

#         # if self.near(self.box):
#         #     reward += self._reward()
#         #     done = True

#         return obs, reward, done, info

# class OneRoomWithObstacleS6NoTask(OneRoomWithObstacleNoTask):
#     def __init__(self, max_episode_steps=200, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=6,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=80, obs_height=80,
#             **kwargs
#         )

# class OneRoomWithObstacleS6NoTaskHighRes(OneRoomWithObstacleNoTask):
#     def __init__(self, max_episode_steps=200, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=6,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=160, obs_height=160,
#             **kwargs
#         )

# class OneMidSizeRoomWithObstacleNoTask(MiniWorldEnv):
#     """
#     Environment in which the goal is to go to a red box
#     placed randomly in one big room.
#     """

#     def __init__(self, size=10, max_episode_steps=180, simple_env=False, **kwargs):
#         assert size == 8
#         self.size = size
#         self._simple_env = simple_env

#         super().__init__(
#             max_episode_steps=max_episode_steps,
#             **kwargs
#         )

#         # Allow only movement actions (left/right/forward)
#         self.action_space = spaces.Discrete(self.actions.move_forward+1)

#     def _gen_world(self):
#         room = self.add_rect_room(
#             min_x=0,
#             max_x=self.size,
#             min_z=0,
#             max_z=self.size
#         )

#         self.box = self.place_entity(
#             Box(color='red', size=0.4), pos=np.array([4., 0., 4.]), dir=0)
        
#         random_pos = np.array([4., 0., 4.])        

#         while np.array_equal(random_pos, np.array([4., 0., 4.])):

#             possible_pos = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
#             # possible_pos = [0.5, 5.5]

#             if not self._simple_env:
#                 possible_dir_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
#                 possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#             else:
#                 possible_dir_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#                 possible_dir_radians = [i * math.pi / 180 for i in possible_dir_angles]

#             random_pos_x = np.random.choice(possible_pos)
#             random_pos_y = np.random.choice(possible_pos)
#             random_pos = [random_pos_x, 0., random_pos_y]

#             random_dir = np.random.choice(possible_dir_radians)

#         self.place_agent(
#             dir=random_dir, pos=np.array(random_pos)
#         )

#     def step(self, action):
#         obs, reward, done, info = super().step(action)

#         if action == 2:  # if move forward, then move forward twice
#             obs, reward, done, info = super().step(action)

#         # if self.near(self.box):
#         #     reward += self._reward()
#         #     done = True

#         return obs, reward, done, info

# class OneMidSizeRoomWithObstacleS6NoTask(OneMidSizeRoomWithObstacleNoTask):
#     def __init__(self, max_episode_steps=400, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=8,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=80, obs_height=80,
#             **kwargs
#         )

# class OneMidSizeRoomWithObstacleS6NoTaskHighRes(OneMidSizeRoomWithObstacleNoTask):
#     def __init__(self, max_episode_steps=400, turn_step=30, forward_step=0.5, **kwargs):
#         # Parameters for larger movement steps, fast stepping
#         params = DEFAULT_PARAMS.no_random()
#         params.set('forward_step', forward_step)
#         params.set('turn_step', turn_step)

#         super().__init__(
#             size=8,
#             max_episode_steps=max_episode_steps,
#             domain_rand=False,
#             params=params,
#             obs_width=160, obs_height=160,
#             **kwargs
#         )
