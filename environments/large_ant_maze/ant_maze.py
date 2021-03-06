# Heavily based on the environment created by Giuseppe Paolo (paolo@isir.upmc.fr)
import numpy as np
import os
import pdb

from gym import utils
from gym.spaces import Dict, Box
from gym.utils import EzPickle
from gym.envs.mujoco import mujoco_env
from gym.utils import seeding

from collections import deque
from functools import reduce

class GoalArea:
    def __init__(self, coords, color, ray):
        """
        coords np array
        color  str
        ray    float
        """
        self.coords = coords
        self.color = color
        self.ray = ray

    def dist(self, x):
        return np.linalg.norm(x - self.coords)

    def solved_by(self, x):
        return self.dist(x) < self.ray


class AntObstaclesBigEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_path, max_ts, fixed_init=True, goal_type="huge"):
        self.ts = 0
        if goal_type == "large":
            self.goals = [
                GoalArea(np.array([34, -25]), "green", 5),
                GoalArea(np.array([-24, 33]), "red", 5),
                GoalArea(np.array([15, 15]), "blue", 5)
            ]
        elif goal_type == "huge":
            self.goals = [
                GoalArea(
                    np.array([34, -25]), "green",
                    5),  #color strings should be compatible with matplotlib
                GoalArea(np.array([-24, 33]), "green", 5),
                GoalArea(np.array([15, 15]), "green", 5),
                GoalArea(np.array([-21, -40]), "green", 5)
            ]
        else:
            raise Exception("wronge goal type")

        self.max_ts = max_ts
        self.xml_path = xml_path
        self.ts = 0
        self._obs_hist = deque(maxlen=60) if goal_type == "huge" else deque(
            maxlen=30)  #to check if the ant is stuck

        self.fixed_init = fixed_init

        mujoco_env.MujocoEnv.__init__(
            self, self.xml_path, frame_skip=5
        )  #not that the max number of steps displayed in the viewer will be frame_skip*self.max_ts, NOT self.max_ts
        utils.EzPickle.__init__(self, xml_path, max_ts, fixed_init, goal_type)

        #note: don't add members after the call to MujocoEnv.__init__ as it seems to call step

        #also note: change init pose of ant from the xml, not from code
        #init_qpos=np.array([-4. , -24. ,   0.6,   1. ,   0. ,   0. ,   0. ,   0. ,   0. ,
        #    0. ,   0. ,   0. ,   0. ,   0. ,   0. ])
        #self.set_state(init_qpos, self.init_qvel)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]

    def step(self, a):
        """
        Note that you shouldn't return goal successes here as step is going to be called in parallel by many processes/threads via a problem's __call__ unless you want to handle the 
        concurrent access issues. It's easier to just evaluate for sucess inside the aforementioned __call__, this way you don't need the hassle of mutexes etc
        """
        self.do_simulation(a, self.frame_skip)
        planar_position = self.data.qpos[:2]

        end_episode = False
        if self.ts > self.max_ts:
            end_episode = True
        self.ts += 1

        reward = 0
        #we should NOT check for that type of reward here because if the agent reaches this destination but gets stuck it will continue to receive positive rewards
        #this sould be handled from __call__
        #for gl in self.goals:
        #    if gl.solved_by(np.array([planar_position[0], planar_position[1]])):
        #        reward+=1

        ob = self._get_obs()

        self._obs_hist.append(ob)
        is_stuck = False
        if len(self._obs_hist) == self._obs_hist.maxlen and np.linalg.norm(
                reduce(lambda x, y: y - x, self._obs_hist, 0)) < 1:
            is_stuck = True
            reward = -5
            end_episode = True

        return ob, reward, end_episode, dict(x_position=planar_position[0],
                                             y_position=planar_position[1],
                                             is_stuck=is_stuck)

    def _get_obs(self):
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        if self.fixed_init:
            qpos = self.init_qpos
            qvel = self.init_qvel
            self.set_state(qpos, qvel)
            self.ts = 0
        else:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
            self.set_state(qpos, qvel)
            self.ts = 0

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[0] = 4.2
        self.viewer.cam.lookat[1] = 0
        self.viewer.opengl_context.set_buffer_size(4024, 4024)

