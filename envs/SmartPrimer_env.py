import gym
from gym import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class SmartPrimerEnv(gym.Env):

    def __init__(self):
        self.env = {}
        self.info = {}

        self.RewardsPerChild = []
        self.performance = []

        self.nFinish = []
        self.avgFinish = []

        self.Nquit = []
        self.avgQuit = []

        self.childrenSimulated = 0

        # pre-test, grade, age, seconds of last interaction, seconds of last correct answer,
        # [0,0,0] (positive, idk, negative) words since last action taken, stage of the problem,
        # seconds since last interaction with wizard, anxiety
        # low = np.array((0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0), dtype=float)
        # high = np.array((10, 6, 10, 1000, 1000, 1, 1, 1, 3, 1000, 45),
        #                 dtype=float)  # pre-test, 4 words dim, 3 prev-hints

        # grade, pre-score, stage, failed attempts, pos, neg, hel, anxiety
        # low = np.array((2, 0, 0, 0, 0, 0, 0, 9), dtype=float)
        # high = np.array((4, 8, 6, 20, 1, 1, 1, 45), dtype=float)

        low = np.array((-1, -1, -1, -1, -1, -1, -1, -1), dtype=float)
        high = np.array((1, 1, 1, 1, 1, 1, 1, 1), dtype=float)

        self.observation_space = spaces.Box(low, high, dtype=np.float)

        self.action_space = spaces.Discrete(4)  # do nothing, encourage, ask question or provide hints
        self.actions = ['hint', 'nothing', 'encourage', 'question']
        self.reward_range = (-8, 9)

        self.actionInfo = {'nothing': [], 'encourage': [], 'question': [], 'hint': []}
        self.avgActionInfo = {'nothing': [], 'encourage': [], 'question': [], 'hint': []}

        kids = range(0, 1)
        for kid in kids:
            self.actionInfo[str(kid)] = [[], [], [], []]
            self.avgActionInfo[str(kid)] = [[], [], [], []]

        self.reset()


    def step(self, action):
        pass


    def reset(self):
        '''Starts a new episode by creating a new child and resetting performance, stage, observation space.'''
        self.state = np.array((-1, -1, -1, -1, -1, -1, -1, -1), dtype=float)
        return self.state
