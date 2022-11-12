from env_MAB import *
from functools import lru_cache
import numpy as np


def random_argmax(a):
    '''
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    '''
    return np.random.choice(np.where(a == a.max())[0])


def calc_reward(record):
    return record[:, 1] / np.sum(record, axis=1)


class Explore():
    def __init__(self, MAB):
        self.MAB = MAB
        self.pulls = np.zeros(MAB.get_K())

    def reset(self):
        self.MAB.reset()
        self.pulls = np.zeros(self.MAB.get_K())

    def play_one_step(self):
        ind = random_argmax(-self.pulls)
        self.MAB.pull(ind)
        self.pulls[ind] += 1


class Greedy():
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        pulls = int(np.sum(self.MAB.get_record()))
        if pulls < self.MAB.get_K():
            self.MAB.pull(pulls)
        else:
            ind = random_argmax(calc_reward(self.MAB.get_record()))
            self.MAB.pull(ind)


class ETC():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta
        self.Ne = self.calc_Ne()
        self.pulls = np.zeros(MAB.get_K())
        self.best_arm = None

    def calc_Ne(self):
        numer = self.MAB.get_T() * np.sqrt(np.log(2 * self.MAB.get_K() / self.delta) / 2)
        return int(np.power(numer / self.MAB.get_K(), 2 / 3))

    def reset(self):
        self.MAB.reset()
        self.pulls = np.zeros(self.MAB.get_K())
        self.best_arm = None

    def play_one_step(self):
        if np.sum(self.pulls) < self.Ne * self.MAB.get_K():
            ind = random_argmax(self.pulls < self.Ne)
            self.MAB.pull(ind)
            self.pulls[ind] += 1
        else:
            # print(self.best_arm)
            # print(self.MAB.get_regrets())
            if self.best_arm is None:
                self.best_arm = random_argmax(calc_reward(self.MAB.get_record()))
            self.MAB.pull(self.best_arm)


class Epgreedy():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def calc_epsilon(self):
        t = int(np.sum(self.MAB.get_record()))
        if t < self.MAB.get_K():
            return 0
        return np.power(self.MAB.get_K() * np.log(t) / t, 1 / 3)

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        pulls = int(np.sum(self.MAB.get_record()))
        if pulls < self.MAB.get_K():
            self.MAB.pull(pulls)
        else:
            if np.random.random() < self.calc_epsilon():
                ind = np.random.randint(self.MAB.get_K())
                self.MAB.pull(ind)
            else:
                ind = random_argmax(calc_reward(self.MAB.get_record()))
                self.MAB.pull(ind)


class UCB():
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def calc_upper_bounds(self):
        K, T = self.MAB.get_K(), self.MAB.get_T()
        record = self.MAB.get_record()
        mu = calc_reward(record)
        bound = np.sqrt(np.log(K * T / self.delta) / np.sum(record, axis=1))
        return mu + bound

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()

    def play_one_step(self):
        pulls = int(np.sum(self.MAB.get_record()))
        if pulls < self.MAB.get_K():
            self.MAB.pull(pulls)
        else:
            ind = random_argmax(self.calc_upper_bounds())
            self.MAB.pull(ind)


class Thompson_sampling():
    def __init__(self, MAB):
        self.MAB = MAB
        self.params = np.ones((MAB.get_K(), 2))

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.params = np.ones((self.MAB.get_K(), 2))

    def play_one_step(self):
        '''
        Implement one step of the Thompson sampling algorithm. 
        '''
        theta = np.random.beta(self.params[:, 0], self.params[:, 1])
        ind = random_argmax(theta)
        self.MAB.pull(ind)
        self.params[ind] += self.MAB.get_record()[ind, ::-1]


class Gittins_index():
    def __init__(self, MAB, gamma=0.90, epsilon=1e-4, N=100):
        self.MAB = MAB
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = N
        self.lower_bound = 0
        self.upper_bound = 1 / (1 - self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())
        self.start_flag = False

    def reset(self):
        '''
        Reset the instance and eliminate history.
        '''
        self.MAB.reset()
        self.lower_bound = 0
        self.upper_bound = 1 / (1 - self.gamma)
        self.gittins_indices = np.zeros(self.MAB.get_K())
        self.start_flag = False

    @lru_cache(maxsize=None)
    def calculate_value_oab(self, successes, total_num_samples, lambda_hat, stage_num=0):
        '''
        Helper function for calculating the OAB value. Recursive function
        '''
        prop = successes / total_num_samples
        val = prop - lambda_hat / (1 - self.gamma)
        if stage_num == self.N:
            return (self.gamma ** self.N) / (1 - self.gamma) * max(val, 0)
        else:
            recursive_val1 = self.calculate_value_oab(successes + 1, total_num_samples + 1, lambda_hat, stage_num + 1)
            recursive_val2 = self.calculate_value_oab(successes, total_num_samples + 1, lambda_hat, stage_num + 1)
            return max(val + self.gamma * (prop * recursive_val1 + (1 - prop) * recursive_val2), 0)

    def compute_gittins_index(self, arm_index):
        '''
        Calibration for Gittins Index (Algorithm 1)
        '''
        self.lower_bound = 0
        self.upper_bound = 1 / (1 - self.gamma)
        failure, success = self.MAB.get_record()[arm_index]
        while self.upper_bound - self.lower_bound > self.epsilon:
            lambda_hat = (self.upper_bound + self.lower_bound) / 2

            V = self.calculate_value_oab(success + 1, success + failure + 2, lambda_hat)
            if V > 0:
                self.lower_bound = lambda_hat
            else:
                self.upper_bound = lambda_hat

        return self.upper_bound

    def play_one_step(self):
        '''
        Select the arm with the highest Gittins Index and about its Gittins Index based on the value return by pull 
        '''
        if not self.start_flag:
            self.gittins_indices = np.array([self.compute_gittins_index(0)] * self.MAB.get_K())
            self.start_flag = True
        # pulls = int(np.sum(self.MAB.get_record()))
        # if pulls < self.MAB.get_K():
        #     self.MAB.pull(pulls)
        #     self.gittins_indices[pulls] = self.compute_gittins_index(pulls)

        ind = random_argmax(self.gittins_indices)
        self.MAB.pull(ind)
        self.gittins_indices[ind] = self.compute_gittins_index(ind)
