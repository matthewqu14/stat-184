import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr


class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, s, a):
        """
        Cost function of the env.
        It sets the state of environment to `s` and then execute the action `a`, and
        then return the cost. 
        Parameter:
            s (1D numpy array) with shape (4,) 
            a (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        observation, cost, done, info = env.step(a)
        return cost

    def f(self, s, a):
        """
        State transition function of the environment.
        Return the next state by executing action `a` at the state `s`
        Parameter:
            s (1D numpy array) with shape (4,)
            a (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert s.shape == (4,)
        assert a.shape == (1,)
        env = self.env
        env.reset(state=s)
        next_observation, cost, done, info = env.step(a)
        return next_observation

    def compute_approximation(self, Q, M, R, lam=1e-7):
        H = np.block([[Q, M], [M.T, R]])
        v, w = np.linalg.eig(H)
        H_approx = np.zeros(H.shape)
        for i, vec in enumerate(v):
            if vec > 0:
                H_approx += vec * w[:, i:i + 1] @ w[:, i:i + 1].T
        H_approx += lam * np.identity(H.shape[0])
        q1, q2 = Q.shape
        m1, m2 = M.shape
        r1, r2 = R.shape
        Q_new = H_approx[:q1, :q2]
        M_new = H_approx[:m1, q2:q2 + m2]
        R_new = H_approx[q1:q1 + r1, q2:q2 + r2]
        return Q_new, M_new, R_new

    def compute_local_policy(self, s_star, a_star, T):
        """
        This function perform a first order taylor expansion function f and
        second order taylor expansion of cost function around (s_star, a_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            s_star (numpy array) with shape (4,)
            a_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimal policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        A = jacobian(lambda s1: self.f(s1, a_star), s_star)
        B = jacobian(lambda a1: self.f(s_star, a1), a_star)

        q = gradient(lambda s1: self.c(s1, a_star), s_star)
        r = gradient(lambda a1: self.c(s_star, a1), a_star)

        Q = hessian(lambda s1: self.c(s1, a_star), s_star)
        R = hessian(lambda a1: self.c(s_star, a1), a_star)
        M = hessian(lambda sa: self.c(sa[:-1], sa[-1:]), np.concatenate((s_star, a_star)))[:-1, -1:]

        Q_2 = Q / 2
        R_2 = R / 2
        q_2 = (q.T - s_star.T @ Q - a_star.T @ M.T).T
        q_2 = q_2[:, None]
        r_2 = (r.T - a_star.T @ R - s_star.T @ M).T
        r_2 = r_2[:, None]
        b = self.c(s_star, a_star) + 0.5 * s_star.T @ Q_2 @ s_star + 0.5 * a_star.T @ R @ a_star + \
            s_star.T @ M @ a_star - q.T @ s_star - r.T @ a_star
        b = b.flatten()
        m = self.f(s_star, a_star) - A @ s_star - B @ a_star
        m = m[:, None]

        Q1, M1, R1 = self.compute_approximation(Q_2, M, R_2)

        return lqr(A, B, m, Q1, R1, M1, q_2, r_2, b, T)


class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a
