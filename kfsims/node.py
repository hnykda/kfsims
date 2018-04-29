from kfsims.iw_prior import IWPrior
import numpy as np
from kfsims import common
from kfsims.exp_family import update_IW
from archive import exp_family
from typing import Tuple, Callable
from collections import defaultdict
from filterpy.kalman import KalmanFilter

Matrix = np.ndarray
MultiShape = np.ndarray
Scalar = np.number
Vector = np.ndarray


class MeasurementNode:
    P_prior: IWPrior = None
    R_prior: IWPrior = None
    F: Matrix = None
    Q: Matrix = None
    last_state: Vector = None

    logger = None

    def __init__(self,
                 x: Vector,
                 P_prior: IWPrior,
                 R_prior: IWPrior,
                 F: Matrix,
                 Q: Matrix,
                 H: Matrix,
                 rho: Scalar,
                 tau: Scalar,
                 observe: Callable,
                 N=10,
                 label: str = None,
                 ):
        """
        Args:
            observe: a function without any arguments which
              performs a measurement by the node. In a dummy
              case, this is just an iterator of data faking
              measurements of the node.
        """
        self.last_state = x
        self.P_prior = P_prior
        self.R_prior = R_prior
        self.F = F
        self.Q = Q
        self.H = H
        self.rho = rho
        self.tau = tau
        self.observe = observe
        self.N = N

        self.logger = defaultdict(list)

        self.P = self.P_prior.expect()

        self.observe_gen = observe()

        self.label = label if label else str(id(self))[-5:]

    def predict_state(self) -> Tuple[Vector, Matrix]:
        return common.time_update(self.last_state, self.P, self.F, self.Q)

    def _single_update(self, state_prediction, P_prediction, measurement,
                       init_hyp_P, init_hyp_R):
        x = state_prediction
        P = P_prediction
        for i in range(self.N):
            R, hyp_R = update_IW(init_hyp_R,
                                 measurement,
                                 self.H @ x,
                                 self.H @ P @ self.H.T)
            P, hyp_P = update_IW(init_hyp_P, x, state_prediction, P)
            P, x = common.kalman_correction(self.H, P, R, state_prediction, measurement)
        return x, P, hyp_P, hyp_R

    def _init_hyp_P(self, P_predicted):
        init = exp_family.init_P_hyp(self.tau, P_predicted)
        return init

    def _init_hyp_R(self):
        init = exp_family.init_R_hyp(self.rho, self.R_prior.psi, self.R_prior.nu)
        return init

    def _single_kf(self, measurement):
        x_predicted, P_predicted = self.predict_state()
        init_hyp_P = self._init_hyp_P(P_predicted)
        init_hyp_R = self._init_hyp_R()
        return self._single_update(x_predicted, P_predicted, measurement,
                                   init_hyp_P, init_hyp_R)

    def single_kf(self, measurement):
        x, P, hyp_P, hyp_R = self._single_kf(measurement)
        self.last_state = x
        self.P = P
        self.P_prior.hp = hyp_P
        self.R_prior.hp = hyp_R
        self.log('x', x)
        self.log('P', P)
        self.log('R', self.R_prior.expect())
        self.log('y', measurement)
        return x, P, hyp_P, hyp_R

    @property
    def _kf_iterator(self):
        for i, measurement in enumerate(self.observe()):
            x, P, hyp_P, hyp_R = self.single_kf(measurement)
            yield x, P, hyp_P, hyp_R

    def __call__(self):
        _ = list(self._kf_iterator)

    def __str__(self):
        return f'<Node: {self.label}>'

    def __repr__(self):
        return self.__str__()

    def log(self, key, val):
        self.logger[key].append(val)

    def _rmse_diff(self, true, start_element):
        x_log = np.array(self.logger['x']).squeeze().T
        s = np.sqrt((x_log[:, start_element:] - true[:, start_element:]) ** 2)
        return s

    def post_rmse(self, true, start_element=0):
        s = self._rmse_diff(true, start_element)
        return np.mean(s, axis=1).round(3)

    def post_rmse_std(self, true, start_element=0):
        s = self._rmse_diff(true, start_element)
        return np.std(s, axis=1).round(3)

    def rmse_stats(self, true):
        m = self.post_rmse(true)
        s = self.post_rmse_std(true)
        return m, s

    def norm_r(self):
        return np.array([np.linalg.norm(x) for x in self.logger['R']])

    def norm_p(self):
        return np.array([np.linalg.norm(x) for x in self.logger['P']])

    def __next__(self):
        return next(self.observe_gen)


def observe_factory(traj):
    def f():
        return (y for y in traj)

    return f


def node_factory(x, P, u, U, F, Q, H, rho, tau, observe_func, iterations=10):
    P_p = IWPrior(P.shape[0] + tau + 1, tau * P)
    R_p = IWPrior(u, U)
    return MeasurementNode(x, P_p, R_p, F, Q, H, rho, tau, observe_func, N=iterations)


def make_simple_nodes(n=5, iterations=10, traj=None, noise_modifier=None):
    if not noise_modifier:
        noise_modifier = lambda _: 10
    nodes = []
    for i in range(n):
        traj2, xk, P, tau, rho, u, U, H, F, Q, N = common.init_all(traj)
        np.random.seed(i)
        traj2.Y = traj2.Y + (np.random.normal(size=traj2.Y.shape) * noise_modifier(traj2.Y.shape))
        nd = node_factory(xk, P, u, U, F, Q, H, rho, tau, observe_factory(traj2.Y.T.copy()), iterations)
        nodes.append(nd)
    return nodes
