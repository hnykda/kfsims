import networkx as nx
import numpy as np
import types

from kfsims.node import node_factory, observe_factory
from kfsims.common import init_all
from kfsims import noise


def make_node(measurements, cov, rho=0.9, tau=5, u=5):
    _, xk, P, _, _, _, _, H, F, Q, N = init_all()
    U = cov * (u - 3)
    nd = node_factory(xk, P, u, U, F, Q, H, rho, tau,
                      observe_factory(measurements.T), 10)
    return nd


def gen_noise(N, trj, seed, mod=20, halves=3):
    np.random.seed(seed)
    s = noise.sin_noise(N, halves, shift=seed)
    n_sin = noise.static_noise(N, mod=mod) * s
    msrms = trj.Y + n_sin.T
    return msrms


def create_nodes(n_nodes, trj, cov_init, seed_mod=0):
    nodes = []
    for i in range(n_nodes):
        msrms = gen_noise(trj.X.shape[-1], trj, i + seed_mod)
        nd = make_node(msrms, cov_init)
        nodes.append(nd)
    return nodes


def _get_neighbors_att(cluster, prior_pref):
    res = []
    for ngh in cluster:
        res.append(getattr(ngh, prior_pref + '_prior').hp)
    return res


def fuse_parameters(params_):
    params = np.array(params_)
    return np.mean(params, axis=0)


def get_cluster_params(cluster):
    Ps = _get_neighbors_att(cluster, 'P')
    Rs = _get_neighbors_att(cluster, 'R')
    new_P = fuse_parameters(Ps)
    new_R = fuse_parameters(Rs)
    return new_P, new_R


def update_nodes_neighbors_cluster(G, in_queue):
    """
    TODO: Seems, that the following fives same good results
    as when it's not commented out.
    """
    node = in_queue.pop()
    cluster = (set(G.neighbors(node)) & in_queue) | {node}
    hyp_P, hyp_R = get_cluster_params(cluster)
    for n in cluster:
        if n.R_updated != True:
            n.R_prior.hp = hyp_R
            n.P_prior.hp = hyp_P
            n.log('R_post', n.R_prior.expect())
            n.log('P_post', n.P_prior.expect())
            n.R_updated = True
    return in_queue# - cluster


def update_hyperparams(self):
    in_queue = set(np.random.permutation(self))

    for nod in in_queue:
        nod.R_updated = False

    while in_queue:
        in_queue = update_nodes_neighbors_cluster(self, in_queue)


def single_kf_for_all(self):
    for node in self.nodes:
        node.single_kf(next(node))


def vbadkf_step(self):
    self._single_kf()
    self.diffuse_hyperparams()


def collect_rmse(self, true):
    mean = []
    std = []
    for node in self.nodes:
        mn, st = node.rmse_stats(true)
        mean.append(mn)
        std.append(st)
    return np.mean(mean, axis=0).round(4), np.mean(std, axis=0).round(4)


def kf_no_diffusion(net, trj):
    for i in range(trj.shape[1]):
        net._single_kf()
    return net.collect_rmse(trj)


def kf_no_diffusion2(self, trj):
    # init
    xs = []
    Ps = []

    for node in self.nodes:
        xs.append(node.last_state)
        Ps.append(node.P_prior.expect())

    invPs = [np.linalg.inv(P_s) for P_s in Ps]
    P = np.linalg.inv(np.sum(invPs, axis=0))

    consensus_x = []
    for i in range(trj.shape[1]):
        self._single_kf()

        xs = []
        Ps = []
        for node in self.nodes:
            xs.append(node.last_state)
            Ps.append(node.P_prior.expect())

        invPs = [np.linalg.inv(P_s) for P_s in Ps]
        x = P @ np.sum([P_s @ x_s for P_s, x_s in zip(invPs, xs)], axis=0)
        P = np.linalg.inv(np.sum(invPs, axis=0))
        consensus_x.append(x)

    return np.mean((np.array(consensus_x) - trj.X.T) ** 2, axis=0)


def kf_w_diffusion(net, trj):
    for i in range(trj.shape[1]):
        net.diffused_single_kf()
    return net.collect_rmse(trj)


def create_network(nodes, k=4):
    G_ = nx.random_regular_graph(k, len(nodes))
    G = nx.relabel_nodes(G_, {ix: nodes[ix] for ix in range(len(nodes))})

    G._single_kf = types.MethodType(single_kf_for_all, G)
    G.diffuse_hyperparams = types.MethodType(update_hyperparams, G)
    G.diffused_single_kf = types.MethodType(vbadkf_step, G)

    G.collect_rmse = types.MethodType(collect_rmse, G)

    G.kf_no_diffusion = types.MethodType(kf_no_diffusion, G)
    G.kf_no_diffusion2 = types.MethodType(kf_no_diffusion2, G)
    G.kf_w_diffusion = types.MethodType(kf_w_diffusion, G)

    return G

def create_w_nodes(n_nodes, trj, cov_init, seed_mod=0):
    nodes = create_nodes(n_nodes, trj, cov_init, seed_mod=seed_mod)
    G = create_network(nodes)
    return G