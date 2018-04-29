import numpy as np

from filterpy.kalman import KalmanFilter
import matplotlib.pylab as plt
import seaborn as sns

from kfsims.common import init_all
from kfsims.node import observe_factory, node_factory


def daniels_variant(measurements, true, cov, rho=0.95, tau=5, u=5, debug=False):
    iterations = 10
    _, xk, P, _, _, _, _, H, F, Q, N = init_all()
    U = cov * (u - 3)
    nd = node_factory(xk, P, u, U, F, Q, H, rho, tau,
                      observe_factory(measurements.T), iterations)
    nd()
    preds = np.array(nd.logger['x']).squeeze()
    m, s = nd.rmse_stats(true.T)
    if debug:
        return preds, m, s, nd
    return preds, m, s


def classic_kf(traj, measurements, true, cov):
    my_filter = KalmanFilter(dim_x=4, dim_z=2)
    my_filter.x = np.array([[0], [0], [1], [1]])
    my_filter.F = traj.A
    my_filter.H = traj.H
    my_filter.P = 100 * np.eye(4)

    my_filter.R = cov
    my_filter.Q = traj.Q

    rec = []
    for zk in observe_factory(measurements.T)():
        my_filter.predict()
        my_filter.update(zk)
        x = my_filter.x
        rec.append(x)

    preds = np.array(rec)[:, :, 0]
    s = np.sqrt((preds[:] - true[:]) ** 2)
    return preds, np.mean(s, axis=0).round(3), np.std(s, axis=0).round(3)


def plot_variants_only(av, kfc, true, start_pos):
    f, axs = plt.subplots(2, 2, figsize=(15, 10))
    for sl, ax in enumerate(axs.reshape(-1)):
        ax.plot(av[start_pos:, sl], label='VBKF', alpha=0.8)
        ax.plot(kfc[start_pos:, sl], label='CKF', alpha=0.8)
        ax.plot(true[start_pos:, sl], label='True', alpha=0.8)
        ax.legend()
        ax.set_title('{}. coord'.format(sl))


def plot_single(ax1, sl, av, kfc, true, measurements=None, lw=2, start_pos=0):
    ax1.plot(av[start_pos:, sl], label='VBAKF', lw=lw, alpha=0.8)
    ax1.plot(kfc[start_pos:, sl], label='CKF', lw=lw, alpha=0.8)
    ax1.plot(true[start_pos:, sl], linestyle='--', label='True', lw=lw, alpha=0.8)
    if measurements is not None:
        ax1.plot(measurements.T[start_pos:, sl], label='Measurements',
                 linestyle='-', lw=1, alpha=0.4)
    ax1.legend()
    #ax1.xticks(list(range(start_pos, measurements.shape[0])))


def plot_variants(av, kfc, measurements, true, start_pos):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_single(ax1, 0, av, kfc, true, measurements, start_pos=start_pos)
    plot_single(ax2, 1, av, kfc, true, measurements, start_pos=start_pos)


def hists_subplots(df, supsuf, legpos='best'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
    sns.barplot(x='Component', y='RMSE', data=df, hue='Algorithm', ax=ax1)
    ax1.set_title(f'RMSE', fontsize=15, y=0.99)
    ax1.yaxis.grid(True)
    ax1.set_ylabel('')
    ax1.legend(loc=legpos)

    sns.barplot(x='Component', y='STD', data=df, hue='Algorithm', ax=ax2)
    ax2.set_title(f'Standard Deviation', fontsize=15, y=0.99)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('')
    ax2.yaxis.grid(True)
    ax2.legend(loc=legpos)
    # savef('bars-known-static-noise.pdf')

    plt.suptitle('Comparison of RMSE and STD for ' + supsuf, y=1)
    # plt.subplots_adjust(wspace=0.5, top=3)

    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # f.subplots_adjust(top=0.85)

    return ax1, ax2