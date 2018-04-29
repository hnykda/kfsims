from kfsims.trajectory import trajectory
import matplotlib.pylab as plt
import numpy as np


def init(P_k__l, tau, n, rho, u_l__l, m, U_l__l):
    """
    Step 3.
    """
    t_k__l = n + tau + 1
    T_k__l = tau * P_k__l
    u_k__l = rho * (u_l__l - m - 1) + m + 1
    U_k__l = rho * U_l__l
    return t_k__l, T_k__l, u_k__l, U_k__l


def predict_state(F_l, x_l__l):
    """
    Step 1
    """
    return F_l @ x_l__l


def predict_PECM(F_l, P_l__l, Q_l):
    """
    Step 2
    """
    return F_l @ P_l__l @ F_l.T + Q_l


def init_trajectory(ndat=250):
    traj = trajectory(ndat=ndat)
    traj.R = 5 * np.eye(2)
    traj._simulate()
    return traj


def init_all(traj=None, N=10, add_cov=None):
    if not traj:
        traj = init_trajectory()
    xk = np.array([0, 0, 1, 1])
    P = 100 * np.eye(4)
    tau = 5
    rho = .99

    u = 13
    cv = add_cov if add_cov else 0
    U = (traj.R + cv) * (u - traj.R.shape[0] - 1)

    H = traj.H
    F = traj.A
    Q = 2 * traj.Q
    return traj, xk, P, tau, rho, u, U, H, F, Q, N


def kalman_correction(H, Pik, Rik, xk, zk):
    bracket = H @ Pik @ H.T + Rik
    Kik = Pik @ H.T @ np.linalg.inv(bracket)
    xikk = xk + Kik @ (zk - H @ xk)
    Pik = Pik - Kik @ H @ Pik
    return Pik, xikk


def plot_results(traj, x_log):
    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.plot(traj.X[0], traj.X[1], label='True states')
    plt.plot(traj.Y[0], traj.Y[1], 'r+', label='Measurements')
    plt.plot(x_log[0], x_log[1], 'g+', label='Estimates from KF')
    plt.legend()

    for m_point, est_point in zip(x_log.T, traj.Y.T):
        plt.plot([m_point[0], est_point[0]], [m_point[1], est_point[1]], 'y', alpha=0.5)

    for label, x, y in zip(list(range(traj.X.shape[1])), traj.Y[0], traj.Y[1]):
        plt.annotate(
            str(label),
            xy=(x, y), xytext=(10, -15),
            textcoords='offset points', ha='right', va='bottom', alpha=0.4)
    #        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0', alpha=0.1))

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(traj.X[2], label='velocity x')
    plt.plot(x_log[2], label='KF estimate')
    plt.legend()
    plt.subplot(122)
    plt.plot(traj.X[3], label='velocity y')
    plt.plot(x_log[3], label='KF estimate')
    plt.legend()


def time_update(x, P, F, Q):
    x_new = predict_state(F, x)
    P_new = predict_PECM(F, P, Q)
    return x_new, P_new
