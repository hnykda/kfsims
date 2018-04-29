import numpy as np

Matrix = np.ndarray
MultiShape = np.ndarray
Scalar = np.number
Vector = np.ndarray


class IWPrior:
    hp: MultiShape = None

    def __init__(self, nu: Scalar, psi: Matrix):
        self.hp = np.array([
            -0.5 * psi,
            -0.5 * (nu + psi.shape[0] + 1)
        ])

    @property
    def psi(self) -> Matrix:
        return -2 * self.hp[0]

    @property
    def nu(self) -> Scalar:
        return -2 * self.hp[1] - self.p - 1

    @property
    def p(self) -> int:
        return self.hp[0].shape[0]

    def expect(self):
        return self.psi / (self.nu - self.p - 1)
