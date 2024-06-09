import numpy as np


class PCHIP:
    def __init__(self, points):
        points = np.array(points)
        self.x = points[:, 0]
        sort_indices = np.argsort(self.x)
        self.x = self.x[sort_indices]
        self.y = points[sort_indices, 1]

        self.m = np.zeros_like(self.x)
        self._update_m()

    def _update_m(self, index_range=None):
        if index_range is None:
            index_range = (0, len(self.x))

        # case3
        if index_range[0] <= 0:
            self.m[0] = (self.y[0] - self.y[1]) / (self.x[0] - self.x[1])
            self._update_m((1, index_range[1]))
            return
        if index_range[1] >= len(self.x):
            self.m[-1] = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
            self._update_m((index_range[0], len(self.x) - 1))
            return

        # case1
        x = self.x[index_range[0] - 1 : index_range[1] + 1]
        y = self.y[index_range[0] - 1 : index_range[1] + 1]

        delta_x = x[1:] - x[:-1]
        delta_x_2 = delta_x[1:] + delta_x[:-1]
        delta_y = y[1:] - y[:-1]
        w_l = delta_x[1:] / delta_x_2
        w_r = delta_x[:-1] / delta_x_2
        rgrad = delta_x / delta_y  # 需要支持除以0时候为inf且1/inf=0，否则需要单独处理
        m = 1 / (w_l * rgrad[:-1] + w_r * rgrad[1:])

        # case2
        m[rgrad[:-1] * rgrad[1:] < 0] = 0
        self.m[index_range[0] : index_range[1]] = m

    def __call__(self, x):
        x = np.array(x)
        res = np.zeros_like(x)

        indices = np.searchsorted(self.x, x, side="right") - 1
        unique_indices = np.unique(indices)
        for index in unique_indices:
            res[indices == index] = self._base_func(x[indices == index], index)

        return res

    def _base_func(self, x, index):
        if index < 0:
            return self.y[0]
        if index + 1 >= len(self.x):
            return self.y[-1]

        x0, x1, y0, y1, m0, m1 = (
            self.x[index],
            self.x[index + 1],
            self.y[index],
            self.y[index + 1],
            self.m[index],
            self.m[index + 1],
        )

        x1_sub_x0 = x1 - x0
        x_sub_x0 = x - x0
        x_sub_x1 = x - x1
        l0 = x_sub_x1 / (-x1_sub_x0)
        l1 = x_sub_x0 / x1_sub_x0
        l0_square = l0**2
        l1_square = l1**2

        h0 = (1 + 2 * l1) * l0_square
        h1 = (1 + 2 * l0) * l1_square
        g0 = x_sub_x0 * l0_square
        g1 = x_sub_x1 * l1_square

        return y0 * h0 + y1 * h1 + m0 * g0 + m1 * g1

    def remove(self, index):
        index = index % len(self.x)
        self.x = np.delete(self.x, index)
        self.y = np.delete(self.y, index)
        self.m = np.delete(self.m, index)
        self._update_m((index - 1, index + 1))

    def add(self, x, y):
        index = np.searchsorted(self.x, x, side="right")
        if index >= 0 and self.x[index - 1] == x:
            self.update(index - 1, y)
            return
        self.x = np.insert(self.x, index, x)
        self.y = np.insert(self.y, index, y)
        self.m = np.insert(self.m, index, 0)
        self._update_m((index - 1, index + 2))

    def update(self, index, y):
        index = index % len(self.x)
        self.y[index] = y
        self._update_m((index - 1, index + 2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test: init and call
    points = [
        (6, 0.0),
        (0.0, 6.8),
        (2.8, 3.3),
        (5.2, 2.0),
        (0.5, 5.3),
        (2.0, 0.8),
    ]
    curve = PCHIP(points)
    # print(curve.x, curve.y, curve.m)
    print(curve(2.01))
    x = np.linspace(-1, 7, 100)
    y = curve(x)

    plt.plot(x, y)
    plt.scatter(curve.x, curve.y, c="r")
    plt.show()

    # test: remove
    curve.remove(-2)
    # print(curve.x, curve.y, curve.m)
    x = np.linspace(-1, 7, 100)
    y = curve(x)
    plt.plot(x, y)
    plt.scatter(curve.x, curve.y, c="r")
    plt.show()

    # test: add
    curve.add(5.2, 2.0)
    # print(curve.x, curve.y, curve.m)
    x = np.linspace(-1, 7, 100)
    y = curve(x)
    plt.plot(x, y)
    plt.scatter(curve.x, curve.y, c="r")
    plt.show()

    # test: update
    curve.update(-2, 3.2)
    # print(curve.x, curve.y, curve.m)
    x = np.linspace(-1, 7, 100)
    y = curve(x)
    plt.plot(x, y)
    plt.scatter(curve.x, curve.y, c="r")
    plt.show()
