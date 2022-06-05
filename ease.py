import numpy
import tqdm


class EaseModel:
    """Ease Recommender.

    X is (User, Item) matrix,
    B is (Item, Item) matrix.
    """

    X: numpy.ndarray
    B: numpy.ndarray
    lambda_: float

    def loss(self, X: numpy.ndarray, B: numpy.ndarray, lambda_: float) -> float:
        return ((X - X.dot(B)) ** 2).sum() + lambda_ * (B**2).sum()

    def predict(self) -> numpy.ndarray:
        return self.X.dot(self.B)

    def fit(self, X: numpy.ndarray, lambda_: float) -> float:
        """Returns loss"""
        B = self._fit(X, lambda_)
        self.X = X
        self.B = B
        self.lambda_ = lambda_
        return self.loss(X, B, lambda_)

    def _fit(self, _: numpy.ndarray, __: float) -> numpy.ndarray:
        """Returns B"""
        raise NotImplementedError


class EaseExact(EaseModel):
    def _fit(self, X: numpy.ndarray, lambda_: float) -> numpy.ndarray:
        """Exact Fitting

        O(U*I*I) for computing G,
        O(I*I*I) for inv(G).
        """
        G = X.T.dot(X)
        diags = numpy.diag_indices(G.shape[0])
        G[diags] += lambda_
        P = numpy.linalg.inv(G)
        B = P / (-numpy.diag(P))
        B[diags] = 0.0
        return B


class EaseApprox(EaseModel):
    def _fit(self, X: numpy.ndarray, lambda_: float) -> numpy.ndarray:
        """Approximately

        O(U*I*I) for computing G,
        O(I*I) for an update step.
        """
        _, i = X.shape
        diags = numpy.diag_indices(i)
        G = X.T.dot(X)
        gd = numpy.diag(G)
        B = (G > 0) * 1.0
        B[diags] = 0.0
        lr = 1.0
        for _ in tqdm.tqdm(range(1000)):
            Z = -G + (gd * B.T).T
            Z += lambda_
            Z = Z.clip(min=-2.0, max=2.0)
            B -= Z * lr
            B[diags] = 0.0
            lr = max(0.001, lr * 0.99)
        while self.loss(X, B * 0.9, lambda_) < self.loss(X, B, lambda_):
            B *= 0.9
        return B.astype(numpy.float32)


def test(name, modelClass, X, lambda_, small=True):
    print(f"# {name} -- lambda={lambda_}")
    print("X=")
    if small:
        print(X)
    else:
        print(X[:5, :5])
    model = modelClass()
    loss = model.fit(X, lambda_)
    print(f"{loss=}")
    print("model.B=")
    print(model.B if small else model.B[:5, :5])
    print("model.predict=")
    print(model.predict() if small else model.predict()[:5, :5])
    print()


X_trivial = numpy.array(
    [
        [1, 1, 0],
    ],
    dtype="f",
)
test("EaseExact with X_trivial", EaseExact, X_trivial, 0.0001)
test("EaseApprox with X_trivial", EaseApprox, X_trivial, 0.0001)

X = numpy.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ],
    dtype="f",
)
test("EaseExact with X", EaseExact, X, 0.1)
test("EaseApprox with X", EaseApprox, X, 0.1)

u = 5
i = 5
X_random = (numpy.random.randn(u, i) > 0.1) * 1.0
test("EaseExact with X_random", EaseExact, X_random, 0.1)
test("EaseApprox with X_random", EaseApprox, X_random, 0.1)

u = 2000
i = 1000
X_large = (numpy.random.randn(u, i) > 1.0) * 1.0
test("EaseExact with X_large", EaseExact, X_large, 0.1, small=False)
test(
    "EaseApprox with X_large", EaseApprox, X_large, 0.1, small=False
)  # not good performanct
