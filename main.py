import numpy


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
        lr = 0.1
        for _ in range(100):
            Z = -G + (gd * B.T).T
            Z += lambda_
            B -= Z * lr
            B -= B * lr * lambda_
            B[diags] = 0.0
            B2 = B * 0.9
            if self.loss(X, B2, lambda_) < self.loss(X, B, lambda_):
                B = B2
        return B.astype(numpy.float32)


X = numpy.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
    ],
    dtype="f",
)

X_trivial = numpy.array(
    [
        [1, 1, 0],
    ],
    dtype="f",
)


def test(name, modelClass, X, lambda_):
    print(f"# {name} -- lambda={lambda_}")
    print("X=")
    print(X)
    model = modelClass()
    loss = model.fit(X, lambda_)
    print(f"{loss=}")
    print("model.B=")
    print(model.B)
    print("model.predict=")
    print(model.predict())
    print()


test("EaseExact with X_trivial", EaseExact, X_trivial, 0.0001)
test("EaseApprox with X_trivial", EaseApprox, X_trivial, 0.0001)

test("EaseExact with X", EaseExact, X, 0.1)
test("EaseApprox with X", EaseApprox, X, 0.1)
