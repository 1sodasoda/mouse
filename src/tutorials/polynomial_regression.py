"""PyTorch polynomial regression, run locally so Week 4 can record real numbers.

Follows the polynomial example from
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html: fit
``y = a + b*x + c*x^2 + d*x^3`` to ``sin(x)`` on ``[-pi, pi]`` by gradient
descent.

Three extra measurements are taken on top of the tutorial, because the point
of Part 4 is to compare this against predicting mouse movement:

1. **Held-out error** on fresh x drawn from the same range -- how much the fit
   degrades on data it did not train on.
2. **Extrapolation error** outside the training range -- the polynomial has
   never seen those x values.
3. **A noisy version**, where the target is ``sin(x) + noise`` -- the closest
   this tutorial gets to a target that is not perfectly determined by the
   input, which is the situation mouse data is always in.

Run from the src/ directory:
    python tutorials/polynomial_regression.py
"""
import argparse
import math

import torch


def fit(x, y, iters, lr, seed=0):
    """Learn a, b, c, d for y ~= a + bx + cx^2 + dx^3. Returns (coeffs, losses)."""
    torch.manual_seed(seed)
    a, b, c, d = (torch.randn((), requires_grad=True) for _ in range(4))
    losses = []
    for t in range(iters):
        pred = a + b * x + c * x ** 2 + d * x ** 3
        loss = (pred - y).pow(2).sum()
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            for p in (a, b, c, d):
                p -= lr * p.grad
                p.grad = None
    return (a, b, c, d), losses


def predict(coeffs, x):
    a, b, c, d = coeffs
    with torch.no_grad():
        return a + b * x + c * x ** 2 + d * x ** 3


def rmse(pred, true):
    return float(torch.sqrt((pred - true).pow(2).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--points", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.1,
                   help="std of the noise added in the noisy run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    x = torch.linspace(-math.pi, math.pi, args.points)
    y = torch.sin(x)

    print(f"target: sin(x) on [-pi, pi], {args.points} points")
    print(f"model:  a + b*x + c*x^2 + d*x^3   (4 parameters)")
    print(f"loss:   sum of squared error      optimizer: plain gradient descent")
    print(f"iters {args.iters}  lr {args.lr}\n")

    coeffs, losses = fit(x, y, args.iters, args.lr, args.seed)
    a, b, c, d = (float(v.detach()) for v in coeffs)
    print(f"first loss {losses[0]:.4f}   final loss {losses[-1]:.4f}")
    print(f"learned: y = {a:.5f} + {b:.5f} x + {c:.5f} x^2 + {d:.5f} x^3")
    print(f"taylor : y = 0.00000 + 1.00000 x + 0.00000 x^2 + -0.16667 x^3")
    print(f"train RMSE {rmse(predict(coeffs, x), y):.6f}")

    # 1. held-out x from the same range
    torch.manual_seed(args.seed + 1)
    xh = (torch.rand(args.points) * 2 - 1) * math.pi
    print(f"\nheld-out RMSE (same range, unseen x): "
          f"{rmse(predict(coeffs, xh), torch.sin(xh)):.6f}")

    # 2. outside the training range
    for lo, hi in ((math.pi, 2 * math.pi), (2 * math.pi, 3 * math.pi)):
        xe = torch.linspace(lo, hi, 500)
        print(f"extrapolation RMSE on [{lo / math.pi:.0f}pi, {hi / math.pi:.0f}pi]: "
              f"{rmse(predict(coeffs, xe), torch.sin(xe)):.6f}")

    # 3. a target that is not fully determined by the input
    torch.manual_seed(args.seed)
    y_noisy = torch.sin(x) + torch.randn(args.points) * args.noise
    coeffs_n, losses_n = fit(x, y_noisy, args.iters, args.lr, args.seed)
    pred_n = predict(coeffs_n, x)
    print(f"\nwith noise std {args.noise}:")
    print(f"  final loss {losses_n[-1]:.4f}")
    print(f"  RMSE vs the noisy observations: {rmse(pred_n, y_noisy):.6f}")
    print(f"  RMSE vs the true sin(x):        {rmse(pred_n, y):.6f}")
    print(f"  noise std itself:               {args.noise:.6f}")

    print("\n=== record these in docs/week4 ===")
    print(f"iterations:          {args.iters}")
    print(f"learning rate:       {args.lr}")
    print(f"final loss:          {losses[-1]:.4f}")
    print(f"train RMSE:          {rmse(predict(coeffs, x), y):.6f}")
    print(f"held-out RMSE:       {rmse(predict(coeffs, xh), torch.sin(xh)):.6f}")


if __name__ == "__main__":
    main()
