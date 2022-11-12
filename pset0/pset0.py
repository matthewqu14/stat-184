import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Part 1
A = np.array([[0, 2, 4],
              [2, 4, 2],
              [3, 3, 1]])

A_inv = np.linalg.inv(A)

b = np.array([-2, -2, -4])
c = np.array([1, 1, 1])

print(A_inv.dot(b))
print(A.dot(c))

# Part 2
n = 40000
z = np.random.randn(n)

plt.figure(figsize=(12, 8))
plt.step(sorted(z), np.arange(1, n + 1) / float(n))
plt.xlim(-3, 3)
plt.xlabel("x")
plt.ylabel("Empirical CDF")
plt.show()

plt.figure(figsize=(12, 8))
for k in [1, 8, 64, 512]:
    y = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1. / k), axis=1)
    plt.step(sorted(y), np.arange(1, n + 1) / float(n), label=f"k = {k}")

plt.step(sorted(z), np.arange(1, n + 1) / float(n), label="True CDF")
plt.xlim(-3, 3)
plt.xlabel("x")
plt.ylabel("Empirical CDF")
plt.legend()
plt.show()
