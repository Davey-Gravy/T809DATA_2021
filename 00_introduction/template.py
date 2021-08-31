import numpy as np
import matplotlib.pyplot as plt

def normal_prob(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    return ( 1 / np.sqrt(2 * np.pi * np.power(sigma, 2)) ) * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    xs = np.linspace(x_start, x_end, num=500)
    plt.plot(xs, normal_prob(xs, sigma, mu))

def _plot_three_normals():
    # Part 1.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1, 1.5, -5, 5)
    plt.show()

def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
    # Part 2.1
    sum = 0
    for i in range(len(sigmas)):
        sum += (weights[i] / np.sqrt(2 * np.pi * np.power(sigmas[i], 2))) * np.exp(-np.power(x-mus[i], 2) / (2 * np.power(sigmas[i], 2)))
    return sum

def plot_mixture(x: np.ndarray,sigmas: list, mus:list, weights: list):
    # Part 1.2
    plt.plot(x, normal_mixture(x, sigmas, mus, weights))
    
def _compare_components_and_mixture():
    # Part 2.2
    plot_normal(0.5, 0, -5, 5)
    plot_normal(1.5, -0.5, -5, 5)
    plot_normal(0.25, 1.5, -5, 5)
    xs = np.linspace(-5, 5, 500)
    plt.plot(xs, normal_mixture(xs, [0.5, 1.5, 0.25], [0, -0.5, 1.5], [1/3, 1/3, 1/3]))
    plt.show()

def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
    # Part 3.1
    p = np.random.multinomial(n_samples, weights)
    arry = np.ndarray(shape=n_samples, dtype=float)
    print(p)
    n = 0
    for i in range(len(p)):
        for j in range(p[i]):
            arry[n] = np.random.normal(mus[i], sigmas[i])
            n += 1
    return arry

def _plot_mixture_and_samples():
    # Part 3.2
    sigmas  = [0.3, 0.5, 1]
    mus     = [0, -1, 1.5]
    weights = [0.2, 0.3, 0.5]
    xs = np.linspace(-5, 5, 500)
    num = 1
    for i in [10, 100, 500, 1000]:
        plt.subplot(220+num)
        plot_mixture(xs, sigmas, mus, weights)
        samples = sample_gaussian_mixture(sigmas, mus, weights, i)
        plt.hist(samples, 100, density=True)
        num += 1
    plt.show()

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    # normal_prob(0, 1, 0)
    # plot_normal(0.5, 0, -2, 2)
    # _plot_three_normals()
    # normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
    # _compare_components_and_mixture()
    # plot_mixture([0.5, 1.5, 0.25], [0, -0.5, 1.5], [1/3, 1/3, 1/3], -5, 5)
    # sample_gaussian_mixture([0, 1, 1.5], [-1, 1, 5], [0.1, 0.1, 0.8], 10)
    _plot_mixture_and_samples()
    # print(sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3))