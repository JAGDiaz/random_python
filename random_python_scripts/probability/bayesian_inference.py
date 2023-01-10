import scipy.stats as sts
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num = 500)
test = np.linspace(0, 2)
uniform_dist = sts.uniform.pdf(mu) + 1 
# sneaky advanced note: I'm using the uniform distribution for clarity, 
# but we can also make the beta distribution look completely flat by tweaking alpha and beta!
uniform_dist = uniform_dist/uniform_dist.sum() 
#Normalizing the distribution to make the probability densities sum into 1
beta_dist = sts.beta.pdf(mu, 2, 5, loc = 1.65, scale = 0.2) 
beta_dist = beta_dist/beta_dist.sum()


def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    # Note that mu here is an array of values, so the output is also an array!
    return likelihood_out/likelihood_out.sum()

def compute_percentile(parameter_values, distribution_values, percentile):
    cumulative_distribution = integrate.cumtrapz(
        distribution_values, parameter_values)
    percentile_index = np.searchsorted(cumulative_distribution, percentile)
    return parameter_values[percentile_index]


heights_data = sts.norm.rvs(loc = 1.7, scale = 0.1, size = 10001)

prior = uniform_dist
posterior_dict = {}

plt.figure(figsize = (10, 8))

for ind, datum in enumerate(heights_data):

    if ind % 1000 == 0:
        plt.plot(mu, prior, label = f'Model after observing {ind} data')

    likelihood = likelihood_func(datum, mu)
    unnormalized_posterior = prior * likelihood
    normalized_posterior = unnormalized_posterior/integrate.trapz(unnormalized_posterior, mu)
    prior = normalized_posterior
    posterior_dict[ind] = normalized_posterior

plt.legend()
plt.show()

plt.plot(mu, prior, label="Final Posterior Model")
plt.axvline(x = compute_percentile(mu, posterior_dict[1000], 0.005), ls = '--', color = 'y', label = '99% Conf Int')
plt.axvline(x = compute_percentile(mu, posterior_dict[1000], 0.995), ls = '--', color = 'y')
plt.legend()
plt.show()

