import scipy.stats as stats

class DistributionFactory:
    """
    A factory class for creating SciPy stats distribution objects.
    """
    
    def __init__(self):
        self.distributions = {
            'norm': stats.norm,
            'expon': stats.expon,
            'gamma': stats.gamma,
            'weibull_min': stats.weibull_min,
            'uniform': stats.uniform,
            'binom': stats.binom
        }
    
    def get_distribution(self, distribution_name):
        """
        Returns a SciPy stats distribution object based on the specified distribution name.
        
        Args:
        distribution_name (str): The name of the distribution.
        
        Returns:
        A SciPy stats distribution object corresponding to the specified distribution name.
        """
        try:
            return self.distributions[distribution_name]
        except KeyError:
            raise ValueError('Invalid distribution name.')
