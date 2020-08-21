# Reference Codebook
## Numerical Optimization Routines from Scratch

### Contents:

#### Base Classes:

  - ProbabilityDistribution: base template class for all probability distributions
  - Optimizer: base template class for performing single-output function optimization
  - Kernel: base template class for kernel transformations
  
#### Probability Distribution Classes:

  - NormalDistribution1D(ProbabilityDistribution): Gaussian/normal distribution for one dimension
  - NormalDistribution2D(ProbabilityDistribution): Gaussian/normal bivariate distribution
  - MVNormalDistribution(ProbabilityDistribution): Multivariate Gaussian/normal distribution

#### Kernel Classes:

  - LinearKernel(Kernel): Linear (dot product) kernel
  - RBFKernel(Kernel): Gaussian/radial basis function kernel (AKA squared exponential kernel)
  - RationalQuadraticKernel(Kernel): Rational quadratic kernel


#### Optimizer Classes:

  - TBD