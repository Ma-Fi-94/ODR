# ODR
Convenience function for fitting an orthogonal distance regression using scipy. We account for pointwise x and y errors, and use bootstrapping to assess predictive uncertainty, since reduced chi-square becomes problematic in case of nonlinear functions (the number of degrees of freedom cannot be properly assessed).
