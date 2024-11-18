import numpy as np
from scipy.stats import t
from scipy.optimize import minimize
from scipy.special import gammaln

class UDCS_t_Model:
    def __init__(self, omega, phi, k, varsigma, nu):
        self.omega = omega
        self.phi = phi
        self.k = k
        self.varsigma = varsigma
        self.nu = nu

    def uSTDT_rnd(self, n, mu):
        t_dist = t(df=self.nu, loc=mu, scale=np.sqrt(self.varsigma))
        return t_dist.rvs(size=n)
    
    def martingale_diff_u_t(self, y, mu_t, varsigma, nu):
        return (1 / (1 + ((y - mu_t) ** 2) / (nu * varsigma))) * (y - mu_t)


    def uSTDT_uDCS_t(self, y, mu_t, varsigma, nu, log=True):
        ulpdf = (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(varsigma)
            - 0.5 * np.log(np.pi * nu)
            - ((nu + 1) / 2) * np.log(1 + ((y - mu_t) ** 2) / (nu * varsigma))
        )

        return ulpdf if log else np.exp(ulpdf)

    def simulate(self, T):
        y = np.empty(T)
        mu_t = np.empty(T)
        u_t = np.empty(T - 1)

        mu_t[0] = self.omega
        y[0] = self.uSTDT_rnd(1, mu_t[0])

        for t in range(1, T):
            u_t[t - 1] = self.martingale_diff_u_t(y[t - 1], mu_t[t - 1])
            mu_t[t] = self.omega + self.phi * (mu_t[t - 1] - self.omega) + self.k * u_t[t - 1]
            y[t] = self.uSTDT_rnd(1, mu_t[t])

        return {'y_t_gen': y, 'Dynamic_Location': mu_t, 'Innovation_u_t': u_t}

    def fit(self, dati):
        T = len(dati)

        bounds = [(-np.inf, np.inf), (-0.999, 0.999), (-2, 2), (1e-5, np.inf), (2.099, 300)]
        theta_initial = [self.omega, self.phi, self.k, self.varsigma, self.nu]

        def objective(theta):
            return -self.filter(dati, theta)['Log_Likelihood']

        result = minimize(objective, theta_initial, method='L-BFGS-B', bounds=bounds)
        optimized_params = result.x
        self.omega, self.phi, self.k, self.varsigma, self.nu = optimized_params

        return {'theta': optimized_params, 'optimizer': result}

    def filter(self, dati, theta=None):
        T = len(dati)
        
        if theta is None:
            theta = [self.omega, self.phi, self.k, self.varsigma, self.nu]

        omega, phi, k, varsigma, nu = theta

        mu_t = np.empty(T)
        u_t = np.empty(T - 1)
        dloglik = np.empty(T - 1)
        loglik = 0.0

        mu_t[0] = omega

        dloglik[0] = self.uSTDT_uDCS_t(dati[0], mu_t[0], varsigma, nu, log=True)
        loglik += dloglik[0]

        for t in range(1, T):
            u_t[t - 1] = self.martingale_diff_u_t(dati[t - 1], mu_t[t - 1], varsigma, nu)
            mu_t[t] = omega + phi * (mu_t[t - 1] - omega) + k * u_t[t - 1]

            if t < T:
                dloglik[t - 1] = self.uSTDT_uDCS_t(dati[t], mu_t[t], varsigma, nu, log=True)
                loglik += dloglik[t - 1]

        mu_t = mu_t[:-1]
        v_t = dati[1:] - mu_t

        return {
            'Dynamic_Location': mu_t,
            'Innovation_u_t': u_t,
            'Innovation_v_t': v_t,
            'Log_Densities_i': dloglik,
            'Log_Likelihood': loglik,
        }
    
    def predict(self, dati, future_steps):
        T = len(dati)
        predictions = np.empty(future_steps)
        mu_t = np.empty(T + future_steps)
        mu_t[0] = self.omega

        for t in range(T - 1):
            u_t = self.martingale_diff_u_t(dati[t], mu_t[t], self.varsigma, self.nu)
            mu_t[t + 1] = self.omega + self.phi * (mu_t[t] - self.omega) + self.k * u_t

        for t in range(future_steps):
            u_t = self.martingale_diff_u_t(0 if t == 0 else predictions[t - 1], mu_t[T + t - 1], self.varsigma, self.nu)
            mu_t[T + t] = self.omega + self.phi * (mu_t[T + t - 1] - self.omega) + self.k * u_t
            predictions[t] = self.uSTDT_rnd(1, mu_t[T + t])

        return predictions

