import numpy as np
from scipy.stats import t

def uDCS_t_model_simulator(T, omega, phi, k, varsigma, nu):
    y = np.empty(T)
    mu_t = np.empty(T)
    u_t = np.empty(T - 1)

    mu_t[0] = omega
    y[0] = uSTDT_rnd(1, mu_t[0], varsigma, nu)

    for t in range(1, T):
        u_t[t - 1] = martingale_diff_u_t(y[t - 1], mu_t[t - 1], varsigma, nu)
        mu_t[t] = omega + phi * (mu_t[t - 1] - omega) + k * u_t[t - 1]
        y[t] = uSTDT_rnd(1, mu_t[t], varsigma, nu)

    out = {
        'y_t_gen': y,
        'Dynamic_Location': mu_t,
        'Innovation_u_t': u_t
    }

    return out


def uSTDT_rnd(n, mu, varsigma, nu):
    t_dist = t(df=nu, loc=mu, scale=np.sqrt(varsigma))
    y = t_dist.rvs(size=n)
    return y


import numpy as np
from scipy.optimize import minimize


def uDCS_t_model_estimator(dati, param):
    # Take T
    T = len(dati)

    # Parameter Selections Dynamic Location
    omega = param[0]
    phi = param[1]
    k = param[2]
    varsigma = param[3]
    nu = param[4]

    # Create a vector with the parameters
    theta_st = np.array([omega, phi, k, varsigma, nu])

    # Take Bounds
    lower = np.array([-np.inf, -0.999, -2, 1e-05, 2.099])
    upper = np.array([np.inf, 0.999, 2, np.inf, 300])

    # Define the objective function
    objective = lambda theta: interprete_uDCS_t_model(dati, theta)

    # Optimize using L-BFGS-B
    optimizer = minimize(objective, theta_st, method='L-BFGS-B', bounds=list(zip(lower, upper)))

    # Save the optimized parameters Dynamic Location
    omega_opt = optimizer.x[0]
    phi_opt = optimizer.x[1]
    k_opt = optimizer.x[2]
    varsigma_opt = optimizer.x[3]
    nu_opt = optimizer.x[4]

    # Create a vector with ALL the optimized parameters
    theta_opt = np.array([omega_opt, phi_opt, k_opt, varsigma_opt, nu_opt])

    # Create a dictionary with ALL the optimized parameters
    theta_dict = {
        'omega': omega_opt,
        'phi': phi_opt,
        'k': k_opt,
        'varsigma': varsigma_opt,
        'nu': nu_opt
    }

    # Make dictionary for output
    out = {
        'theta_list': theta_dict,
        'theta': theta_opt,
        'optimizer': optimizer
    }

    return out


def interprete_uDCS_t_model(dati, param):
    # Take T
    T = len(dati)

    # Parameter Selections Dynamic Location
    omega = param[0]
    phi = param[1]
    k = param[2]
    varsigma = param[3]
    nu = param[4]

    # Create a new vector with the parameters
    theta_new = np.array([omega, phi, k, varsigma, nu])

    # Fitness function
    fitness = uDCS_t_model_filter(dati, theta_new)['Log_Likelihood']

    if np.isnan(fitness) or not np.isfinite(fitness):
        fitness = -1e10

    return -fitness

import numpy as np
from scipy.special import gammaln

def uDCS_t_model_filter(y, theta):
    # Take T
    T = len(y)

    # Define Log-Likelihoods
    dloglik = np.empty(T-1)
    loglik = 0.0

    # Parameter Selections Dynamic Location
    omega = theta[0]
    phi = theta[1]
    k = theta[2]
    varsigma = theta[3]
    nu = theta[4]

    # Define Dynamic Location and Innovations
    mu_t = np.empty(T)
    u_t = np.empty(T-1)
    v_t = np.empty(T-1)

    # Initialize Dynamic Location
    mu_t[0] = omega

    # Initialize Likelihood
    dloglik[0] = uSTDT_uDCS_t(y[0], mu_t[0], varsigma, nu, log=True)
    loglik = dloglik[0]

    for t in range(T-1):
        # Dynamic Location Innovations
        u_t[t] = martingale_diff_u_t(y[t], mu_t[t], varsigma, nu)
        # Updating Filter
        mu_t[t + 1] = omega + phi * (mu_t[t] - omega) + k * u_t[t]
        # Calculate v_t
        v_t[t] = y[t + 1] - mu_t[t + 1]

        if t < T - 2:
            # Updating Likelihoods
            dloglik[t + 1] = uSTDT_uDCS_t(y[t + 1], mu_t[t + 1], varsigma, nu, log=True)
            loglik += dloglik[t + 1]


    ######################
    ####### OUTPUT #######
    ######################
    mu_t = mu_t[:-1] 
    u_t = u_t

    # Make dictionary for output
    out = {
        'Dynamic_Location': mu_t,
        'Innovation_u_t': u_t,
        'Innovation_v_t': v_t,
        'Log_Densities_i': dloglik,
        'Log_Likelihood': loglik
    }

    return out



def martingale_diff_u_t(y, mu_t, varsigma, nu):
    u_t = (1 / (1 + ((y - mu_t)**2) / (nu * varsigma))) * (y - mu_t)
    return u_t


def uSTDT_uDCS_t(y, mu_t, varsigma, nu, log=True):
    ulpdf = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - (1 / 2) * np.log(varsigma) -
             (1 / 2) * np.log(np.pi * nu) - ((nu + 1) / 2) * np.log(1 + ((y - mu_t)**2) / (nu * varsigma)))

    if not log:
        ulpdf = np.exp(ulpdf)

    return ulpdf


def uDCS_t_model_predict(dati, theta, future_steps):
    T = len(dati)
    omega, phi, k, varsigma, nu = theta
    
    # Perform the filtering step to estimate dynamic location and innovations
    # Similar to the uDCS_t_model_filter function, but only update mu_t and u_t
    mu_t = np.empty(T)
    u_t = np.empty(T-1)
    
    # Initialize Dynamic Location
    mu_t[0] = omega

    for t in range(T-1):
        u_t[t] = martingale_diff_u_t(dati[t], mu_t[t], varsigma, nu)
        mu_t[t + 1] = omega + phi * (mu_t[t] - omega) + k * u_t[t]

    # Generate predictions for future time steps
    future_steps = future_steps  # Replace this with the desired number of future time steps
    predictions = np.empty(future_steps)

    for t in range(T, T + future_steps):
        u_t_pred = martingale_diff_u_t(predictions[t - T - 1], mu_t[-1], varsigma, nu)
        mu_t_pred = omega + phi * (mu_t[-1] - omega) + k * u_t_pred
        # Use the predicted mu_t_pred to generate a new value for the next time step
        y_pred = uSTDT_rnd(1, mu_t_pred, varsigma, nu)
        predictions[t - T] = y_pred

        # Update mu_t for the next prediction
        mu_t = np.append(mu_t, mu_t_pred)

    return predictions
