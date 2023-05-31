import numpy as np 
import pandas as pd

def reshape_to_matrix(x):
    """
    Reshapes a 1D numpy array to a 2D column vector if necessary.
    """
    if x.ndim == 1:
        return np.reshape(x, (x.size, 1))
    else:
        return x


def generate_common_factors(f_initial, discard_observations, theta, T, m):
    """
    Generates common factors for the specified number of observations, time periods, and factors.

    Args:
        f_initial: Initial value of the common factors.
        discard_observations: Number of observations to discard.
        theta: Autoregressive parameter in the GDP of the factors.
        T: Number of time periods.
        m: Number of common factors.

    Returns:
        A tuple containing the common factors f_j_t and discarded factors f_discard.
    """

    f_discard = np.zeros((discard_observations, m))
    mu_j_t = np.random.normal(0, (1 - theta*theta)/m, size=(discard_observations, m))
    theta_vect = np.full((1, m), theta)
    f_initial = np.full((1, m), f_initial)

    f_discard[0] = theta_vect * f_initial + mu_j_t[0]

    for i in range(1, discard_observations):
        # Generate discarded common factors based on the persistence parameter and random shocks
        f_discard[i] = theta_vect * f_discard[i-1] + mu_j_t[0]

    f_j_t = np.zeros((T, m))
    mu_j_t = np.random.normal(0, (1 - theta*theta)/m, size=(T, m))

    f_j_t[0] = theta_vect * f_discard[-1] + mu_j_t[0]

    for i in range(1, T):
        # Generate common factors for each time period based on the persistence parameter and random shocks
        f_j_t[i] = theta_vect * f_j_t[i-1] + mu_j_t[0]

    return f_j_t, f_discard


def generate_factor_loadings(N, gamma_u, m):
    """
    Generates factor loadings for the specified number of groups and factors.

    Args:
        N: Number of cross-sectional units.
        gamma_u: Upper bound of the idiosyncratic errors in the total variance of yit.
        m: Number of common factors.

    Returns:
        A tuple containing the factor loadings C, gamma_accent, GAMMA_transpose_x, and GAMMA_transpose_g.
    """

    C = []

    if m == 1:
        gamma_accent = np.zeros(N)
        GAMMA_transpose_x = np.zeros(N)
        GAMMA_transpose_g = np.zeros(N)

        for n in range(N):
            # Generate factor loadings for a single factor (m=1)
            C.append(np.matrix([[np.random.uniform(0, gamma_u)],        # Loading for the idiosyncratic shock
                                [np.random.uniform(0, 1)],              # Loading for the observable factor (x)
                                [np.random.uniform(-0.6, 0)]]))          # Loading for the latent factor (g)
            gamma_accent[n] = C[n][0]
            GAMMA_transpose_x[n] = C[n][1]
            GAMMA_transpose_g[n] = C[n][2]

    elif m == 2:
        gamma_accent = np.zeros((N, 2))
        GAMMA_transpose_x = np.zeros((N, 2))
        GAMMA_transpose_g = np.zeros((N, 2))

        for n in range(N):
            # Generate factor loadings for two factors (m=2)
            C.append(np.matrix([[np.random.uniform(0, gamma_u),         # Loading for the idiosyncratic shock
                                 np.random.uniform(0, gamma_u - 0.6)],  # Loading for the first observable factor (x)
                                [np.random.uniform(0, 1),               # Loading for the first latent factor (g)
                                 np.random.uniform(0, 0.2)],            # Loading for the second observable factor (x)
                                [np.random.uniform(-0.6, 0),            # Loading for the second latent factor (g)
                                 np.random.uniform(-1.4, 0)]]))
            gamma_accent[n] = C[n][0]
            GAMMA_transpose_x[n] = C[n][1]
            GAMMA_transpose_g[n] = C[n][2]

    return C, gamma_accent, GAMMA_transpose_x, GAMMA_transpose_g




def generate_z(z_initial, f_j_init, theta, discard_observations, lags, lambda_, N, T, m, gamma_u):
    """
    Generates z and x values based on the provided parameters.

    Args:
        z_initial: Initial values for z.
        f_j_init: Initial values for the common factors f.
        theta: Autoregressive parameter in the GDP of the factors.
        discard_observations: Number of observations to discard before generating the final data.
        lags: Number of lags.
        lambda_: Coefficients for the lags.
        N: Number of cross-sectional units.
        T: Number of time periods.
        m: Number of common factors.
        gamma_u: Upper bound of the idiosyncratic errors in the total variance of yit.

    Returns:
        A list containing the generated z and x values, followed by x and z discard arrays.
    """

    # Generate factor loadings
    C, gamma_accent, GAMMA_transpose_x, GAMMA_transpose_g = generate_factor_loadings(N, gamma_u, m)

    # Generate common factors
    f_t, f_j_t = generate_common_factors(f_j_init, 50, theta, T, m)

    # Generate fixed effects 
    v_i = np.random.normal(0, (1 - lambda_[0] * lambda_[0]), size=(discard_observations, N))
    c_i = np.random.normal(0, (1 - lambda_[0] * lambda_[0]), size=(discard_observations, N))

    x_discard = np.zeros((discard_observations, N))
    g_discard = np.zeros((discard_observations, N))

    dynamic_sum_x = 0
    dynamic_sum_g = 0

    # Generate discarded x and g values
    for p in range(1, lags + 1):
        dynamic_sum_x += lambda_[p - 1] * z_initial
        dynamic_sum_g += lambda_[p - 1] * z_initial

    x_discard[0] = (reshape_to_matrix(c_i[0]) + dynamic_sum_x + np.matmul(reshape_to_matrix(GAMMA_transpose_x),
                                                                           reshape_to_matrix(f_j_t[0])) +
                    reshape_to_matrix(v_i[0])).flatten()
    g_discard[0] = (reshape_to_matrix(c_i[0]) + dynamic_sum_g + np.matmul(reshape_to_matrix(GAMMA_transpose_g),
                                                                           reshape_to_matrix(f_j_t[0])) +
                    reshape_to_matrix(v_i[0])).flatten()

    for i in range(1, discard_observations):
        dynamic_sum_x = np.zeros((1, N))
        dynamic_sum_g = np.zeros((1, N))

        for p in range(1, lags + 1):
            dynamic_sum_x += lambda_[p - 1] * x_discard[i - p]
            dynamic_sum_g += lambda_[p - 1] * g_discard[i - p]

        x_discard[i] = (reshape_to_matrix(c_i[i]) + dynamic_sum_x.transpose() +
                        np.matmul(reshape_to_matrix(GAMMA_transpose_x), reshape_to_matrix(f_j_t[i])) +
                        reshape_to_matrix(v_i[i])).flatten()
        g_discard[i] = (reshape_to_matrix(c_i[i]) + dynamic_sum_g.transpose() +
                        np.matmul(reshape_to_matrix(GAMMA_transpose_g), reshape_to_matrix(f_j_t[i])) +
                        reshape_to_matrix(v_i[i])).flatten()

    # Generate fixed effects for final x and g
    v_i = np.random.normal(0, (1 - lambda_[0] * lambda_[0]), size=(T, N))
    c_i = np.random.normal(0, (1 - lambda_[0] * lambda_[0]), size=(T, N))

    x_initial = x_discard[discard_observations - 1]
    g_initial = g_discard[discard_observations - 1]

    x = np.zeros((T, N))
    g = np.zeros((T, N))

    dynamic_sum_x = 0
    dynamic_sum_g = 0

    # Generate final x and g values
    for p in range(1, lags + 1):
        dynamic_sum_x += lambda_[p - 1] * x_initial
        dynamic_sum_g += lambda_[p - 1] * g_initial

    x[0] = (reshape_to_matrix(c_i[0]) + reshape_to_matrix(dynamic_sum_x) +
            np.matmul(reshape_to_matrix(GAMMA_transpose_x), reshape_to_matrix(f_t[0])) +
            reshape_to_matrix(v_i[0])).flatten()
    g[0] = (reshape_to_matrix(c_i[0]) + reshape_to_matrix(dynamic_sum_g) +
            np.matmul(reshape_to_matrix(GAMMA_transpose_g), reshape_to_matrix(f_t[0])) +
            reshape_to_matrix(v_i[0])).flatten()

    for i in range(1, T):
        dynamic_sum_x = np.zeros((1, N))
        dynamic_sum_g = np.zeros((1, N))

        for p in range(1, lags + 1):
            dynamic_sum_x += lambda_[p - 1] * x[i - p]
            dynamic_sum_g += lambda_[p - 1] * g[i - p]

        x[i] = (reshape_to_matrix(c_i[i]) + dynamic_sum_x.transpose() +
                np.matmul(reshape_to_matrix(GAMMA_transpose_x), reshape_to_matrix(f_t[i])) +
                reshape_to_matrix(v_i[i])).flatten()
        g[i] = (reshape_to_matrix(c_i[i]) + dynamic_sum_g.transpose() +
                np.matmul(reshape_to_matrix(GAMMA_transpose_g), reshape_to_matrix(f_t[i])) +
                reshape_to_matrix(v_i[i])).flatten()

    return [x, g], x, g, x_discard, g_discard



def generate_data(y_i_init, z_i_init, f_j_init, discard_observations, T, N, beta, rho, lambda_, lags, m, gamma_u, theta):
    """
    Generates synthetic data based on the provided parameters.

    Args:
        y_i_init: Initial values for the dependent variable y.
        z_i_init: Initial values for the exogenous regressor z.
        f_j_init: Initial values for the common factors f.
        discard_observations: Number of observations to discard before generating the final data.
        T: Number of time periods for the final data.
        N: Number of cross-sectional units for the final data.
        beta: Coefficient for the exogenous regressor in the data generation process.
        rho: Autoregressive coefficient in the data generation process.
        lambda_: Coefficient for the lags in the data generation process.
        lags: Number of lags for the factor in the data generation process.
        m: Number of common factors.
        gamma_u: Upper bound of the idiosyncratic errors n the total variance of yit.
        theta: Autoregressive parameter in the GDP of the factors.

    Returns:
        df: A pandas DataFrame containing the generated data with columns 'N', 'T', 'y', and 'X'.
    """

    # Generate z and x values using the generate_z function
    _, x, _, x_discard, _ = generate_z(z_i_init, f_j_init, theta, discard_observations, lags, lambda_, N, T, m, gamma_u)

    # Initialize arrays for discarded data and final data
    y_discard = np.zeros((discard_observations, N))
    y = np.zeros((T, N))

    # Generate discarded data
    alpha = np.random.normal(0, (1 - rho * rho), size=(1, N))
    epsilon_i = np.random.normal(0, (1 - rho * rho), size=(discard_observations, N))
    y_discard[0] = alpha + rho * y_i_init + beta * x_discard[0] + epsilon_i[0]
    
    for i in range(1, discard_observations):
        y_discard[i] = alpha + rho * y_discard[i - 1] + beta * x_discard[i] + epsilon_i[i]

    # Generate final data
    alpha = np.random.normal(0, (1 - rho * rho), size=(1, N))
    epsilon_i = np.random.normal(0, (1 - rho * rho), size=(T, N))
    y[0] = alpha + rho * y_discard[discard_observations - 1] + beta * x[0] + epsilon_i[0]

    for t in range(1, T):
        y[t] = alpha + rho * y[t - 1] + beta * x[t] + epsilon_i[t]

    # Create a DataFrame with the generated data
    df = pd.DataFrame({'N': np.tile(range(1, N + 1), T), 'T': np.repeat(range(1, T + 1), repeats=N),
                       'y': np.concatenate(y), 'X': np.concatenate(x)})

    # Sort the DataFrame by 'N' and 'T' columns
    df = df.sort_values(by=['N', 'T'])
    
    return df



