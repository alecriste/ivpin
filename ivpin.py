import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def sig(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def logit(x):
    """Logit function (inverse of sigmoid)."""
    return np.log(x / (1 - x))

def get_volume_bars(df, threshold, bvc_window):
    """
    Generate volume bars from tick data 
    
    param df: pd.DataFrame with columns ['date_time', 'price', 'volume']
    return: pd.DataFrame with volume bars including ['date_time', 'price', 'volume', 't', 'buy_volume', 'bvc_buy_volume',]
    
    """

    bars = []
    
    current_volume = 0
    cum_buy_volume = 0
    open_price = None
    high_price = float('-inf')
    low_price = float('inf')
    close_price = None
    start_time = None
    
    prev_price = None

    for index, row in df.iterrows():
        # If starting a new bar
        if current_volume == 0:
            start_time = row['date_time']
            open_price = row['price']
                
        # Update closing price to the latest price
        close_price = row['price']
        
        # Update cumulative buy volume based on tick rule
        if prev_price is not None:
            if row['price'] > prev_price:
                cum_buy_volume += row['volume']
        
        # Update current volume
        current_volume += row['volume']
        
        # Check if the volume has reached the threshold
        if current_volume >= threshold:  
            # Time difference
            end_time = row['date_time']
            time_difference = end_time-start_time
            t = time_difference.microseconds

            bars.append({
                'date_time': start_time,
                'close': close_price,
                'volume': current_volume,
                't': t,
                'buy_volume': cum_buy_volume,
            })
            # Reset for the next bar
            current_volume = 0
            cum_buy_volume = 0
            start_time = None
        
        # Update previous price
        prev_price = row['price']
    
    # Create DataFrame from the bars data
    bars_df = pd.DataFrame(bars)
    volume = bars_df['volume']
    close = bars_df['close']
    bvc = volume * (close.diff() / close.diff().rolling(window=bvc_window).std()).apply(norm.cdf)
    bars_df['bvc_buy_volume'] = bvc
    print(bars_df)

    return bars_df.dropna()

def compute_log_likelihood(params, t, Vb, Vs):
    """Compute the log-likelihood for the IVPIN model."""
    alpha = sig(params[0])
    delta = sig(params[1])
    mu = params[2] ** 2
    eps = params[3] ** 2
    # print(f"Parameters: alpha={alpha}, delta={delta}, mu={mu}, eps={eps}")

    e_1 = np.log(alpha * delta) + Vb * np.log(eps) + Vs * np.log(eps + mu) - (2*eps + mu) * t
    e_2 = np.log(alpha * (1 - delta)) + Vb * np.log(eps + mu) + Vs * np.log(eps) - (2*eps + mu) * t
    e_3 = np.log(1 - alpha) + Vb * np.log(eps) + Vs * np.log(eps) - 2*eps * t
    e_max = np.max([e_1, e_2, e_3])

    log_likelihood = -np.sum(np.log(np.exp(e_1 - e_max) + np.exp(e_2 - e_max) + np.exp(e_3 - e_max)) + e_max)
    return log_likelihood


def get_iVPIN(data, fixed_threshold = None, daily_bucket_num = 50, bvc = False, bvc_window=5, window_size=20):
    """
    param df: pd.DataFrame with columns ['date_time', 'price', 'volume']
    bvc: Bool. Use bvc is true, else use tick rule 
    fixed_threshold: int or float. Size of each volume bar. If null, 50 daily volume buckets will be used
    
    """
    

    if (fixed_threshold == None):
        daily_volumes = data['volume'].resample('D').sum()
        threshold = daily_volumes.mean()/daily_bucket_num
    else:
        threshold = fixed_threshold
    data = data.reset_index(drop=False)

    bars = get_volume_bars(data, threshold, bvc_window)
    ivpin = pd.Series(index=bars.index, dtype=float)

    if bvc:
        Vb = bars['bvc_buy_volume']
    else:
        Vb = bars['buy_volume']
    V = bars['volume']
    Vs = V-Vb
    t = bars['t']
    t = t/1000000 #miccroseconds to seconds


    start_index = int(ivpin.index[window_size])

    '''
    Easley et al. (2012b) derive the VPIN estimator based on the argument of two moment conditions, 
    E[|VτB - VτS|] ≈ αμ and E[|VτB + VτS|] = 2ε + αμ, from the Poisson processes. 
    According to Ke et al. (2017), the two moment conditions should instead be expressed as
    E[|VτB-VτS||tτ;θ]≈ αμtτ and E[|VτB + VτS|tτ; θ ] = (2ε + αμ)tτ,
    '''
    total_arrival_rate = bars['volume']/bars['t']
    informed_arrival = abs(Vb-Vs)*t

    j=0
    for i in range(start_index, ivpin.index[-1]+1):
        index = window_size+j
        parms = np.array([-np.inf, -np.inf, -np.inf, -np.inf]) # alpha, delta, mu, eps

        log_lik = np.inf
        flag = -np.inf
        
        initial_params = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        best_log_likelihood = np.inf
        exit_flag = False


        if i == start_index: 
            for alpha_init in np.arange(0.1, 0.9, 0.1):
                for delta_init in np.arange(0.1, 0.9, 0.1):
                    mu_init = informed_arrival[i] / alpha_init
                    eps_init = abs(total_arrival_rate[i] - informed_arrival[i]) / 2
                    initial_guess = np.array([logit(max(min(alpha_init, 0.999), 0.001)), logit(max(min(delta_init, 0.999), 0.001)), np.sqrt(mu_init), np.sqrt(eps_init)])
                    try:
                        result = minimize(compute_log_likelihood, initial_guess, args=(t[j:index], Vb[j:index], Vs[j:index]), method='BFGS')
                    except Exception as e:
                        print(f"Optimization failed with exception {e}")
                    if not result.success:
                        print(f"Optimization failed with message: {result.message}")

                    if result.success and result.fun < best_log_likelihood and np.isfinite(result.fun):
                        best_params = result.x
                        best_log_likelihood = result.fun
                        exit_flag = result.success
                    else:
                        print(result.success)
        else:
            initial_guess = np.array([logit(best_params[0]), logit(best_params[1]), np.sqrt(best_params[2]), np.sqrt(best_params[3])])
            try:
                result = minimize(compute_log_likelihood, initial_guess, args=(t[j:index], Vb[j:index], Vs[j:index]), method='BFGS')
            except Exception as e:
                print(f"Optimization failed with exception {e}")
            if result.success and result.fun < best_log_likelihood and np.isfinite(result.fun):
                best_params = result.x
            if not result.success:
                print(f"gagaga: {result.message}")
                for alpha_init in np.arange(0.1, 0.9, 0.1):
                    for delta_init in np.arange(0.1, 0.9, 0.1):
                        mu_init = informed_arrival[i] / alpha_init
                        eps_init = abs(total_arrival_rate[i] - informed_arrival[i]) / 2
                        initial_guess = np.array([logit(max(min(alpha_init, 0.999), 0.001)), logit(max(min(delta_init, 0.999), 0.001)), np.sqrt(mu_init), np.sqrt(eps_init)])
                        try:
                            result = minimize(compute_log_likelihood, initial_guess, args=(t[j:index], Vb[j:index], Vs[j:index]), method='BFGS')
                        except Exception as e:
                            print(f"Optimization failed with exception {e}")
                        if not result.success:
                            print(f"Optimization failed with message: {result.message}")

                        if result.success and result.fun < best_log_likelihood and np.isfinite(result.fun):
                            best_params = result.x
                            best_log_likelihood = result.fun
                            exit_flag = result.success
                        else:
                            print(result.success)


        j+=1

        # Transform parameters back to original scale
        best_params[0:2] = sig(best_params[0:2])
        best_params[2:4] = best_params[2:4] ** 2

        ivpin_estimate = best_params[0] * best_params[2] / (2 * best_params[3] + best_params[2])
        ivpin.loc[i] = ivpin_estimate
    print(ivpin)
    
    return ivpin_estimate


file_path = '/Users/alexb/Desktop/datasets/data/tick_data.csv'


data = pd.read_csv(file_path)
data = data.rename(columns={'Volume': 'volume', 'Date and Time': 'date_time', 'Price': 'price'})

data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S.%f')

data.set_index('date_time', inplace=True)


ivpin = get_iVPIN(data)