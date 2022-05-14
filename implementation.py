import numpy as np
import pandas as pd
from scipy.odr import *
import random
import tqdm as tqdm

def odr_bs(f, x, y, beta0, sx = None, sy = None, n_bs = 100):
    # Explicitly doing what ODR does internally anyway,
    # so that resampling later doesn't cause problems
    if sx is None:
        sx = np.ones_like(x)
    if sy is None:
        sy = np.ones_like(y)
        
    # Fit model f to data (x,y) with initial guess beta0 via ODR
    model = Model(f)
    data = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(data, model, beta0=beta0).run()
    
    # Bootstrapping
    bs_samples = []
    bs_odrs = []
    bs_params = []
    np.random.seed(0)
    df = pd.DataFrame({"x": x, "y": y, "sx": sx, "sy": sy})
    for _ in tqdm.tqdm(range(n_bs)):
        bs_sample = df.sample(n=len(df), replace=True)
        bs_data = RealData(bs_sample.x, bs_sample.y, sx=bs_sample.sx, sy=bs_sample.sy)
        bs_odr = ODR(bs_data, model, beta0=beta0).run()
        
        bs_samples.append(bs_sample)
        bs_odrs.append(bs_odr)
        bs_params.append(bs_odr.beta)
        
    
    # Find bootstrap prediction for every x value
    y_predictions = []
    for xval in x:
        # Select all models which were not trained on the current x value
        selection = [xval not in bs_sample.x for bs_sample in bs_samples]
        selected_odrs = np.array(bs_odrs)[selection]
        
        # Predict y value using selected models, and save prediction
        current_predictions = [f(bs_odr.beta, xval) for bs_odr in selected_odrs]
        y_predictions.append(current_predictions)

    
    return odr, bs_params, y_predictions
    
    