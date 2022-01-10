#! /usr/bin/env python
#-*- coding: UTF- -*-

# from pykalman import KalmanFilter
# from pykalman import UnscentedKalmanFilter

from pykalman import KalmanFilter
import numpy as np
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
# measurements = np.asarray([[1], [0], [0]])  # 3 observations
# kf = kf.em(measurements, n_iter=5)
# (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
measure = [0, 1, 2, 4,5,9,10,2]

print(kf.filter(measure))