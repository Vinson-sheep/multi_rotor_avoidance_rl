#! /usr/bin/env python
#-*- coding: UTF- -*-

import numpy as np
import os
import matplotlib.pyplot as plt

url = os.path.dirname(os.path.realpath(__file__))
y = np.load(url + "/step_rewards.npy")
x = np.array(range(y.size))

plt.plot(x, y)
plt.show()