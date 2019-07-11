import math
import numpy as np


def Q_BE_fn(phi,theta):

    Q_BE=np.array( [[1, math.sin(phi)*math.tan(theta), math.cos(phi)*math.tan(theta)],[0, math.cos(phi), -math.sin(phi)],[0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]])

    return Q_BE

