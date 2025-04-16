# hand-gesture-tracker/utils/math_utils.py

import math

def calculate_distance(a, b):
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
