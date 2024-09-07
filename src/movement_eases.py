import math

def linear_ease(t):
    return t

def quadratic_ease_in(t):
    return t * t

def quadratic_ease_out(t):
    return t * (2 - t)

def quadratic_ease_io(t, ease_in_power=2, ease_out_power=2):
    if t < 0.5:
        return (t * 2) ** ease_in_power / 2
    else:
        return 1 - ((1 - t) * 2) ** ease_out_power / 2

def cubic_ease_in(t):
    return t * t * t

def cubic_ease_out(t):
    return (t -1) * t * t + 1

def cubic_ease_in_out(t):
    if t < 0.5:
        return 4 * t * t * t
    else:
        return (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
    
def quartic_ease_in(t):
    return t * t * t * t

def quartic_ease_out(t):
    return 1 - (t-1) * t * t * t

def quartic_ease_in_out(t):
    if t < 0.5:
        return 8 * t * t * t * t
    else:
        return 1 - 8 * (t-1) * t * t * t

def exponential_ease_in(t):
    return 2**(10 * (t - 1)) if t != 0 else 0

def exponential_ease_out(t):
    return 1 - 2**(-10 * t) if t != 1 else 1

def exponential_ease_in_out(t):
    if t == 0 or t == 1:
        return t
    if t < 0.5:
        return 0.5 * 2**(10 * (2 * t - 1))
    else:
        return 0.5 * (2 - 2**(-10 * (2 * t - 1)))

def elastic_ease_in(t):
    return math.sin(13 * math.pi / 2 * t) * 2**(10 * (t - 1))

def elastic_ease_out(t):
    return math.sin(-13 * math.pi / 2 * (t + 1)) * 2**(-10 * t) + 1

def elastic_ease_in_out(t):
    if t < 0.5:
        return 0.5 * math.sin(13 * math.pi / 2 * (2 * t)) * 2**(10 * (2 * t - 1))
    else:
        return 0.5 * (math.sin(-13 * math.pi / 2 * (2 * t - 1 + 1)) * 2**(-10 * (2 * t - 1)) + 2)
    
def bounce_ease_in(t):
    return 1 - bounce_ease_out(1 - t)

def bounce_ease_out(t):
    if t < 1/2.75:
        return 7.5625 * t * t
    elif t < 2/2.75:
        t -= 1.5/2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5/2.75:
        t -= 2.25/2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625/2.75
        return 7.5625 * t * t + 0.984375

# def bounce_ease_in_out(t):
#     if t < 0:

def back_ease_in(t, overshoot=1.70158):
    return t * t * ((overshoot + 1) * t - overshoot)

def back_ease_out(t, overshoot=1.70158):
    t -= 1
    return t * t * ((overshoot + 1) * t + overshoot) + 1

def back_ease_in_out(t, overshoot=1.70158):
    overshoot *= 1.525
    t *= 2
    if t < 1:
        return 0.5 * (t * t * ((overshoot + 1) * t - overshoot))
    else:
        t -= 2
        return 0.5 * (t * t * ((overshoot + 1) * t + overshoot) + 2)

def bounce_inverted_ease_out(t):
    if t < 1/2.75:
        return 7.5625 * (1-t) * (1-t)
    elif t < 2/2.75:
        t -= 1.5/2.75
        return 7.5625 * (1-t) * (1-t) + 0.75
    elif t < 2.5/2.75:
        t -= 2.25/2.75
        return 7.5625 * (1-t) * (1-t) + 0.9375
    else:
        t -= 2.625/2.75
        return 7.5625 * (1-t) * (1-t) + 0.984375

def bounce_inverted_ease_in(t):
    return 1 - bounce_inverted_ease_out(1 - t)

def bounce_inverted_ease_in_out(t):
    if t < 0.5:
        return bounce_inverted_ease_in(t * 2) * 0.5
    return bounce_inverted_ease_out(t * 2 - 1) * 0.5 + 0.5

def hermite_ease(t):
    return 3 * t**2 - 2 * t**3

def exponential_back_ease_in(t):
    return t * t * ((2 + 1) * t - 2) * 2 ** (10 * (t - 1))

def exponential_back_ease_out(t):
    t -= 1
    return t * t * ((2 + 1) * t + 2) * 2 ** (-10 * t) + 1

def exponential_back_ease_in_out(t):
    if t < 0.5:
        return 0.5 * exponential_back_ease_in(t * 2)
    else:
        return 0.5 * exponential_back_ease_out(t * 2 - 1) + 0.5