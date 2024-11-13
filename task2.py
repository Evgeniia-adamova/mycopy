def power_calc(x, y):
    if y <= 1:
        return x == 1
    if x < 1:
        return False

    while x > 1:
        if x % y != 0:
            return False
        x = x // y

    return x == 1


print(power_calc(16, 2))
