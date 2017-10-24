import numpy as np

def eff(a, RETRACE_PNTS):
    pivots = []
    up = True
    p1, p2 = 0, 0
    p1_bn, p2_bn = 0, 0

    for x, y in a.iterrows():
        y = y.values[0]
        if up:
            if y > p1:
                p2_bn = p1_bn = x
                p2 = p1 = y
            elif y < p2:
                p2_bn = x
                p2 = y
        else:
            if y < p1:
                p2_bn = p1_bn = x
                p2 = p1 = y
            elif y > p2:
                p2_bn = x
                p2 = y

    # Found new pivot
        if abs(p1 - p2) >= RETRACE_PNTS:
            pivots.append([p1_bn, p1])
            up = not up
            p0_bn, p0 = p1_bn, p1
            p1_bn, p1 = p2_bn, p2

    if pivots[0][0] == 0:
        pivots = pivots[1:]
    pivots = np.array(pivots).T

    return pivots

'''
fig = plt.figure()
ax1 = fig.add_axes([0.03, 0.03, 0.95, 0.95]) # left, bottom, width, height
ax1.grid(True)
ax1.plot(a, 'k-')
print(pivots)
xs, ys = [], []
for x, y in pivots:
	xs.append(x)
	ys.append(y)
ax1.plot(xs, ys, 'r-')
plt.show()
'''










