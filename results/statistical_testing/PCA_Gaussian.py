import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]

xC = np.array([2, 1])
sig = np.array([2, 0.5])

theta = np.pi/3

R = np.array([[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])

nPoints = 10000
X = np.matmul(R, np.matmul(np.diag(sig), np.random.randn(2, nPoints)))+ np.matmul(np.diag(xC) , np.ones([2, nPoints]))
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(X[0,:], X[1,:], '.', Color='k')
ax1.grid()
plt.xlim((-6, 8))
plt.ylim((-6,8))

Xavg = np.mean(X, axis=1)
B = X - np.tile(Xavg,(nPoints,1)).T
#File pincipal components (SVD)
U, S, T = np.linalg.svd(B/np.sqrt(nPoints), full_matrices= 0)

ax2 = fig.add_subplot(122)
ax2.plot(X[0,:], X[1,:], '.', Color='k')
ax2.grid()
plt.xlim((-6, 8))
plt.ylim((-6,8))

theta = 2* np.pi *np.arange(0,1, 0.01)

Xstd = U @np.diag(S) @np.array([np.cos(theta), np.sin(theta)])

#1-std confidence interval
ax2.plot(Xavg[0] + Xstd[0,:], Xavg[1]+Xstd[1,:], '-', 'r')
ax2.plot(Xavg[0] + 2*Xstd[0,:], Xavg[1]+ 2*Xstd[1,:], '-', 'r')
ax2.plot(Xavg[0] + 3*Xstd[0,:], Xavg[1]+ 3*Xstd[1,:], '-', 'r')

ax2.plot(np.array([Xavg[0], Xavg[0]+ U[0,0]*S[0]]),
         np.array([Xavg[1], Xavg[1]+ U[1,0]*S[0]]), '-', color = 'cyan')
ax2.plot(np.array([Xavg[0], Xavg[0]+ U[0,1]*S[1]]),
         np.array([Xavg[1], Xavg[1]+ U[1,1]*S[1]]), '-', color = 'cyan')
plt.show()








