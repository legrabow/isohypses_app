from app import getXYtrennstromlinie


y_plot_init, x_plot_init = getXYtrennstromlinie(Q=100, K=0.0006, grad=0.0017, b=120, winkel=29)
print(y_plot_init)
