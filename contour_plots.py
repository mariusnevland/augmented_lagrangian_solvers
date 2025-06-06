import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Create contour plots for the normal and tangential complementarity functions
# and the regularized versions of these functions, together with illustrations of
# one iteration of NRM and CRM.

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)

transparency = 0.2
cmap = 'Blues'
contour_lvl = 20
contour_trans = 0.12
xlist = np.linspace(-1.7, 1.7, 1000)
ylist = np.linspace(-2.0, 2.0, 1000)
t_n, u_n = np.meshgrid(xlist, ylist)
alpha_matrix = np.ones((1000, 1000))  # Default all to opaque (1.0)
alpha_matrix[:, 1000//2:] = transparency  # Set the right half to be more transparent
# # # NRM together with the normal complementarity function
gap = 0
c_n = 1
Cn = t_n + np.maximum(np.zeros([len(t_n), len(t_n)]), -t_n - c_n*(u_n-gap))
const = np.ones_like(t_n)
fig,ax=plt.subplots(1, 1)
cp3 = ax.contour(t_n, u_n, Cn, levels=contour_lvl, colors='black', zorder=2, alpha=contour_trans)
cp = ax.contour(t_n, u_n, Cn, levels=0, colors='red', zorder=3)
cp2 = ax.pcolormesh(t_n, u_n, const, zorder=1, alpha=alpha_matrix, cmap=cmap, vmin=0.5, vmax=1.7)
ax.set_xlabel(r'$\lambda_n$', fontsize=16)
ax.set_ylabel(r'$[[\mathbf{u}]]_n$', fontsize=16)
points_x = [-0.5, 0.5, 0]
points_y = [-0.5, 0.2, 0.2]
plt.plot(points_x, points_y, 'o', color='black', zorder=4)
plt.plot(points_x, points_y, color='black')
plt.title('NRM, normal direction', fontsize=16)
plt.savefig("contour_normal_nrm.png", dpi=300, bbox_inches="tight")
plt.close()

# # # CRM together with the regularized normal complementarity function
c_n_reg = 1
t_n_prev = -0.5 * np.ones([len(t_n), len(t_n)])
Cn_reg = t_n + np.maximum(np.zeros([len(t_n), len(t_n)]), - t_n_prev - c_n_reg * u_n)
fig1, ax = plt.subplots(1,1)
cp21 = ax.contour(t_n, u_n, Cn_reg, levels=0, colors='red', zorder=3)
trial_line1 = t_n - t_n_prev - c_n_reg * u_n
cp31 = ax.contour(t_n, u_n, trial_line1, levels=0, colors='red', linestyles='dashed', zorder=3)
cp2a = ax.pcolormesh(t_n, u_n, const, zorder=1, alpha=alpha_matrix, cmap=cmap, vmin=0.5, vmax=1.7)
cp3b = ax.contour(t_n, u_n, Cn_reg, levels=contour_lvl, colors='black', zorder=2, alpha=contour_trans)
ax.set_xlabel(r'$\lambda_n$', fontsize=16)
ax.set_ylabel(r'$[[\mathbf{u}]]_n$', fontsize=16)
points_y_reg = [-0.5, 1, 1]
points_x_reg = [-0.5, -0.5 + c_n_reg * points_y_reg[2], 0]
plt.plot(points_x_reg, points_y_reg, 'o', color='black', zorder=4)
plt.plot(points_x_reg, points_y_reg, color='black')
plt.title('CRM, normal direction', fontsize=16)
plt.savefig("contour_normal_crm.png", dpi=300, bbox_inches="tight")
plt.close()

# # # # NRM together with the tangential complementarity function
xlist2 = np.linspace(-2.0, 2.0, 1000)
ylist2 = np.linspace(-2.0, 2.0, 1000)
alpha_matrix2 = np.ones((1000, 1000))
alpha_matrix2[:, 0:250] = transparency
alpha_matrix2[:, 750:] = transparency
t_t, u_t = np.meshgrid(xlist2, ylist2)
t_n = -np.ones([len(t_t), len(t_t)])  # Normal traction equal to some constant
F = 1
c_t = 1
dil_angle = 0  # Dilation angle, in radians.
gap = np.tan(dil_angle)*np.absolute(u_t)
u_n = gap
b = -F * (t_n + c_n*(u_n-gap))  # Set gap=0, which means dilation angle zero.
u_prev_t = np.zeros([len(t_t), len(t_t)])  # u_t at previous time step
Ct = np.maximum(b, np.absolute(t_t + c_t*(u_t-u_prev_t)))*(-t_t) + \
     np.maximum(np.zeros([len(t_t), len(t_t)]), b)*(t_t + c_t*(u_t-u_prev_t))
stick_glide1 = t_t + c_t * u_t - b
stick_glide2 = t_t + c_t * u_t + b
fig2,ax2=plt.subplots(1, 1)
const2 = np.ones_like(t_t)
cpt = ax2.contour(t_t, u_t, Ct, levels=0, colors='red', zorder=3)
cp2c = ax2.pcolormesh(t_t, u_t, const2, zorder=1, alpha=alpha_matrix2, cmap=cmap, vmin=0.5, vmax=1.7)
cp3d = ax2.contour(t_t, u_t, Ct, levels=contour_lvl, colors='black', zorder=2, alpha=contour_trans)
points_x2 = [0.5, 1.6, 1]
points_y2 = [-0.5, -1.4, -1.4]
plt.plot(points_x2, points_y2, 'o', color='black', zorder=4)
plt.plot(points_x2, points_y2, color='black')
ax2.set_xlabel(r'$\lambda_{\tau}$', fontsize=16)
ax2.set_ylabel(r'$[[\dot{\mathbf{u}}]]_{\tau}$', fontsize=16)
plt.title('NRM, tangential direction', fontsize=16)
plt.savefig("contour_tangential_nrm.png", dpi=300, bbox_inches="tight")
plt.close()

# # # # CRM together with the regularized tangential complementarity function
xlist3 = np.linspace(-2.0, 2.0, 1000)
ylist3 = np.linspace(-2.0, 2.0, 1000)
alpha_matrix3 = np.ones((1000, 1000))
alpha_matrix3[:, 0:250] = transparency
alpha_matrix3[:, 750:] = transparency
t_t, u_t = np.meshgrid(xlist3, ylist3)
c_t_reg = 1
t_t_prev = np.zeros([len(t_t), len(t_t)])
t_n_prev_new = -1.0 * np.ones([len(t_t), len(t_t)])
b = -F * t_n_prev_new
fig3, ax3 = plt.subplots(1, 1)
Ct_reg = np.maximum(b, np.absolute(t_t_prev + c_t*(u_t-u_prev_t)))*(-t_t) + \
     np.maximum(np.zeros([len(t_t), len(t_t)]), b)*(t_t_prev + c_t*(u_t-u_prev_t))
cptr = ax3.contour(t_t, u_t, Ct_reg, levels=0, colors='red', zorder=3)
trial_line2 = t_t - t_t_prev - c_t_reg * (u_t - u_prev_t)
cptr4 = ax3.contour(t_t, u_t, trial_line2, levels=0, colors='red', linestyles='dashed', zorder=3)
cp2e = ax3.pcolormesh(t_t, u_t, const2, zorder=1, alpha=alpha_matrix3, cmap=cmap, vmin=0.5, vmax=1.7)
cp3f = ax3.contour(t_t, u_t, Ct_reg, levels=contour_lvl, colors='black', zorder=2, alpha=contour_trans)
stick_glide1reg = t_t_prev + c_t * u_t - b
stick_glide2reg = t_t_prev + c_t * u_t + b
points_y2_reg = [-0.8, 0.5]
points_x2_reg = [0, 0 + c_t_reg * points_y2_reg[1]]
plt.plot(points_x2_reg, points_y2_reg, 'o', color='black', zorder=4)
plt.plot(points_x2_reg, points_y2_reg, color='black')
ax3.set_xlabel(r'$\lambda_{\tau}$', fontsize=16)
ax3.set_ylabel(r'$[[\dot{\mathbf{u}}]]_{\tau}$', fontsize=16)
plt.title('CRM, tangential direction', fontsize=16)
plt.savefig("contour_tangential_crm.png", dpi=300, bbox_inches="tight")
plt.close()