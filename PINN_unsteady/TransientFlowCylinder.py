import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import shutil
import pickle
import scipy.io
import random

# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CPU:-1; GPU0: 1; GPU1: 0;

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class PINN_laminar_flow:
    # Initialize the class
    def __init__(self, Collo, IC, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):

        # Count for callback function
        self.count=0

        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.rho = 1.0
        self.mu = 0.005

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        self.t_c = Collo[:, 2:3]

        self.x_IC = IC[:, 0:1]
        self.y_IC = IC[:, 1:2]
        self.t_IC = IC[:, 2:3]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.t_INLET = INLET[:, 2:3]
        self.u_INLET = INLET[:, 3:4]
        self.v_INLET = INLET[:, 4:5]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]
        self.t_OUTLET = OUTLET[:, 2:3]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]
        self.t_WALL = WALL[:, 2:3]

        # Define layers
        self.uv_layers = uv_layers

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])

        self.x_IC_tf = tf.placeholder(tf.float32, shape=[None, self.x_IC.shape[1]])
        self.y_IC_tf = tf.placeholder(tf.float32, shape=[None, self.y_IC.shape[1]])
        self.t_IC_tf = tf.placeholder(tf.float32, shape=[None, self.t_IC.shape[1]])

        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])
        self.t_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.t_WALL.shape[1]])

        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])
        self.t_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.t_OUTLET.shape[1]])

        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.t_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.t_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf, self.t_tf)
        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12, \
            self.f_pred_p = self.net_f(self.x_c_tf, self.y_c_tf, self.t_c_tf)
        self.u_IC_pred, self.v_IC_pred, self.p_IC_pred, _, _, _ = self.net_uv(self.x_IC_tf, self.y_IC_tf, self.t_IC_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf, self.t_WALL_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf, self.t_INLET_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf, self.t_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v))\
                      + tf.reduce_mean(tf.square(self.f_pred_s11))\
                      + tf.reduce_mean(tf.square(self.f_pred_s22))\
                      + tf.reduce_mean(tf.square(self.f_pred_s12))\
                      + tf.reduce_mean(tf.square(self.f_pred_p))
                      # + tf.reduce_mean(tf.square(self.f_pred_mass))\
        self.loss_IC = tf.reduce_mean(tf.square(self.u_IC_pred)) \
                       + tf.reduce_mean(tf.square(self.v_IC_pred))\
                       + tf.reduce_mean(tf.square(self.p_IC_pred))
        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) \
                       + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred-self.u_INLET_tf)) \
                         + tf.reduce_mean(tf.square(self.v_INLET_pred))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred-0.0))

        # Coefficients could affect the accuracy and convergence of the result
        self.loss = self.loss_f + 5*self.loss_WALL + 5*self.loss_INLET + self.loss_OUTLET \
                    + self.loss_IC

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1*np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def save_NN(self, fileDir):

        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)

        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num])
                b = tf.Variable(uv_biases[num])
                weights.append(W)
                biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y, t):
        psips = self.neural_net(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        psi = psips[:,0:1]
        p = psips[:,1:2]
        s11 = psips[:, 2:3]
        s22 = psips[:, 3:4]
        s12 = psips[:, 4:5]
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p, s11, s22, s12

    def net_f(self, x, y, t):

        rho=self.rho
        mu=self.mu
        u, v, p, s11, s22, s12 = self.net_uv(x, y, t)

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]

        # Plane stress problem
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]

        # f_u:=Sxx_x+Sxy_y
        f_u = rho*u_t + rho*(u*u_x + v*u_y) - s11_1 - s12_2
        f_v = rho*v_t + rho*(u*v_x + v*v_y) - s12_1 - s22_2

        # f_mass = u_x+u_y

        f_s11 = -p + 2*mu*u_x - s11
        f_s22 = -p + 2*mu*v_y - s22
        f_s12 = mu*(u_y+v_x) - s12

        f_p = p + (s11+s22)/2

        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def callback(self, loss):
        self.count = self.count+1
        print('{} th iterations, Loss: {}'.format(self.count, loss))


    def train(self, iter, learning_rate):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.t_WALL_tf: self.t_WALL,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.t_INLET_tf: self.t_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.t_OUTLET_tf: self.t_OUTLET,
                   self.learning_rate: learning_rate}

        loss_WALL = []
        loss_f = []
        loss_IC = []
        loss = []
        loss_INLET = []
        loss_OUTLET = []

        for it in range(iter):

            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' %
                      (it, loss_value))

            loss_WALL.append(self.sess.run(self.loss_WALL, tf_dict))
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            loss.append(self.sess.run(self.loss, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_OUTLET, tf_dict))

        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss

    def train_bfgs(self):

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.t_WALL_tf: self.t_WALL,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.t_INLET_tf: self.t_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.t_OUTLET_tf: self.t_OUTLET}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        return u_star, v_star, p_star

    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL, self.t_WALL_tf: self.t_WALL,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET, self.t_INLET_tf: self.t_INLET, self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
                   self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET, self.t_OUTLET_tf: self.t_OUTLET}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_WALL = self.sess.run(self.loss_WALL, tf_dict)
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)
        loss_IC = self.sess.run(self.loss_IC, tf_dict)

        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss_IC, loss

def preprocess(dir):
    # Directory of reference solution
    data = scipy.io.loadmat(dir)

    X = data['x']
    Y = data['y']
    Exact_u = data['u']
    Exact_v = data['v']
    Exact_p = data['p']

    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    u_star = Exact_u.flatten()[:, None]
    v_star = Exact_v.flatten()[:, None]
    p_star = Exact_p.flatten()[:, None]

    return x_star, y_star, u_star, v_star, p_star

def postProcess(xmin, xmax, ymin, ymax, field, s=2, num=0):
    ''' num: Number of time step
    '''
    [x_pred, y_pred, _, u_pred, v_pred, p_pred] = field

    # fig, axs = plt.subplots(2)
    fig, ax = plt.subplots(nrows=3, figsize=(6, 8))
    # fig.subplots_adjust(hspace=0.2, wspace=0.2)

    cf = ax[0].scatter(x_pred, y_pred, c=u_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin=0, vmax=1.4)
    ax[0].axis('square')
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title('u predict')
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin= -0.7,vmax=0.7)
    ax[1].axis('square')
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title('v predict')
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    cf = ax[2].scatter(x_pred, y_pred, c=p_pred, alpha=0.7, edgecolors='none', cmap='rainbow', marker='o', s=s, vmin=-0.2, vmax=3)
    ax[2].axis('square')
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title('p predict')
    fig.colorbar(cf, ax=ax[2], fraction=0.046, pad=0.04)

    plt.suptitle('Time: '+str(num*0.01)+'s', fontsize=16)

    plt.savefig('./output/uvp_comparison_'+str(num)+'.png',dpi=150)
    plt.close('all')


def DelSrcPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    # Delete collocation point within cylinder
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]

def CartGrid(xmin, xmax, ymin, ymax, tmin, tmax, num_x, num_y, num_t):
    # num_x, num_y: number per edge
    # num_t: number time step

    x = np.linspace(xmin, xmax, num=num_x)
    y = np.linspace(ymin, ymax, num=num_y)
    xx, yy = np.meshgrid(x, y)
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, yyy, ttt = np.meshgrid(x, y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt

def GenCirclePT(xc, yc, r, tmin, tmax, num_r, num_t):
    # Generate collocation points at cylinder uniformly
    theta = np.linspace(0.0, np.pi*2.0, num_r)
    x = np.multiply(r, np.cos(theta)) + xc
    y = np.multiply(r, np.sin(theta)) + yc
    t = np.linspace(tmin, tmax, num_t)
    xx, tt = np.meshgrid(x, t)
    yy, _ = np.meshgrid(y, t)
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    tt = tt.flatten()[:, None]

    return xx, yy, tt

if __name__ == "__main__":

    # Domain bounds
    xmax = 1.1
    tmax = 0.5
    lb = np.array([0, 0, 0])
    ub = np.array([xmax, 0.41, tmax])

    # Network configuration
    uv_layers = [3] + 7*[50] + [5]

    # Sample collocation points with Cartesian grid, another option is to sample with LHS, like in steady case
    x_IC, y_IC, t_IC = CartGrid(xmin=0, xmax=xmax,
                                ymin=0, ymax=0.41,
                                tmin=0, tmax=0,
                                num_x=81, num_y=41, num_t=1)
    IC = np.concatenate((x_IC, y_IC, t_IC), 1)
    IC = DelSrcPT(IC, xc=0.2, yc=0.2, r=0.05)

    x_upb, y_upb, t_upb = CartGrid(xmin=0, xmax=xmax,
                                   ymin=0.41, ymax=0.41,
                                   tmin=0, tmax=tmax,
                                   num_x=81, num_y=1, num_t=41)

    x_lwb, y_lwb, t_lwb = CartGrid(xmin=0, xmax=xmax,
                                   ymin=0, ymax=0,
                                   tmin=0, tmax=tmax,
                                   num_x=81, num_y=1, num_t=41)
    wall_up = np.concatenate((x_upb, y_upb, t_upb), 1)
    wall_lw = np.concatenate((x_lwb, y_lwb, t_lwb), 1)

    U_max = 0.5
    T = tmax*2  # Period
    x_inb, y_inb, t_inb = CartGrid(xmin=0, xmax=0,
                                   ymin=0, ymax=0.41,
                                   tmin=0, tmax=tmax,
                                   num_x=1, num_y=61, num_t=61)
    u_inb = 4*U_max*y_inb*(0.41-y_inb)/(0.41**2)*(np.sin(2*3.1416*t_inb/T+3*3.1416/2)+1.0)
    v_inb = np.zeros_like(x_inb)
    INB = np.concatenate((x_inb, y_inb, t_inb, u_inb, v_inb), 1)

    # Visualize the velocity profile of the inlet boundary
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(INB[:, 1:2], INB[:, 2:3], INB[:, 3:4], marker='o', alpha=0.1, s=2, color='blue')
    ax.set_xlabel('y axis')
    ax.set_ylabel('t axis')
    ax.set_zlabel('v axis')
    plt.show()

    x_outb, y_outb, t_outb = CartGrid(xmin=1.1, xmax=1.1,
                                      ymin=0, ymax=0.41,
                                      tmin=0, tmax=tmax,
                                      num_x=1, num_y=81, num_t=41)
    OUTB = np.concatenate((x_outb, y_outb, t_outb), 1)

    # Cylinder surface
    r = 0.05
    x_surf, y_surf, t_surf= GenCirclePT(xc=0.2, yc=0.2, r=r, tmin=0, tmax=tmax, num_r=81, num_t=51)
    HOLE = np.concatenate((x_surf, y_surf, t_surf), 1)

    WALL = np.concatenate((HOLE, wall_up, wall_lw), 0)

    # Collocation point on domain, with refinement near the wall
    XY_c = lb + (ub - lb) * lhs(3, 80000)
    XY_c_refine = [0.0, 0.0, 0.0] + [0.4, 0.4, tmax] * lhs(3, 15000)
    XY_c_lw = [0.0, 0.0, 0.0] + [1.1, 0.02, tmax] * lhs(3, 3000)
    XY_c_up = [0.0, 0.39, 0.0] + [1.1, 0.02, tmax] * lhs(3, 3000)
    XY_c = np.concatenate((XY_c, XY_c_refine, XY_c_lw, XY_c_up), 0)
    XY_c = DelSrcPT(XY_c, xc=0.2, yc=0.2, r=0.05)
    XY_c = np.concatenate((XY_c, WALL, OUTB, INB[:,0:3]), 0)

    print(XY_c.shape)

    # Visualize ALL the training points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(XY_c[:,0:1], XY_c[:,1:2], XY_c[:,2:3], marker='o', alpha=0.1, s=2, color='blue')
    # ax.scatter(IC[:, 0:1], IC[:, 1:2], IC[:, 2:3], marker='o', alpha=0.1, s=2, color='green')
    # ax.scatter(WALL[:, 0:1], WALL[:, 1:2], WALL[:, 2:3], marker='o', alpha=0.1, s=2, color='orange')
    # ax.scatter(INB[:, 0:1], INB[:, 1:2], INB[:, 2:3], marker='o', alpha=0.1, s=2, color='yellow')
    # ax.scatter(OUTB[:, 0:1], OUTB[:, 1:2], OUTB[:, 2:3], marker='o', alpha=0.1, s=2, color='blue')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('T axis')
    # plt.show()

    with tf.device('/device:GPU:1'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Train from scratch
        model = PINN_laminar_flow(XY_c, IC, INB, OUTB, WALL, uv_layers, lb, ub)

        # Load trained model for inference
        # model = PINN_laminar_flow(XY_c, IC, INB, OUTB, WALL, uv_layers, lb, ub, ExistModel=1, uvDir='uvNN.pickle')

        start_time = time.time()
        loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(iter=5000, learning_rate=5e-4)
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))

        # Print loss components
        model.getloss()

        # Save model for later use
        model.save_NN('uvNN.pickle')

        # Plot the pressure history on the leading point of cylinder
        t_front = np.linspace(0, 0.5, 100)
        x_front = np.zeros_like(t_front)
        x_front.fill(0.15)
        y_front = np.zeros_like(t_front)
        y_front.fill(0.20)
        t_front = t_front.flatten()[:, None]
        x_front = x_front.flatten()[:, None]
        y_front = y_front.flatten()[:, None]

        u_pred, v_pred, p_pred = model.predict(x_front, y_front, t_front)
        plt.plot(t_front, p_pred)
        plt.show()

        # Output u, v, p at each time step
        N_t=51
        x_star = np.linspace(0, 1.1, 401)
        y_star = np.linspace(0, 0.41, 161)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        dst = ((x_star-0.2)**2+(y_star-0.2)**2)**0.5
        x_star = x_star[dst >= 0.05]
        y_star = y_star[dst >= 0.05]
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        for i in range(N_t):
            t_star = np.zeros((x_star.size, 1))
            t_star.fill(i*0.5/(N_t-1))

            u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
            field = [x_star, y_star, t_star, u_pred, v_pred, p_pred]
            amp_pred = (u_pred**2 + v_pred**2)**0.5

            postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field=field, s=2, num=i)

