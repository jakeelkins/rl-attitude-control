'''
Jake Elkins built this boi

spacecraft attitude control simulator, built in the OpenAI gym format for easy interface with popular RL libraries.

this one is continuous control.
'''

import gym
from gym import spaces, logger

import numpy as np
from numba import jit


class AttitudeControlEnv(gym.Env):

    # ---- toolbox ----
    # all the typical functions needed for quaternion math stuff.
    # set as staticmethod so numba will compile them for speeds

    @staticmethod
    @jit(nopython = True)
    def _randomAxisAngle(min_angle, max_angle):
        '''
        generates random axis angle slew in form of
        (x, y, z, angle)
        angle in radians
        '''
        # generate phi and theta. then convert from spherical --> cartesian
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)

        theta = np.arccos(costheta) #takes care of quadrant issue
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # now snag our angle for the slew from the function bounds
        angle = np.random.uniform(min_angle, max_angle)*np.pi/180

        return np.array([x, y, z, angle])

    @staticmethod
    @jit(nopython = True)
    def _axisAngleToQuat(axis_angle):
        '''
        requires np array of axis angle in [x, y, z, angle].
        angle in rad. returns quaternion in [qv q4]
        where q4 is real part and qv = q1 q2 q3
        '''
        # quat is [ehat_x sin(phi/2) ehat_y sin(phi/2) ehat_z sin(phi/2) cos(phi/2)]
        ehat_x, ehat_y, ehat_z, ang = axis_angle
        
        q1 = ehat_x*np.sin(ang/2)
        q2 = ehat_y*np.sin(ang/2)
        q3 = ehat_z*np.sin(ang/2)
        q4 = np.cos(ang/2)
        
        return np.array([q1, q2, q3, q4])

    @staticmethod
    @jit(nopython = True)
    def _getOmegaDot(omega, torque, I):
        Imat = np.diag(I)
        Imat_inv = np.diag(np.divide(1, I))
        
        Iw = np.dot(Imat, omega)
        arg = torque + (np.cross(Iw, omega))
        
        omega_dot = np.dot(Imat_inv, arg)
        return omega_dot

    @staticmethod
    @jit(nopython = True)
    def _getQDot(omega, quat):
        R = np.array([[1-(2*quat[1]**2)-(2*quat[2]**2), 2*quat[0]*quat[1] - 2*quat[2]*quat[3], 2*quat[0]*quat[2] + 2*quat[1]*quat[3]],
          [2*quat[0]*quat[1] + 2*quat[2]*quat[3], 1-(2*quat[0]**2)-(2*quat[2]**2), 2*quat[1]*quat[2] - 2*quat[0]*quat[3]],
          [2*quat[0]*quat[2] - 2*quat[1]*quat[3], 2*quat[1]*quat[2] + 2*quat[0]*quat[3], 1-(2*quat[0]**2)-(2*quat[1]**2)]
         ])
        omega_in = np.dot(R, omega)
        
        s = np.dot(-omega_in, quat[:3])
        v = np.multiply(quat[-1], omega_in) + np.cross(omega_in, quat[:3])
        
        qdot = np.append(v, s)
        
        return qdot

    @staticmethod
    @jit(nopython = True)
    def _RK4_omega_quat(_getOmegaDot, _getQDot, h, omega0, q0, torque, I):
        # two matrices containing what we wanna do. k_w (for omega) and k_q (for quat)
        k_w = np.zeros((3,4))
        k_q = np.zeros((4,4))

        k_q[:, 0] = h * _getQDot(omega0, q0)
        k_w[:, 0] = h * _getOmegaDot(omega0, torque, I)

        k_q[:, 1] = h * _getQDot(omega0 + (0.5*k_w[:,0]), q0+(0.5*k_q[:,0]))
        k_w[:, 1] = h * _getOmegaDot(omega0 + (0.5*k_w[:,0]), torque, I)

        k_q[:, 2] = h * _getQDot(omega0 + (0.5*k_w[:,1]), q0+(0.5*k_q[:,1]))
        k_w[:, 2] = h * _getOmegaDot(omega0 + (0.5*k_w[:,1]), torque, I)

        k_q[:, 3] = h * _getQDot(omega0 + (k_w[:,2]), q0+(k_q[:,2]))
        k_w[:, 3] = h * _getOmegaDot(omega0 + (k_w[:,2]), torque, I)

        q0 = q0 + (1/6)*np.dot(k_q, np.array([1.,2.,2.,1.]))
        omega0 = omega0 + (1/6)*np.dot(k_w, np.array([1.,2.,2.,1.]))

        q0 = np.divide(q0, np.linalg.norm(q0))
        return omega0, q0


    # -------end toolbox, start actual env-------




    def __init__(self, torque_scale = 0.5, steps = 500):

        # multiply this to scale the torques. we'll loop thru training to change this
        self.torque_scale = torque_scale

        self.frameskip_num = 20 # controls agent fideltiy. number of times to integrate between actions

        self.max_angle_slew = 150.    # maximum angle for goal slew generation, in degrees
        self.min_angle_slew = 30.    # minimum angle for goal slew generation, in degrees

        self.I = np.array([0.872, 0.115, 0.797])    # rotational inertia tensor (assuming symmetric, this is the diagonal)

        self.h = 1/240  # integrator fidelity (in seconds)

        self.q_initial = np.array([0., 0., 0., 1.])  # initial quaternion (in world frame--always starts aligned with world frame)

        high = np.array([1.0, 1.0, 1.0, 1.0, 10, 10, 10, 10, 1.0, 1.0, 1.0])

        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.nsteps = None

        self.alpha = np.array([-1., -1., -1., 1.])

        # ---thresholds for episode-----

        self.max_episode_steps = steps

        self.max_ang_velo = 0.5

        # --------------------------

        self.steps_beyond_done = None


    def step(self, action):

        self.nsteps += 1

        action = np.clip(action, -1.0, 1.0)

        torque = self.torque_scale * action

        # step the env once with the torque applied:
        self.omega, self.q_error = self._RK4_omega_quat(self._getOmegaDot, self._getQDot, self.h, self.omega, self.q_error, torque, self.I)

        # empty the torques
        torque = np.array([0., 0., 0.])

        # propagate free rotation forward
        for _ in range(self.frameskip_num):
            self.omega, self.q_error = self._RK4_omega_quat(self._getOmegaDot, self._getQDot, self.h, self.omega, self.q_error, torque, self.I)


        self.state = (self.q_error[3], self.q_error[0], self.q_error[1], self.q_error[2], self.q_error_dot[3], self.q_error_dot[0], self.q_error_dot[1], self.q_error_dot[2], self.omega[0], self.omega[1], self.omega[2])
        
        curr_angle = 2*np.arccos(self.q_error[3])
        omega_magnitude = np.linalg.norm(self.omega)

        done = omega_magnitude > self.max_ang_velo \
                or self.nsteps >= self.max_episode_steps

        done = bool(done)

        qs = self.q_error[3]

        exceed = bool(omega_magnitude > self.max_ang_velo)
        
        if (self.q_error[3]) >= 0.99999762:
            self.rew_mode = 'delta'

        #--------REWARD---------
        if not done:
            arg = curr_angle/(0.14*2*np.pi)
            if qs > self.qs_prev:
                reward = np.exp(-arg) - 0.5*(np.linalg.norm(action))
            else:
                reward = np.exp(-arg) - 0.5*(np.linalg.norm(action)) - 1.

            if (self.q_error[3]) >= 0.99999762:
                reward += 9.

        elif self.steps_beyond_done is None:
            # epsiode just ended
            self.steps_beyond_done = 0
            if exceed:
                reward = -25.
            elif (self.q_error[3]) >= 0.99999762:
                reward = 50.
            else:
                reward = 0.
            
        
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
        self.qs_prev = qs
        return np.array(self.state), reward, done, {}





    def reset(self, init_quat=None):

        self.nsteps = 0

        self.rew_mode = 'grad'


        # generate goal quaternion and set it as initial goal as well. this will be current-->goal orn.
        if init_quat is None:
            goal_aa = self._randomAxisAngle(self.min_angle_slew, self.max_angle_slew)
            self.q_error_0 = self._axisAngleToQuat(goal_aa)
            self.q_error = self.q_error_0
        else:
            self.q_error_0 = init_quat
            if len(self.q_error_0) != 4:
                raise ValueError('invalid quat given')
            self.q_error = self.q_error_0

        self.q_error_0_val = np.dot(self.alpha, np.square(self.q_error_0))

        self.q_error_prev = self.q_error_0_val

        self.initial_angle = 2*np.arccos(self.q_error[3])

        # initial rate of change of error quat for state vector
        self.q_error_dot = np.array([0., 0., 0., 0.])

        # q ref is the reference quaternion, or quat from initial-->current orn. 
        self.q_ref = self.q_initial

        # initial rate of change of ref quat for state vector
        self.q_ref_dot = np.array([0., 0., 0., 0.])

        # start from rest (this is angular velocity)
        self.omega = np.array([0., 0., 0.])

        self.state = (self.q_error[3], self.q_error[0], self.q_error[1], self.q_error[2], self.q_error_dot[3], self.q_error_dot[0], self.q_error_dot[1], self.q_error_dot[2], self.omega[0], self.omega[1], self.omega[2])
        
        self.steps_beyond_done = None

        qs = self.q_error[3]
        self.qs_prev = qs

        return np.array(self.state)