# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""

import rospy
#import ros float array type
from std_msgs.msg import Float32MultiArray
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 10-dimensional state space

        x, y, a, h, d vx, vy, va, vh, vd

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self, image_width, image_height, focal_length, *, params_dict = None):
        if params_dict is not None:
            param_keys = params_dict.keys()
            for key in param_keys:
                setattr(self, key, params_dict[key])
        self.image_width, self.image_height, self.focal_length = image_width, image_height, focal_length
        self.ndim, self.dt = 5, 1.

        # Create Kalman filter model matrices.
        #_motion_mat is the base matrix, should not be directly used in the equations
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)     
        self.control_mat = np.zeros((2 * self.ndim, 1))

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        
        self._q1 = 1./20
        self._q4 = 1./160
        self._r1 = 1/20
        self._d_std = 1/80

        self._r4 = 1./160 #to initiate P matrix ONLY

        params_array = Float32MultiArray()
        params_array.data = [self._q1, self._q4, self._r1, self._r4]
        

    def calculate_motion_mat(self, mean):
        
        # u1 = mean[0] - self.image_width/2
        # robot_yaw_to_pixel_coeff = (u1**2/self.focal_length**2 + 1)*self.focal_length
        # self._motion_mat[0, 2*self.ndim-1] = robot_yaw_to_pixel_coeff*self.dt
        # return self._motion_mat\
        return self._motion_mat

    def calculate_yaw_control_mat(self, mean):
        u1 = mean[0] - self.image_width/2
        robot_yaw_to_pixel_coeff = (u1**2/self.focal_length**2 + 1)*self.focal_length
        self.control_mat[0, 0] = robot_yaw_to_pixel_coeff*self.dt
        return self.control_mat
    
    def calculate_depth_control_mat(self, mean, control_signal):
        if control_signal[2] == 0:
            return np.zeros((2 * self.ndim, 1))
        u1 = mean[0] - self.image_width/2
        v1 = mean[1] - self.image_height/2
        bottom_y = v1 + mean[3]/2 #bottom right corner y wrt image center
        u_coeff = u1*np.sqrt(u1**2 + self.focal_length**2)/(self.focal_length * control_signal[2])
        v_coeff = v1*np.sqrt(v1**2 + self.focal_length**2)/(self.focal_length * control_signal[2])
        h_coeff = (bottom_y*np.sqrt(bottom_y**2 + self.focal_length**2) - v1*np.sqrt(v1**2 + self.focal_length**2))\
                   /(self.focal_length * control_signal[2])  #FIXME: this is wrong; v1 is not the top left corner
        depth_coeff = -1 * np.sqrt(self.focal_length**2/(u1**2 + self.focal_length**2))
        depth_control_mat = np.zeros((2 * self.ndim, 1))
        depth_control_mat[0, 0] = u_coeff
        depth_control_mat[1, 0] = v_coeff
        depth_control_mat[3, 0] = h_coeff
        depth_control_mat[4, 0] = depth_coeff
        return depth_control_mat


    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h, d) with center position (x, y),
            aspect ratio a, and height h, depth d

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            self._r1  * measurement[3],
            self._r1  * measurement[3],
            1e-2,
            self._r1  * measurement[3],
            self._d_std * measurement[4],  
            
            self._r4  * measurement[3],
            self._r4  * measurement[3],
            1e-5,
            self._r4  * measurement[3],
            self._r4 * measurement[4]
            ]
        
        covariance = np.diag(np.square(std))*10
        return mean, covariance

    # def predict(self, mean, covariance, input_signal):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 10x10 dimensional covariance matrix of the object state at the
            previous time step.
        input_signal : ndarray
            The 1 dimensional control signal vector that contains the robot's
            yaw velocity.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Process noise standard deviation
        std_pos = [
            self._q1 * mean[3],
            self._q1 * mean[3] ,
            1e-2,
            self._q1 * mean[3],
            self._d_std * mean[4]] 
        std_vel = [
            self._q4* mean[3] ,
            self._q4 * mean[3],
            1e-5,
            self._q4 * mean[3],
            self._q4 * mean[4]]
                
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # predict new state: x_pred = F * x + B * u
        this_motion_mat = self.calculate_motion_mat(mean) # F matrix
        this_control_mat = self.calculate_yaw_control_mat(mean) # B matrix
        mean = np.dot(mean, this_motion_mat.T) + np.dot(input_signal, this_control_mat.T)
        # predict new covariance: P_pred = F * P * F.T + Q
        covariance = np.linalg.multi_dot((
            this_motion_mat, covariance, this_motion_mat.T)) + motion_cov 

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (10 dimensional array).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # standard deviation of the measurement noise
        # std = [
            # self._std_weight_position ,
            # self._std_weight_position ,
            # 1e-1,
            # self._std_weight_position 
        # ]
        # std = [
        #     np.abs(mean[0] - self.image_width/2)/30,
        #     np.abs(mean[1] - self.image_height/2)/30,
        #     1e-1,
        #     1]
        std = [
            self._r1 * mean[3],
            self._r1 * mean[3],
            1e-1,
            self._r1 * mean[3],
            self._d_std * mean[3]]
        
        innovation_cov = np.diag(np.square(std))

        # mean = H * x_hat 
        mean = np.dot(self._update_mat, mean)
        # covariance = H * P * H.T + R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance, control_signal):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The 10 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx10x10 dimensional covariance matrics of the object states at the
            previous time step.
        control_signal : ndarray
            The Nx2 dimensional control signal vector that contains the robot's
            yaw velocity, forward tranlational velocity.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        use_control_signal = False
        #define x_dis_to_end, y_dis_to_end, x_dis_to_start, y_dis_to_start
        # x_dis_to_end = np.abs(self.image_width - mean[:, 0])
        # x_dis_to_start = np.abs(mean[:, 0])
        #for each track find the maximum distance to the end of the image
        # x_dis_min = np.minimum(x_dis_to_end, x_dis_to_start)*10.0/self.image_width


        std_pos = [
            self._q1 * (mean[:, 3]),
            self._q1 * (mean[:, 3]) ,
            1e-2 * (mean[:, 3]),
            self._q1 * (mean[:, 3]),
            self._d_std * (mean[:, 3])*500]
        std_vel = [
            self._q4 * (mean[:, 3]) ,
            self._q4 * (mean[:, 3]) ,
            1e-5 * (mean[:, 3]),
            self._q4 * (mean[:, 3]),
            self._q4 * (mean[:, 3])*500]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        for i in range(len(mean)):
            this_motion_cov = np.diag(sqr[i])
            this_motion_mat = self.calculate_motion_mat(mean[i])
            yaw_control_mat = self.calculate_yaw_control_mat(mean[i])                
            mean_rot_applied = np.dot(control_signal[i][0], yaw_control_mat.T)[0]
            depth_control_mat = self.calculate_depth_control_mat(mean[i], control_signal[i])
            mean_trans_applied = np.dot(control_signal[i][1], depth_control_mat.T)[0]
            if use_control_signal:
                mean[i] = np.dot(mean[i], this_motion_mat.T) + mean_rot_applied + mean_trans_applied
                covariance[i] = np.linalg.multi_dot((
                    this_motion_mat, covariance[i], this_motion_mat.T)) + this_motion_cov
            else:
                mean[i] = np.dot(mean[i], this_motion_mat.T)
                covariance[i] = np.linalg.multi_dot((
                    this_motion_mat, covariance[i], this_motion_mat.T)) + this_motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (10 dimensional).
        covariance : ndarray
            The state's covariance matrix (10x10 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, a, h, d), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        measurement_mask : ndarray
            A 5 dimensional boolean mask indicating whether the measurement is available for each element in the measurement vector.
            [x, y, a, h, d]

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        # if not np.all(measurement_mask):
        #     mean[9] = (measurement[4] - mean[4])/self.dt
        #     mean[4] = measurement[4]
        #     print(measurement[4])
        #     return mean, covariance
        projected_mean, projected_cov = self.project(mean, covariance)
        PHT = np.dot(covariance, self._update_mat.T)
        SI = np.linalg.inv(projected_cov)
        kalman_gain = np.dot(PHT, SI)

        # chol_factor, lower = scipy.linalg.cho_factor(
        #     projected_cov, lower=True, check_finite=False)
        # kalman_gain = scipy.linalg.cho_solve(
        #     (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
        #     check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        I_KH = np.eye(kalman_gain.shape[0]) - np.dot(kalman_gain, self._update_mat)
        I_KHT = I_KH.T
        new_covariance = np.dot(I_KH, covariance) #simplified covariance update equation
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain, projected_cov, kalman_gain.T))


        return new_mean, new_covariance
    
    def update_dummy(self, mean, covariance, yaw_dot): #FIXME: this is a dummy update function
        mean[4:] = 1/(5+1000*yaw_dot) * mean[4:]
        mean[5] = 1/(10+1000*yaw_dot) * mean[5]
        mean[6] = 0
        return mean, covariance


    # def gating_distance(self, mean, covariance, measurements,
    #                     only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        # only consider x, y, a, h
        mean, covariance = mean[:4], covariance[:4, :4]
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
