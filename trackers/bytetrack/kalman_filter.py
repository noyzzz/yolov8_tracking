# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
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

    The 8-dimensional state space

        x, y, a, h, psi, vx, vy, va, vh, vpsi, accpsi

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    psi is the angle of the robot, vpsi is the angular velocity of the robot

    """

    def __init__(self, image_width, image_height, focal_length, *, params_dict = None):
        if params_dict is not None:
            param_keys = params_dict.keys()
            for key in param_keys:
                setattr(self, key, params_dict[key])
        self.image_width, self.image_height, self.focal_length = image_width, image_height, focal_length
        self.ss_dim, self.dt = 11, 1.0

        # Create Kalman filter model matrices.
        #_motion_mat is the base matrix, should not be directly used in the equations
        self._motion_mat = np.eye(self.ss_dim, self.ss_dim)
        self._motion_mat[4,10] = (self.dt**2) / 2.0
        self._motion_mat[0,10] = (self.dt**2) / 2.0
        self._motion_mat[9,10] = self.dt
        for i in range(6):
            self._motion_mat[i, 5 + i] = self.dt
        self._update_mat = np.eye(5, 11)     

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        # should be tuned
        self._std_weight_angle = 1. / 20
        self._std_weight_angular_velocity = 1. / 16
        self._std_angle_process_noise = 10.
        self._std_angular_vel_process_noise = 0.1
        self._std_angular_acc_process_noise = 0.01

    def calculate_motion_mat(self, mean):
        this_motion_mat = self._motion_mat.copy()
        u1 = mean[0] - self.image_width/2
        robot_yaw_to_pixel_coeff = (u1**2/self.focal_length**2 + 1)*self.focal_length
        this_motion_mat[0, 10] *= robot_yaw_to_pixel_coeff
        this_motion_mat[5, 10] *= robot_yaw_to_pixel_coeff
        return this_motion_mat


    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h, psi) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (10 dimensional) and covariance matrix (10x10
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        # mean_vel = np.zeros_like(mean_pos)
        mean_rest = np.zeros((self.ss_dim - len(mean_pos), ))
        mean = np.r_[mean_pos, mean_rest]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            1e-2, # for psi
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
            1e-1, # for vpsi
            1e-1, # for accpsi
            ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 11 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 11x11 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Process noise standard deviation
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
            self._std_angle_process_noise] 
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
            self._std_angular_vel_process_noise,
            self._std_angular_acc_process_noise]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        #mean = np.dot(self._motion_mat, mean)
        # predict new state: x_pred = F * x
        this_motion_mat = self.calculate_motion_mat(mean)
        mean = np.dot(mean, this_motion_mat.T) # mean is a horizontal vector of shape (1, 10)
        # predict new covariance: P_pred = F * P * F.T + Q
        covariance = np.linalg.multi_dot((
            this_motion_mat, covariance, this_motion_mat.T)) + motion_cov 

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (11 dimensional array).
        covariance : ndarray
            The state's covariance matrix (11x11 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # standard deviation of the measurement noise
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
            1e-2, #FIXME should be tuned
        ]
        # measurement noise covariance matrix (R)
        innovation_cov = np.diag(np.square(std))

        # mean = H * x_hat 
        mean = np.dot(self._update_mat, mean)
        # covariance = H * P * H.T + R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx10 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx10x10 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        # Process noise standard deviation (Q)
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
            self._std_angle_process_noise * np.ones_like(mean[:, 3])]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
            self._std_angular_vel_process_noise * np.ones_like(mean[:, 3]),
            self._std_angular_acc_process_noise * np.ones_like(mean[:, 3]),]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T

        for i in range(len(mean)):
            this_motion_cov = np.diag(sqr[i])
            this_motion_mat = self.calculate_motion_mat(mean[i])
            mean[i] = np.dot(mean[i], this_motion_mat.T) 
            covariance[i] = np.linalg.multi_dot((
                this_motion_mat, covariance[i], this_motion_mat.T)) + this_motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement, measurement_mask):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 5 dimensional measurement vector (x, y, a, h, psi), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        measurement_mask : ndarray
            A 5 dimensional boolean mask indicating whether the measurement is available for each element in the measurement vector.

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
        # if not np.all(measurement_mask):
        #     new_mean[9] = (measurement[4] - mean[4])/self.dt #FIXME this is a hack, should be fixed
        # print("new mean 9                                                           :     ", new_mean[9])
        # new_mean[4] = measurement[4] #FIXME this is a hack, should be fixed
        I_KH = np.eye(kalman_gain.shape[0]) - np.dot(kalman_gain, self._update_mat)
        I_KHT = I_KH.T
        new_covariance = np.dot(I_KH, covariance) #simplified covariance update equation
        # new_covariance = covariance - np.linalg.multi_dot((
        #     kalman_gain, projected_cov, kalman_gain.T))


        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
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
