'''
@file kalman_filter.gyro
@author underflow101 (ikarus125@gmail.com)
@brief Support Kalman Filter for accelerometer & gyroscope sensor
@version 0.1.0
@date 2021-12-28

@copyright Copyright (c) 2021 underflow101
'''

from time import time

class KalmanFilterGyro:
    def __init__(self,
                 lower_x=-2150, upper_x=2210,
                 lower_y=-2150, upper_y=2210,
                 lower_z=-2150, upper_z=2550):
        super(KalmanFilterGyro, self).__init__()
        
        self.R_angle = 0.3
        self.Q_angle = 0.01
        self.Q_gyro = 0.04
        
        self.sensor_limit = {
            'lower_x': lower_x,
            'upper_x': upper_x,
            'lower_y': lower_y,
            'upper_y': upper_y,
            'lower_z': lower_z,
            'upper_z': upper_z
        }
        
        self.prev_time = 0
        self.curr_time = 0
        
        self.avg_idx = 0
        self.avg_size = 5
        
        self.x_avg_list = [0] * self.avg_size
        self.y_avg_list = [0] * self.avg_size
        self.z_avg_list = [0] * self.avg_size
        
        self.x_calc = 0
        self.y_calc = 0
        self.z_calc = 1800
        
        self.axis = _Axis()
        self.sensor = _SensorOutput()
        
    def get(self):
        return self.axis.coord.x, self.axis.coord.y, self.axis.coord.z

    def init_filter(self):
        self._init_data(self.axis.x)
        self._init_data(self.axis.y)
        self._init_data(self.axis.z)
        
        return
    
    def execute(self,
                x_accel=0, y_accel=0, z_accel=0,
                x_gyro=0, y_gyro=0, z_gyro=0):
        self.curr_time = int(time() * 1000)
        
        self.sensor.value.x_accel = x_accel
        self.sensor.value.y_accel = y_accel
        self.sensor.value.z_accel = z_accel
        
        self.sensor.value.x_gyro = x_gyro
        self.sensor.value.y_gyro = y_gyro
        self.sensor.value.z_gyro = z_gyro
        
        if self.prev_time > 0:
            gx1, gy1, gz1 = 0, 0, 0
            gx2, gy2, gz2 = 0, 0, 0
            
            loop_time = self.curr_time - self.prev_time
            
            gx2 = self._angle_in_degree(self.sensor_limit['lower_x'], self.sensor_limit['upper_x'], self.sensor.value.x_gyro)
            gy2 = self._angle_in_degree(self.sensor_limit['lower_y'], self.sensor_limit['upper_y'], self.sensor.value.y_gyro)
            gz2 = self._angle_in_degree(self.sensor_limit['lower_z'], self.sensor_limit['upper_z'], self.sensor.value.z_gyro)
            
            self._predict(self.axis.x, gx2, loop_time)
            self._predict(self.axis.y, gy2, loop_time)
            self._predict(self.axis.z, gz2, loop_time)
            
            gx1 = self._update(self.axis.x, self.sensor.value.x_accel) / 10
            gy1 = self._update(self.axis.y, self.sensor.value.y_accel) / 10
            gz1 = self._update(self.axis.z, self.sensor.value.z_accel) / 10
            
            if self.avg_idx < self.avg_size:
                self.x_avg_list[self.avg_idx] = gx1
                self.y_avg_list[self.avg_idx] = gy1
                self.z_avg_list[self.avg_idx] = gz1
                
                if self.avg_idx == (self.avg_size -1):
                    sum_x, sum_y, sum_z = 0, 0, 0
                    
                    for i in range(1, self.avg_size+1):
                        sum_x += self.x_avg_list[i]
                        sum_y += self.y_avg_list[i]
                        sum_z += self.z_avg_list[i]
                        
                    self.x_calc -= sum_x / (self.avg_size - 1)
                    self.y_calc -= sum_y / (self.avg_size - 1)
                    self.z_calc -= (sum_z / (self.avg_size - 1) - self.z_calc)
                    
                self.avg_idx += 1
            else:
                gx1 += self.x_calc
                gy1 += self.y_calc
        
        self.axis.coord.x = gx1
        self.axis.coord.y = gy1
        self.axis.coord.z = gz1
        
        self.prev_time = self.curr_time
        
        return
        
    def _predict(self, kalman, dot_angle, dt):
        kalman.x_angle += dt * (dot_angle - kalman.x_bias)
        kalman.mat_00 += (-1) * dt * (kalman.mat_10 + kalman.mat_01) + dt * dt * kalman.mat_11 + kalman.q_angle
        kalman.mat_01 += (-1) * dt * kalman.mat_11
        kalman.mat_10 += (-1) * dt * kalman.mat_11
        kalman.mat_11 += kalman.q_gyro
        
        return
    
    def _update(self, kalman, m_angle):
        y = m_angle - kalman.x_angle
        s = kalman.mat_00 + kalman.r_angle
        k0 = kalman.mat_00 / s
        k1 = kalman.mat_10 / s

        kalman.x_angle += k0 * y
        kalman.x_bias += k1 * y
        kalman.mat_00 -= (k0 * kalman.mat_00)
        kalman.mat_01 -= (k0 * kalman.mat_01)
        kalman.mat_10 -= (k1 * kalman.mat_00)
        kalman.mat_11 -= (k1 * kalman.mat_01)
        
        return kalman.x_angle
    
    def _init_data(self, kalman):
        kalman.q_angle = self.Q_angle
        kalman.q_gyro = self.Q_gyro
        kalman.r_angle = self.R_angle
        
        kalman.mat_00, kalman.mat_01, kalman.mat_10, kalman.mat_11 = 0, 0, 0, 0
        
        return
    
    def _angle_in_degree(self, lower_bound, upper_bound, measured):
        x = (upper_bound - lower_bound) / 180.0
        
        return measured / x

class _Axis:
    def __init__(self):
        self.x = _KalmanGyro()
        self.y = _KalmanGyro()
        self.z = _KalmanGyro()
        self.coord = _RefinedCoordinate()
        
class _KalmanGyro:
    def __init__(self):
        self.x_angle = 0
        self.x_bias = 0
        self.mat_00 = 0
        self.mat_01 = 0
        self.mat_10 = 0
        self.mat_11 = 0
        self.q_angle = 0
        self.q_gyro = 0
        self.r_angle = 0
        
class _RefinedCoordinate:
    def __init__(self):
        self.x, self.y, self.z = 0, 0, 0
        
class _SensorOutput:
    def __init__(self):
        self.value = _ValueGyro()

class _ValueGyro:
    def __init__(self):
        self.x_accel, self.y_accel, self.z_accel = 0, 0, 0
        self.x_gyro, self.y_gyro, self.z_gyro = 0, 0, 0