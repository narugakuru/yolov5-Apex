class IncrementalPID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp # 比例系数
        self.Ki = Ki # 积分系数
        self.Kd = Kd # 微分系数
        self.output = 0  # 输出
        self._pre_output = 0 # 上次输出
        self._pre_error = 0  # 上一时刻误差值
        self._pre_pre_error = 0  # 上上一次时刻误差值

    def PID(self, error):  # 增量式PID
            p_change = self.Kp * (error - self._pre_error)
            i_change = self.Ki * error
            d_change = self.Kd * (error - 2 * self._pre_error + self._pre_pre_error)
            delta_output = p_change + i_change + d_change  # 本次增量

            self.output += delta_output # 本次输出

            #存储本次误差，方便下次使用
            self._pre_error = error
            self._pre_pre_error = self._pre_error
            self._pre_output = self.output

            return self.output