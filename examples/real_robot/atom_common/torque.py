"""Measured static support plus bounded, smoothly changing position correction."""

import math

import numpy as np

RAD = math.pi / 180


class OuterTorqueCompensator:
    """PC2 formula kp*error+ki*integral, equivalently applied as motor Kp*correction.

    The supplied PC2 code adds correction to the position command. Keeping the
    physical position reference unchanged and adding Kp*correction to tau is
    algebraically equivalent under the same joint PD law. Only arms are enabled.
    """

    def __init__(self, joint_kp, effort, max_bias_deg=2.0):
        if not 0 < max_bias_deg <= 3:
            raise ValueError("Compensation bias configuration outside reviewed range")
        self.joint_kp = np.asarray(joint_kp, dtype=float)
        self.effort = np.asarray(effort, dtype=float)
        self.active = np.zeros(17, dtype=bool)
        self.active[1:15] = True
        if (
            self.joint_kp.shape != (17,)
            or self.effort.shape != (17,)
            or not np.isfinite(self.joint_kp).all()
            or np.any(self.joint_kp <= 0)
            or not np.isfinite(self.effort[self.active]).all()
            or np.any(self.effort[self.active] <= 0)
        ):
            raise ValueError("Invalid motor gains or rated arm effort")
        self.limit = np.minimum(max_bias_deg * RAD, self.effort / self.joint_kp)
        self.limit[~self.active] = 0.0
        self.base_tau = np.zeros(17)
        self.integral = np.zeros(17)
        self.bias = np.zeros(17)

    def seed_static(self, tau_est):
        tau = np.asarray(tau_est, dtype=float)
        if tau.shape != (17,) or not np.isfinite(tau).all():
            raise RuntimeError("Invalid static effort feedback")
        tau = tau.copy()
        tau[~self.active] = 0.0
        self._check_rated_effort(tau)
        self.base_tau = tau
        self.bias = np.zeros(17)
        self.integral = np.zeros(17)

    def seed_latched(self, reference, feedback, tau_ff):
        tau = np.asarray(tau_ff, dtype=float)
        ref = np.asarray(reference, dtype=float)
        q = np.asarray(feedback, dtype=float)
        if any(v.shape != (17,) or not np.isfinite(v).all() for v in (tau, ref, q)):
            raise RuntimeError("Invalid latched controller state")
        if np.any(np.abs(tau[~self.active]) > 1e-9):
            raise RuntimeError("Unexpected non-arm feedforward in latched state")
        self._check_rated_effort(tau)
        if np.max(np.abs(ref - q)) >= 2 * RAD:
            raise RuntimeError("Pose no longer matches the previous latched reference")
        self.base_tau = tau.copy()
        self.bias = np.zeros(17)
        self.integral = np.clip(-0.8 * (ref - q) / 0.35, -self.limit / 0.35, self.limit / 0.35)
        self.integral[~self.active] = 0.0

    def _check_rated_effort(self, tau):
        invalid = self.active & (np.abs(tau) > self.effort + 1e-9)
        if np.any(invalid):
            details = [
                {"upper_index": int(i), "tau_Nm": float(tau[i]), "rated_Nm": float(self.effort[i])}
                for i in np.where(invalid)[0]
            ]
            raise RuntimeError("Static support exceeds SDK rated effort: " + str(details))

    def current_tau(self):
        tau = np.clip(self.base_tau + self.joint_kp * self.bias, -self.effort, self.effort)
        tau[~self.active] = 0.0
        return tau

    def step(self, reference, feedback, dt):
        error = np.asarray(reference, dtype=float) - np.asarray(feedback, dtype=float)
        if error.shape != (17,) or not np.isfinite(error).all() or not math.isfinite(dt) or dt < 0:
            raise RuntimeError("Invalid outer-loop input")
        dt = min(dt, 0.05)
        candidate = np.clip(self.integral + error * dt, -self.limit / 0.35, self.limit / 0.35)
        raw = 0.8 * error + 0.35 * candidate
        total = self.base_tau + self.joint_kp * raw
        winding = ((np.abs(raw) > self.limit) & (error * raw > 0)) | (
            (np.abs(total) > self.effort) & (error * total > 0)
        )
        self.integral = np.where(winding, self.integral, candidate)
        target = np.clip(0.8 * error + 0.35 * self.integral, -self.limit, self.limit)
        self.bias += np.clip(target - self.bias, -0.05 * RAD * dt, 0.05 * RAD * dt)
        self.bias[~self.active] = 0.0
        return self.current_tau()
