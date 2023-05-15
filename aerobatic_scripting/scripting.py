import math
import time 
import numpy as np
"""cmd_number
cmd = 1: axial rolls, arg1 = roll rate dps, arg2 = number of rolls
cmd = 2: loops or 180deg return, arg1 = pitch rate dps, arg2 = number of loops, if zero do a 1/2 cuban8-like return
cmd = 3: rolling circle, arg1 = radius, arg2 = number of rolls
cmd = 4: knife edge at any angle, arg1 = roll angle to hold, arg2 = duration
cmd = 5: pause, holding heading and alt to allow stabilization after a move, arg1 = duration in seconds
"""




param_dict = {
               'HGT_P': 1,
               'HGT_I': 2,
               'HGT_KE_ADD': 20,
               'THR_PIT_FF': 80,
               'SPD_P': 5,
               'SPD_I': 25,
               'TRIM_THROTTLE': None,
               'TRIM_ARSPD_CM': None,
               'RLL2SRV_TCONST': None,
               'PTCH2SRV_TCONST': None,
}


DO_JUMP = 177
k_throttle = 70

# last_roll_err = 0.0
# last_id = 0
# initial_yaw_deg = 0
# wp_yaw_deg = 0
# initial_height = 0
# repeat_count = 0
# running = False
# roll_stage = 0

# constrain a value between limits
def constrain(v, vmin, vmax):
   if v < vmin:
      v = vmin
   if v > vmax:
      v = vmax
   return v


# roll angle error 180 wrap to cope with errors while in inverted segments
def roll_angle_error_wrap(roll_angle_error):
   if abs(roll_angle_error) > 180:
    if roll_angle_error > 0:
       roll_angle_error = roll_angle_error - 360
    else:
       roll_angle_error= roll_angle_error +360
   return roll_angle_error
    
# roll controller to keep wings level in earth frame. if arg is 0 then level is at only 0 deg, otherwise its at 180/-180 roll also for loops


# a basic PI controller  class
class PI_controller:

   def __init__(self, iMax, kP = 0, kI = 0, kD = 0):
      self._kP = kP
      self._kI = kI
      self._kD = kD
      self._iMax = iMax
      self._last_t = None
      self._I = 0
      self._P = 0
      self._total = 0
      self._counter = 0
      self._target = 0
      self._current = 0

   #  update the controller.
   def update(self, target, current):
      # now = millis():tofloat() * 0.001 #get time?
      now = time.time()
      if not self._last_t:
         self._last_t = now
      dt = now - self._last_t
      self._last_t = now
      err = target - current
      self._counter += 1

      P = self._kP * err
      self._I += (self._kI * err * dt)
      if self._iMax:
         self._I = constrain(self._I, -self._iMax, self._iMax) # contrain python equiv?
      I = self._I
      ret = P + I

      _target = target
      self._current = current
      self._P = P
      self._total = ret
      return ret

   #reset integrator to an initial value
   def reset(self, integrator):
      self._I = integrator

   def set_I(self, I):
      self._kI = I

   def set_P(self, P):
      self._kP = P
   
   def set_Imax(self, Imax):
      _iMax = Imax
   

class height_controller():
   
   def __init__(self, kP_param,kI_param,KnifeEdge_param, Imax, env):

      self.kP = kP_param
      self.kI = kI_param
      self.knife_edge = KnifeEdge_param
      self.PI = PI_controller(self.kP, self.kI, Imax) # TODO
      self.env = env

   def update(self, target):
      target_pitch = self.PI.update(target, self.env.get_altitude())
      roll_rad = self.env.get_euler()[0]
      ke_add = abs(math.sin(roll_rad)) * self.knife_edge # TODO
      target_pitch = target_pitch + ke_add
      # PI.log("HPI", ke_add)
      return target_pitch

   def reset(self):
      self.PI.reset(max(math.degrees(self.env.get_euler()[1]), 3.0))
      self.PI.set_P(self.kP)
      self.PI.set_I(self.kI)


# local height_PI = height_controller(HGT_P, HGT_I, HGT_KE_BIAS, 20.0)
# local speed_PI = PI_controller(SPD_P:get(), SPD_I:get(), 100.0)

# -- a controller to target a zero pitch angle and zero heading change, used in a roll
# -- output is a body frame pitch rate, with convergence over time tconst in seconds

# -- a controller for throttle to account for pitch

class AcroPy:

   def __init__(self, env, arg1, arg2):
      self.env = env
      self.env.acro_states["running"] = False
      self.env.acro_states["stage"] = 0
      self.params = env.acro_params
      self.arg1 = arg1
      self.arg2 = arg2
      self.height_PI = height_controller(self.params['HGT_P'], self.params['HGT_I'], self.params['HGT_KE_ADD'], 20, self.env)
      self.speed_PI = PI_controller(self.params['SPD_P'], self.params['SPD_I'], 100.0, self.env)
      self.cnum = 0

   def do_manoeuvre(self):
      raise NotImplementedError()
   
   def map_controls(self, roll_rate, pitch_rate, yaw_rate,throttle):
      if len(self.env.controls_high) == 3:
         controls = np.array([roll_rate, pitch_rate, yaw_rate])
         self.env.acro_states["throttle"] = self.throttle_controller()
      elif len(self.env.controls_high) == 4:
         controls = np.array([roll_rate, pitch_rate, yaw_rate, throttle])

      controls = self.env.action_mapper(controls, self.env.controls_low, self.env.controls_high, self.env.action_space.low, self.env.action_space.high)
      # print(f"controls: {controls}")
      return controls

   def initial_states(self):
      self.initial_yaw_deg = self.env.get_euler()[2]
      self.env.acro_states["initial_height"] = self.env.get_altitude()
      self.wp_yaw_deg = self.get_wp_yaw()
      print(f"wp_yaw_deg: {self.wp_yaw_deg}")

   def resolve_jump(self, i):
      m = self.env.waypoints[i]
      while m["command"] == 177:
         i = math.floor(m["param1"])
         m = self.env.waypoints[i]
      return i

   def get_wp_yaw(self):
      cnum = self.env.request_current_waypoint_index()
      loc_prev = self.env.waypoints[cnum-1]
      loc_next = self.env.waypoints[cnum+1]

      i = cnum - 1
      while self.env.waypoints[i]["x"] == 0 and self.env.waypoints[i]["y"] == 0:
         i = i-1
         loc_prev = self.env.waypoints[i]
      i = cnum+1
      while self.env.waypoints[i]["x"] == 0 and self.env.waypoints[i]["y"] == 0:
         i = i+1
         loc_next = self.env.waypoints[self.resolve_jump(i)]
      return self.get_bearing(loc_prev, loc_next)

   # def get_wp_yaw(self):
   #    return np.degrees(self.env.get_euler()[2])


   def get_bearing(self, aLocation1, aLocation2):
    """
    Returns the bearing between the two LocationGlobal objects passed as parameters.

    This method is an approximation, and may not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """	
    off_x = aLocation2["y"] - aLocation1["y"]
    off_y = aLocation2["x"] - aLocation1["x"]
    bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
    if bearing < 0:
        bearing += 360.00
    return bearing


   def _earth_frame_wings_level(self, level_type):
   #  roll_deg = math.deg(ahrs:get_roll()) # get roll angle from pixhawk
      roll_deg = np.degrees(self.env.get_euler()[0]) # get roll angle from pixhawk
      # print("roll_deg", roll_deg)
      roll_angle_error = 0.0
      if (roll_deg > 90 or roll_deg < -90) and level_type != 0:
         roll_angle_error = 180 - roll_deg
      else:
         roll_angle_error = - roll_deg
      return roll_angle_error_wrap(roll_angle_error)/(self.env.acro_params["RLL2SRV_TCONST"])


   def _wrap_360(self, angle):
      res = math.fmod(angle, 360.0)
      if res < 0:
         res = res + 360.0
      return res

   def _wrap_180(self, angle): 
      res = self._wrap_360(angle)
      if res > 180:
         res = res - 360
      return res

   def throttle_controller(self):
      pitch_rad = self.env.get_euler()[1]
      thr_ff = self.env.acro_params["THR_PIT_FF"]
      throttle = self.env.acro_params["TRIM_THROTTLE"] + math.sin(pitch_rad) * thr_ff
      return constrain(throttle, 0, 100.0)

   def _recover_alt(self):
      # target_pitch = self.env.acro_params["HGT_P"]
      target_pitch = self.height_PI.update(self.env.acro_states["initial_height"])
      pitch_rate, yaw_rate = self.pitch_controller(target_pitch, self.wp_yaw_deg, self.env.acro_params["PTCH2SRV_TCONST"])
      return target_pitch, pitch_rate, yaw_rate

   def pitch_controller(self, target_pitch_deg, target_yaw_deg, tconst):

      target_pitch_deg = target_pitch_deg
      target_yaw_deg = target_yaw_deg
      tconst = tconst

      roll_deg = math.degrees(self.env.get_euler()[0]) # get roll angle from pixhawk
      pitch_deg = math.degrees(self.env.get_euler()[1])
      yaw_deg = math.degrees(self.env.get_euler()[2]) # get yaw angle from pixhawk

      # -- get earth frame pitch and yaw rates
      ef_pitch_rate = (target_pitch_deg - pitch_deg) / tconst
      ef_yaw_rate = self._wrap_180(target_yaw_deg - yaw_deg) / tconst

      bf_pitch_rate = math.sin(math.radians(roll_deg)) * ef_yaw_rate + math.cos(math.radians(roll_deg)) * ef_pitch_rate
      bf_yaw_rate   = math.cos(math.radians(roll_deg)) * ef_yaw_rate - math.sin(math.radians(roll_deg)) * ef_pitch_rate
      return bf_pitch_rate, bf_yaw_rate

   # def convert_controls(self, roll_rate, pitch_rate, yaw_rate):
   #    pitch_rate = pitch_rate
   #    yaw_rate = yaw_rate
   #    return pitch_rate, yaw_rate




class LooPy(AcroPy):

   def __init__(self, env, arg1, arg2):
      print("LooPy init")
      super().__init__(env, arg1, arg2)
      self.target_velocity = 0.0


   def do_manoeuvre(self):
      # print("Looping")
      if not self.env.acro_states["running"]:
         self.env.acro_states["running"] = True
         self.env.acro_states["repeat_count"] = self.arg2 - 1

         self.env.acro_states["stage"] = 0
         self.target_velocity = np.linalg.norm(self.env.get_velocity_NED())

         self.initial_states()

         if self.arg2 != 0:
            print("Starting loop")
            # print("Repeat count: ", self.arg2)
         else:
            print("Starting immelman")

      self.env.acro_states["throttle"] = self.throttle_controller()
      # throttle = self.throttle_controller()
      pitch_deg = np.degrees(self.env.get_euler()[1])
      roll_deg = np.degrees(self.env.get_euler()[0])
      yaw_deg = np.degrees(self.env.get_euler()[2])
      # print("Pitch: %f, Roll: %f, Yaw: %f" % (pitch_deg, roll_deg, yaw_deg))
      vel = np.linalg.norm(self.env.get_velocity_NED())
      pitch_rate = self.arg1
      pitch_rate = pitch_rate = pitch_rate * (1+ 2*((vel/self.target_velocity)-1))
      pitch_rate = constrain(pitch_rate, 0.5*self.arg1, 3*self.arg1)
      # print(f"Stage, repeat_count, roll, pitch,{self.env.acro_states['stage']}, {self.env.acro_states['repeat_count']}, {abs(roll_deg)}, {pitch_deg}")

      if self.env.acro_states["stage"] == 0:

         if pitch_deg > 60:
            self.env.acro_states["stage"] = 1
      elif self.env.acro_states["stage"] == 1:
            if (abs(roll_deg) < 90 and pitch_deg > -10 and pitch_deg < 10 and self.env.acro_states["repeat_count"] >= 0):
               print(f"Finished loop {pitch_deg}")
               self.env.acro_states["stage"] = 2
               self.height_PI.reset()
            elif (abs(roll_deg) > 90 and pitch_deg > -10 and pitch_deg < 10 and self.env.acro_states["repeat_count"] < 0):
               print(f"Finished immelman {pitch_deg}")
               self.env.acro_states["stage"] = 2
               self.height_PI.reset()
      # elif self.env.acro_states["stage"] == 2:
      #    # recover alt if abobve or below start and terminate 
      #    if abs(self.env.get_altitude() - self.env.acro_states["initial_height"]) >= 3: # TODO check this
      #       print("Recovering alt")
      #       throttle, pitch_rate, yaw_rate = self._recover_alt()
         # elif self.env.acro_states["repeat_count"] > 0:
         #    self.env.acro_states["stage"] = 0
         #    self.env.acro_stats["repeat_count"]  = self.env.acro_states["repeat_count"] - 1
         # else:
         #    self.running = False
         #    self.end_manoeuvre() # TODO what to do here 
         #    return
      throttle = self.throttle_controller()
      # throttle = self.throttle_controller()
      if self.env.acro_states["stage"] == 2 or self.env.acro_states["stage"] == 0:
         level_type = 0
      else:
         level_type = 1
      if abs(pitch_deg) > 85 and  abs(pitch_deg) < 95:
         roll_rate = 0
      else:
         roll_rate = self._earth_frame_wings_level(level_type)
      # controls = np.array([
      #    constrain(roll_rate, -self.env.pitch_roll_rate_max, self.env.pitch_roll_rate_max),
      #    pitch_rate,
      #    0, throttle])
      # controls = np.array([
      #    constrain(roll_rate, -self.env.pitch_roll_rate_max, self.env.pitch_roll_rate_max),
      #    pitch_rate,
      #    0])
      # print(controls)
      controls = self.map_controls(roll_rate, pitch_rate, 0, throttle)
      # print(controls)
      return controls



class RollPy(AcroPy):

   def __init__(self, env, arg1, arg2):
      super().__init__(env, arg1, arg2)


   def do_manoeuvre(self):
      if not self.env.acro_states["running"]:
         self.env.acro_states["running"] = True
         self.env.acro_states["repeat_count"] = self.arg2 - 1

         self.env.acro_states["stage"] = 0
         self.initial_states()
         self.height_PI.reset()
         print("Starting roll")

      roll_rate = self.arg1
      # pitch_deg = np.degrees(self.env.get_euler()[1])s
      roll_deg = np.degrees(self.env.get_euler()[0])
      if self.env.acro_states["stage"] == 0:
         if roll_deg > 45:
            self.env.acro_states["stage"] = 1
      # commented out as moved to terminal function
      # elif self.env.acro_states["stage"= 1:
      #    if roll_deg > -5 and roll_deg < 5:
      #       print(f"Finished roll r: {roll_deg} p: {pitch_deg}")

      if self.env.acro_states["stage"] < 2:
         throttle = self.throttle_controller()
         # throttle = self.throttle_controller()
         target_pitch = self.height_PI.update(self.env.acro_states["initial_height"])
         pitch_rate, yaw_rate = self.pitch_controller(target_pitch, self.wp_yaw_deg, self.env.acro_params["PTCH2SRV_TCONST"])
         # controls = np.array([roll_rate, pitch_rate, yaw_rate, throttle])
         # print(controls)
         controls = self.map_controls(roll_rate, pitch_rate, yaw_rate, throttle)
         # print(controls)
         return controls


class KnifePy(AcroPy):

   def __init__(self, env, arg1, arg2):
      super().__init__(env, arg1, arg2)
      self.knife_edge_s = 0

   def do_manoeuvre(self):
      now = time.time()
      if not self.env.acro_states["running"]:
         self.env.acro_states["running"] = True
         self.initial_states()
         self.height_PI.reset()
         self.knife_edge_s = now
         print(f"Starting knife edge {self.arg1}")

      i = 0
      self.env.acro_states["duration"] = now - self.knife_edge_s
      # print(f"Duration {self.env.acro_states['duration']}")
      roll_deg = np.degrees(self.env.get_euler()[0])
      roll_angle_error = self.arg1 - roll_deg
      if abs(roll_angle_error) > 180:
         if roll_angle_error > 0:
            roll_angle_error = roll_angle_error - 360
         else:
            roll_angle_error = roll_angle_error + 360
      roll_rate = roll_angle_error / self.env.acro_params["RLL2SRV_TCONST"]
      target_pitch = self.height_PI.update(self.env.acro_states["initial_height"])
      pitch_rate, yaw_rate = self.pitch_controller(target_pitch, self.wp_yaw_deg, self.env.acro_params["PTCH2SRV_TCONST"])
      throttle = self.throttle_controller()
      # throttle = self.throttle_controller()
      # controls = np.array([roll_rate, pitch_rate, yaw_rate, throttle])
      controls = self.map_controls(roll_rate, pitch_rate, yaw_rate, throttle)
      return controls

class PausePy(AcroPy):
   def __init__(self, env, arg1, arg2):
      super().__init__(env, arg1, arg2)
      self.knife_edge_s = 0

   def do_manoeuvre(self):
      now = time.time()
      # print(f" Running? {self.env.acro_states['running']}")
      if not self.env.acro_states["running"]:
         self.env.acro_states["running"] = True
         self.initial_states()
         self.height_PI.reset()
         self.knife_edge_s = now
         print(f"Starting pause {self.arg1}")
      i = 0
      self.env.acro_states["duration"] = now - self.knife_edge_s
      roll_rate = self._earth_frame_wings_level(0)
      target_pitch = self.height_PI.update(self.env.acro_states["initial_height"])
      pitch_rate, yaw_rate = self.pitch_controller(target_pitch, self.wp_yaw_deg, self.env.acro_params["PTCH2SRV_TCONST"])
      throttle = self.throttle_controller()
      
      # throttle = self.throttle_controller()
      # controls = np.array([roll_rate, pitch_rate, yaw_rate, throttle])
      controls = self.map_controls(roll_rate, pitch_rate, yaw_rate, throttle)
      return controls

class CirclePy(AcroPy):
   def __init__(self, env, arg1, arg2):
      super().__init__(env, arg1, arg2)
      self.rolling_circle_last_ms = 0
   
   def do_manoeuvre(self):
      if not self.env.acro_states["running"]:
         self.env.acro_states["running"] = True
         self.env.acro_states["stage"] = 0
         self.env.acro_states["rolling_circle_yaw_deg"] = 0
         self.rolling_circle_last_ms = time.time()
         self.initial_states()
         self.height_PI.reset()
         print("Starting rolling circle")
      yaw_rate_dps = self.arg1
      roll_rate_dps = self.arg2
      pitch_deg = np.degrees(self.env.get_euler()[1])
      roll_deg = np.degrees(self.env.get_euler()[0])
      yaw_deg = np.degrees(self.env.get_euler()[2])
      now = time.time()
      dt = now - self.rolling_circle_last_ms
      self.rolling_circle_last_ms = now

      self.env.acro_states["rolling_circle_yaw_deg"] += yaw_rate_dps * dt

      if self.env.acro_states["stage"] == 0:
         if abs(self.env.acro_states["rolling_circle_yaw_deg"]) > 10:
            self.env.acro_states["stage"] = 1

      if self.env.acro_states["stage"] < 2:
         target_pitch = self.height_PI.update(self.env.acro_states["initial_height"])
         vel = self.env.get_velocity_NED()
         pitch_rate, yaw_rate = self.pitch_controller(
            target_pitch,
            self._wrap_360(self.env.acro_states["rolling_circle_yaw_deg"] + self.initial_yaw_deg),
            self.env.acro_params["PTCH2SRV_TCONST"]
         )
         throttle = self.throttle_controller()
         # throttle = self.throttle_controller()
         # throttle = constrain(throttle, 0, 100)
         # controls = np.array([roll_rate_dps, pitch_rate, yaw_rate, throttle])
         controls =  self.map_controls(roll_rate_dps, pitch_rate, yaw_rate, throttle)
         
         return controls