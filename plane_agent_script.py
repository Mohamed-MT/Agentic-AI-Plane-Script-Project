"""
=============================================================
 Autonomous Fixed-Wing Flight Agent  —  plane_agent.py
 ArduPlane SITL | Agno + OpenRouter | DroneKit + MAVLink
=============================================================

HOW ARDUPLANE NAVIGATION ACTUALLY WORKS
----------------------------------------
A plane cannot hover. Every navigation command is a MAVLink mission
uploaded to the autopilot, which then executes it in AUTO mode.

Mission building rules:
  • Each destination is a NAV_WAYPOINT (fly-through) followed by a
    NAV_LOITER_TURNS or NAV_LOITER_UNLIM (orbit at that point).
  • For multi-stop missions the agent queues ALL stops first, then
    calls execute_plan() ONCE to upload the complete mission.
  • "Return home" → NAV_RETURN_TO_LAUNCH  (loiters over home, NO landing)
  • "Land"        → DO_LAND_START + NAV_LAND at home coords (actual touchdown)
  • Dwell time    → NAV_LOITER_TIME (orbit for N seconds then continue)
  • Hold / stop   → LOITER mode switch (plane orbits current position)
  • Resume        → switch back to AUTO (mission continues from current WP)

Speed: DO_CHANGE_SPEED speed_type=0 (airspeed). Range 9–22 m/s.
Altitude: all AGL. Min 30 m in flight. Max 120 m.
"""

try:
    from dronekit import Command, connect, VehicleMode, LocationGlobalRelative
except Exception:
    from collections import abc
    import collections
    collections.MutableMapping = abc.MutableMapping
    from dronekit import Command, connect, VehicleMode, LocationGlobalRelative

import os

os.environ["OPENROUTER_API_KEY"] = "your-key-here"

import time, math, threading, queue, datetime, json, logging, operator as op_module, re
from pymavlink import mavutil
import tcp_relay
from agno.agent import Agent
from agno.tools import Toolkit
from agno.db.sqlite import SqliteDb
from agno.models.openrouter import OpenRouter
from agno.learn import (LearningMachine, LearningMode,
    UserProfileConfig, UserMemoryConfig, SessionContextConfig, DecisionLogConfig)
from agno.compression.manager import CompressionManager
from agno.skills import Skills, LocalSkills
from agno.utils.log import configure_agno_logging
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional
from pydantic import BaseModel, Field

# ───────────────────────────────────────────────────────────────
# LOGGING
#
# Two log files, both in logs/, both rotating automatically at
# midnight using TimedRotatingFileHandler — so a long-running
# session (or a script restarted daily) always lands in the
# correct day's file without any manual date computation.
#
#   logs/agno.log    — Agno's own internal logs (tool calls, model
#                       calls, agent reasoning). Per Agno's custom
#                       logging docs, the logger named exactly "agno"
#                       is auto-detected and used for ALL Agent logs
#                       (flight_agent, safety_agent, summary_agent,
#                       _planner, _safety_validator, _summariser all
#                       share this since they're all Agent instances).
#   logs/flight.log   — our own flight events (arm, takeoff, goto,
#                       land, errors, etc.) via flog().
#
# When rotated, old files are renamed with a date suffix, e.g.
# agno.log.2026-06-29, flight.log.2026-06-29 — exactly like Python's
# standard TimedRotatingFileHandler behaviour.
# ───────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

def _make_daily_handler(path, level):
    """TimedRotatingFileHandler that rolls over at local midnight, keeps 30 days."""
    h = TimedRotatingFileHandler(
        path, when="midnight", interval=1, backupCount=30,
        encoding="utf-8", delay=False, utc=False,
    )
    h.suffix = "%Y-%m-%d"
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    return h

# ── Agno logger ──────────────────────────────────────────────────────────────
# Named exactly "agno" — Agno auto-detects this name and routes ALL Agent
# logging through it (see https://docs.agno.com/custom-logging, "Using Named
# Loggers"). We still call configure_agno_logging explicitly too, since that
# is the documented supported entry point and is more robust across versions.
_AGNO_LOG_PATH = os.path.join(_LOG_DIR, "agno.log")
_agno_logger = logging.getLogger("agno")
_agno_logger.setLevel(logging.INFO)
_agno_logger.propagate = False
for _h in _agno_logger.handlers[:]:
    _agno_logger.removeHandler(_h)
_ah = _make_daily_handler(_AGNO_LOG_PATH, logging.INFO)
_agno_logger.addHandler(_ah)
configure_agno_logging(custom_default_logger=_agno_logger)

# ── Flight logger ─────────────────────────────────────────────────────────────
# Our own flight-event logger (arm, takeoff, goto, land, errors, etc).
# Uses a unique logger name to avoid colliding with anything else in the
# process; propagate=False keeps messages from bubbling to the root logger.
_FLIGHT_LOG_PATH = os.path.join(_LOG_DIR, "flight.log")
flight_logger = logging.getLogger(f"plane_agent.flight.{os.getpid()}")
flight_logger.setLevel(logging.DEBUG)
flight_logger.propagate = False
for _h in flight_logger.handlers[:]:
    flight_logger.removeHandler(_h)
_fh = _make_daily_handler(_FLIGHT_LOG_PATH, logging.DEBUG)
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
flight_logger.addHandler(_fh)
flight_logger.addHandler(_ch)

def flog(level: str, msg: str):
    getattr(flight_logger, level.lower(), flight_logger.info)(msg)
    # Explicit flush so messages reach disk even if the process exits abruptly
    try:
        _fh.flush()
    except Exception:
        pass

# Startup test writes — confirm both log files are writable, and prove the
# rotation is wired up correctly, before anything else runs.
flog("info", f"=== plane_agent.py started | PID {os.getpid()} | flight log: {_FLIGHT_LOG_PATH} ===")
_agno_logger.info(f"=== plane_agent.py started | PID {os.getpid()} | agno log: {_AGNO_LOG_PATH} ===")


# ───────────────────────────────────────────────────────────────
# STRUCTURED OUTPUT SCHEMAS
# ───────────────────────────────────────────────────────────────
class PlaneCommand(BaseModel):
    action:    str             = Field(description="Plane action to execute")
    altitude:  Optional[float] = Field(None, description="Target altitude AGL metres")
    latitude:  Optional[float] = Field(None, description="Target latitude")
    longitude: Optional[float] = Field(None, description="Target longitude")
    direction: Optional[str]   = Field(None, description="Compass direction")
    distance:  Optional[float] = Field(None, description="Distance in metres")
    speed:     Optional[float] = Field(None, description="Target airspeed m/s")
    dwell:     Optional[float] = Field(None, description="Loiter time in seconds")
    reason:    str             = Field(default="", description="Why this command")

class MissionPlan(BaseModel):
    mission_name:           str               = Field(description="Short mission name")
    objective:              str               = Field(description="Mission goal")
    steps:                  List[PlaneCommand]= Field(description="Ordered commands")
    risk_level:             str               = Field(description="LOW/MEDIUM/HIGH")
    estimated_time_seconds: int               = Field(description="Estimated duration")
    notes:                  str               = Field(default="")

class SafetyAssessment(BaseModel):
    is_safe:         bool      = Field(description="True if mission is safe")
    risk_level:      str       = Field(description="LOW/MEDIUM/HIGH/CRITICAL")
    issues:          List[str] = Field(description="Safety issues found")
    recommendations: List[str] = Field(description="Suggested mitigations")
    approved:        bool      = Field(description="Final approval")

# ───────────────────────────────────────────────────────────────
# PRESET LOCATIONS
# ───────────────────────────────────────────────────────────────
PRESET_LOCATIONS = {
    "home":        {"lat": -35.363261, "lon": 149.165230, "description": "SITL home / launch point"},
    "airfield":    {"lat": -35.362749, "lon": 149.165353, "description": "Canberra Airfield"},
    "runway 35":   {"lat": -35.363328, "lon": 149.165223, "description": "Runway 35"},
    "runway 17":   {"lat": -35.362227, "lon": 149.165074, "description": "Runway 17"},
    "hospital":    {"lat": -35.354167, "lon": 149.150560, "description": "Mugga Mugga Hospital"},
    "prison":      {"lat": -35.371077, "lon": 149.172684, "description": "West Jerrabomberra Prison"},
    "camp a":      {"lat": -35.360338, "lon": 149.151874, "description": "West Jerrabomberra Camp A"},
    "camp b":      {"lat": -35.361530, "lon": 149.154562, "description": "West Jerrabomberra Location 2"},
    "reserve":     {"lat": -35.366030, "lon": 149.150095, "description": "West Jerrabomberra Reserve"},
    "residence 1": {"lat": -35.357340, "lon": 149.170626, "description": "Jerrabomberra Residence 1"},
    "residence 2": {"lat": -35.346840, "lon": 149.154976, "description": "Jerrabomberra North Residence"},
    "creek south": {"lat": -35.363393, "lon": 149.175728, "description": "Jerrabomberra Creek South"},
    "location 1":  {"lat": -35.364759, "lon": 149.152459, "description": "West Jerrabomberra Location 1"},
}

# ───────────────────────────────────────────────────────────────
# VEHICLE CONNECTION
# ───────────────────────────────────────────────────────────────
print("Connecting to vehicle on tcp:127.0.0.1:5763 ...")
vehicle = connect("tcp:127.0.0.1:5763", wait_ready=True, baud=57600, rate=60)
print("Vehicle connected.")
while vehicle.location.local_frame.north is None:
    time.sleep(1)
    print("  Waiting for local frame...")
print("Local frame ready.")

# Clear any mission left on the flight controller from a previous session.
# ArduPlane refuses to arm if a DO_LAND_START mission is still stored,
# even after a full script restart. This runs once at startup to guarantee
# a clean state every time.
try:
    _init_cmds = vehicle.commands
    _init_cmds.download()
    _init_cmds.wait_ready()
    if len(_init_cmds) > 0:
        _init_cmds.clear()
        _init_cmds.upload()
        print("Startup: stale mission cleared — arm lock released.")
    else:
        print("Startup: no stale mission found.")
except Exception as _ie:
    print(f"Startup: mission clear failed: {_ie}")

# Cache home location (set once after GPS lock)
_home_lat = PRESET_LOCATIONS["home"]["lat"]
_home_lon = PRESET_LOCATIONS["home"]["lon"]

def _refresh_home():
    """Pull home from vehicle if available, else use preset."""
    global _home_lat, _home_lon
    try:
        h = vehicle.home_location
        if h and h.lat and abs(h.lat) > 0.001:
            _home_lat = h.lat
            _home_lon = h.lon
    except Exception:
        pass

# ───────────────────────────────────────────────────────────────
# TCP RELAY  →  Unreal Engine visualisation
# ───────────────────────────────────────────────────────────────
relay = tcp_relay.TCP_Relay()

def _unreal_loop():
    while True:
        try:
            loc  = vehicle.location.local_frame
            att  = vehicle.attitude
            flds = [0.0] * relay.num_fields
            flds[0] = (loc.north or 0) * 100
            flds[1] = (loc.east  or 0) * 100
            flds[2] = (loc.down  or 0) * 100 * -1
            flds[3] = math.degrees(att.roll)
            flds[4] = math.degrees(att.pitch)
            flds[5] = math.degrees(att.yaw)
            relay.message = tcp_relay.create_fields_string(flds)
        except Exception:
            pass
        time.sleep(1/60)

threading.Thread(target=_unreal_loop, daemon=True).start()

# ───────────────────────────────────────────────────────────────
# SESSION STATE
# ───────────────────────────────────────────────────────────────
SESSION_START = datetime.datetime.now()
SESSION_ID    = f"plane-{SESSION_START.strftime('%H%M%S')}"
flog("info", f"SESSION START — {SESSION_ID}")

mission_state = {
    "phase":           "idle",
    "flight_log":      [],
    "max_altitude":    0.0,
    "battery_start":   None,
    "battery_current": None,
    "incidents":       [],
    "last_command":    {},
    "pending_mission": [],
    "current_mission": None,
}
try:
    mission_state["battery_start"] = vehicle.battery.level
except Exception:
    pass

# ───────────────────────────────────────────────────────────────
# NAV_CONTROLLER_OUTPUT listener  (gives wp_dist in AUTO)
# ───────────────────────────────────────────────────────────────
vehicle.wp_dist = None
vehicle.wp_eta  = None

def _nav_listener(self, name, msg):
    self.wp_dist = msg.wp_dist
    spd = (self.airspeed if self.airspeed and self.airspeed > 1 else self.groundspeed) or 1
    self.wp_eta = int(msg.wp_dist / spd)

vehicle.add_message_listener("NAV_CONTROLLER_OUTPUT", _nav_listener)

# ───────────────────────────────────────────────────────────────
# MAVLink COMMAND BUILDERS
# All use MAV_FRAME_GLOBAL_RELATIVE_ALT  (alt = metres AGL)
# ───────────────────────────────────────────────────────────────
_FR = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

def _wp(lat, lon, alt, acc_radius=50):
    """
    NAV_WAYPOINT — plane flies to this point and moves on.
    acc_radius (param2): distance in m at which waypoint is considered reached.
    Set to 50 m so the plane doesn't have to fly directly over the point.
    param3=0: pass straight through (not orbit).
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                   0, 0,
                   0,          # hold time (ignored by plane)
                   acc_radius, # acceptance radius
                   0,          # pass-through (0 = straight through)
                   0,          # yaw (ignored)
                   lat, lon, alt)

def _loiter_unlim(lat, lon, alt, radius=150):
    """
    NAV_LOITER_UNLIM — orbit forever until mode change.
    Used as the FINAL item when the plane should stay at a location.
    param3 = radius (positive = CW).
    lat/lon/alt = 0 means use current position.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
                   0, 0,
                   0, radius, 0, 0,
                   lat, lon, alt)

def _loiter_turns(lat, lon, alt, turns=2, radius=150):
    """
    NAV_LOITER_TURNS — orbit N times then continue to next WP.
    Used as a dwell command mid-mission.
    param1 = turns, param3 = radius.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
                   0, 0,
                   turns, 0, radius, 0,
                   lat, lon, alt)

def _loiter_time(lat, lon, alt, seconds=30, radius=150):
    """
    NAV_LOITER_TIME — orbit for N seconds then continue.
    param1 = seconds, param3 = radius.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME,
                   0, 0,
                   seconds, 0, radius, 0,
                   lat, lon, alt)

def _rtl():
    """
    NAV_RETURN_TO_LAUNCH — fly home and LOITER there (NO landing).
    ArduPlane loiters at RTL_ALTITUDE over home until mode is changed.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                   0, 0,
                   0,0,0,0, 0,0,0)

def _do_land_start():
    """
    DO_LAND_START — marker that tells ArduPlane a landing sequence begins.
    Must immediately precede NAV_LAND.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_DO_LAND_START,
                   0, 0,
                   0,0,0,0, 0,0,0)

def _nav_land(lat, lon):
    """
    NAV_LAND — actual touchdown.
    lat/lon = landing point (use home coords).
    alt = 0 (ground level).
    param1 = abort altitude (0 = use default).
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_LAND,
                   0, 0,
                   0,   # abort alt
                   0,0,0,
                   lat, lon, 0)

def _nav_takeoff(alt, pitch=15):
    """
    NAV_TAKEOFF — climb to alt at pitch angle then go to next WP.
    For SITL this is usually followed immediately by a NAV_WAYPOINT.
    param1 = pitch angle degrees.
    """
    return Command(0,0,0, _FR,
                   mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                   0, 0,
                   pitch, 0,0,0,
                   0, 0, alt)

# ───────────────────────────────────────────────────────────────
# NON-STANDARD ORBIT PATTERN BUILDERS
# racetrack / figure-eight / lawnmower — built from the same _wp() and
# _loiter_turns() primitives above, no new MAV_CMD types needed.
#
# Each builder returns (commands, last_lat, last_lon). last_lat/last_lon is
# where the pattern naturally ends, so the caller (executor_loop) can tack
# on the SAME end-of-mission behaviour execute_plan() already uses:
# _loiter_unlim() to hold there, _rtl(), or _do_land_start()+_nav_land().
#
# Laps/loops are fully unrolled into the command list rather than using
# MAV_CMD_DO_JUMP to loop. DO_JUMP's target sequence number depends on
# whether the flight controller counts home as mission item 0, which is
# inconsistent across ArduPilot versions — unrolling avoids that landmine
# entirely; the mission lists here are small enough that it doesn't matter.
# ───────────────────────────────────────────────────────────────
def _build_racetrack(center_lat, center_lon, altitude, heading,
                      leg_length_m=400.0, turn_radius_m=150.0,
                      turn_direction="right", laps=3):
    """
    Two parallel straight legs joined by two turns — a stretched oval,
    like an aviation holding pattern. Good for loitering over something
    elongated (a road, ridge, fence line) instead of a single point.
    heading is the OUTBOUND leg direction. turn_direction: 'right' (CW,
    positive loiter radius) or 'left' (CCW, negative radius).
    """
    hdg = _bearing(heading) if isinstance(heading, str) else float(heading) % 360.0
    half = max(float(leg_length_m), 1.0) / 2.0
    a_lat, a_lon = _offset_coord(center_lat, center_lon, hdg, half)
    b_lat, b_lon = _offset_coord(center_lat, center_lon, (hdg + 180.0) % 360.0, half)
    r = abs(turn_radius_m) if str(turn_direction).lower().startswith("r") else -abs(turn_radius_m)

    cmds = []
    for _ in range(max(1, int(laps))):
        cmds.append(_wp(a_lat, a_lon, altitude))
        cmds.append(_loiter_turns(a_lat, a_lon, altitude, turns=1, radius=r))
        cmds.append(_wp(b_lat, b_lon, altitude))
        cmds.append(_loiter_turns(b_lat, b_lon, altitude, turns=1, radius=r))
    return cmds, b_lat, b_lon

def _build_figure_eight(center_lat, center_lon, altitude, heading,
                         lobe_radius_m=150.0, loops=2, first_lobe="right"):
    """
    Two orbit circles of equal radius, tangent at (center_lat, center_lon),
    flown in opposite directions so the ground track crosses itself — a
    figure-8 aligned along `heading`. Returns the crossing point as the
    last point, since each loiter_turns exits back near where it entered.
    """
    hdg = _bearing(heading) if isinstance(heading, str) else float(heading) % 360.0
    r = abs(lobe_radius_m)
    lobe_a_lat, lobe_a_lon = _offset_coord(center_lat, center_lon, hdg, r)
    lobe_b_lat, lobe_b_lon = _offset_coord(center_lat, center_lon, (hdg + 180.0) % 360.0, r)
    r_a = r if str(first_lobe).lower().startswith("r") else -r
    r_b = -r_a

    cmds = [_wp(center_lat, center_lon, altitude)]
    for _ in range(max(1, int(loops))):
        cmds.append(_loiter_turns(lobe_a_lat, lobe_a_lon, altitude, turns=1, radius=r_a))
        cmds.append(_loiter_turns(lobe_b_lat, lobe_b_lon, altitude, turns=1, radius=r_b))
    return cmds, center_lat, center_lon

def _build_lawnmower(center_lat, center_lon, altitude, heading,
                      length_m, width_m, track_spacing_m=100.0):
    """
    Parallel back-and-forth legs covering a rectangular area — a classic
    survey/search "mowing the lawn" grid. heading is the LONG-leg direction.
    """
    hdg = _bearing(heading) if isinstance(heading, str) else float(heading) % 360.0
    perp = (hdg + 90.0) % 360.0
    half_len = max(float(length_m), 1.0) / 2.0
    half_wid = max(float(width_m), 1.0) / 2.0
    spacing = max(float(track_spacing_m), 10.0)

    n_tracks = max(1, int(round(width_m / spacing)) + 1)
    if n_tracks == 1:
        offsets = [0.0]
    else:
        step = (2 * half_wid) / (n_tracks - 1)
        offsets = [-half_wid + i * step for i in range(n_tracks)]

    cmds = []
    last_lat, last_lon = center_lat, center_lon
    for i, off in enumerate(offsets):
        row_lat, row_lon = _offset_coord(center_lat, center_lon, perp, off)
        end_a = _offset_coord(row_lat, row_lon, hdg, +half_len)
        end_b = _offset_coord(row_lat, row_lon, hdg, -half_len)
        first, second = (end_a, end_b) if i % 2 == 0 else (end_b, end_a)
        cmds.append(_wp(first[0], first[1], altitude))
        cmds.append(_wp(second[0], second[1], altitude))
        last_lat, last_lon = second
    return cmds, last_lat, last_lon

# ───────────────────────────────────────────────────────────────
# MISSION UPLOAD  +  DOWNLOAD
# ───────────────────────────────────────────────────────────────
def _upload_mission(commands: list):
    """
    Clear existing mission, upload new list of Command objects,
    reset mission index to 0, then switch to AUTO.
    IMPORTANT: we do NOT reset vehicle.commands.next here —
    that is only done on a fresh mission start, not on resume.
    """
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    cmds.clear()
    for c in commands:
        cmds.add(c)
    cmds.upload()
    flog("info", f"Mission uploaded: {len(commands)} item(s)")

def _download_mission():
    """Return current uploaded mission as list of readable dicts."""
    _MAP = {
        16: "NAV_WAYPOINT",      17: "NAV_LOITER_UNLIM",
        18: "NAV_LOITER_TURNS",  19: "NAV_LOITER_TIME",
        20: "NAV_RETURN_TO_LAUNCH", 21: "NAV_LAND",
        22: "NAV_TAKEOFF",       189: "DO_LAND_START",
        112: "CONDITION_DELAY",
    }
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    out = []
    for i, c in enumerate(cmds):
        d = c.__dict__
        out.append({
            "index":   i,
            "command": _MAP.get(d["command"], f"CMD_{d['command']}"),
            "lat":     round(d["x"], 6),
            "lon":     round(d["y"], 6),
            "alt":     round(d["z"], 1),
            "p1":      d["param1"],
            "p2":      d["param2"],
            "p3":      d["param3"],
        })
    return out

# ───────────────────────────────────────────────────────────────
# MODE HELPERS
# ───────────────────────────────────────────────────────────────
def _set_mode(mode_name: str, retries=5) -> bool:
    for _ in range(retries):
        vehicle.mode = VehicleMode(mode_name)
        for _ in range(20):
            if vehicle.mode.name == mode_name:
                return True
            time.sleep(0.2)
    flog("warning", f"_set_mode: failed to reach {mode_name}")
    return False

# ───────────────────────────────────────────────────────────────
# SPEED CONTROL  (ArduPlane = airspeed, type=0)
# ───────────────────────────────────────────────────────────────
_SPEED_MIN =  9.0   # m/s  — near stall
_SPEED_MAX = 22.0   # m/s
_SPEED_CRZ = 15.0   # m/s  — cruise default

# Tracks the last speed we commanded so status readouts can show
# target vs measured airspeed. vehicle.airspeed is the pitot reading
# and lags behind the commanded value while the PID catches up.
_target_airspeed: float = _SPEED_CRZ

def _set_speed(speed_ms: float) -> float:
    global _target_airspeed
    speed_ms = max(_SPEED_MIN, min(_SPEED_MAX, float(speed_ms)))
    _target_airspeed = speed_ms
    try:
        vehicle.parameters["TRIM_ARSPD_CM"] = int(speed_ms * 100)
    except Exception as e:
        flog("warning", f"_set_speed param write: {e}")
    for _ in range(3):
        msg = vehicle.message_factory.command_long_encode(
            0, 0,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            0,        # speed_type 0 = airspeed
            speed_ms,
            -1,       # throttle no change
            0,0,0,0)
        vehicle.send_mavlink(msg)
        vehicle.flush()
        time.sleep(0.05)
    flog("info", f"Airspeed target set to {speed_ms:.1f} m/s")
    return speed_ms

# ───────────────────────────────────────────────────────────────
# NAVIGATION GEOMETRY
# ───────────────────────────────────────────────────────────────
def _offset_coord(lat, lon, bearing_deg, dist_m):
    """Return (lat2, lon2) reached by travelling dist_m on bearing_deg from (lat,lon)."""
    R = 6378137.0
    b = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(dist_m/R) +
                     math.cos(lat1)*math.sin(dist_m/R)*math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b)*math.sin(dist_m/R)*math.cos(lat1),
                              math.cos(dist_m/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

_COMPASS = {
    "north":0, "n":0, "northeast":45, "ne":45,
    "east":90, "e":90, "southeast":135, "se":135,
    "south":180, "s":180, "southwest":225, "sw":225,
    "west":270, "w":270, "northwest":315, "nw":315,
}

def _bearing(direction_str: str) -> float:
    d = direction_str.lower().strip()
    if d in _COMPASS:
        return _COMPASS[d]
    try:
        return float(d)
    except ValueError:
        flog("warning", f"Unknown direction '{direction_str}', defaulting north")
        return 0.0

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6378137.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

# ───────────────────────────────────────────────────────────────
# ARRIVAL MONITOR
# Waits for wp_dist (from NAV_CONTROLLER_OUTPUT) to drop inside
# WP_LOITER_RAD, which means the plane has entered its orbit.
# Falls back to haversine if wp_dist is not yet available.
# ───────────────────────────────────────────────────────────────
def _wait_arrival(target_lat, target_lon, timeout=300):
    loiter_rad = abs(vehicle.parameters.get("WP_LOITER_RAD", 300))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_flag.is_set():
            return False
        wp = getattr(vehicle, "wp_dist", None)
        if wp is not None:
            if wp <= loiter_rad:
                flog("info", f"Arrival confirmed — wp_dist={wp:.0f}m")
                return True
        else:
            loc = vehicle.location.global_relative_frame
            if loc.lat and loc.lon:
                if _haversine(loc.lat, loc.lon, target_lat, target_lon) <= loiter_rad:
                    return True
        time.sleep(1.0)
    flog("warning", "Arrival timeout")
    return False

def _wait_mission_complete(timeout=600):
    """
    Wait until ArduPlane finishes executing ALL waypoints in AUTO.
    Detected when mode automatically changes away from AUTO, or when
    vehicle.commands.next reaches the end of the mission list.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop_flag.is_set():
            return False
        mode = vehicle.mode.name
        # ArduPlane drops out of AUTO when mission ends (goes to RTL or LOITER)
        if mode != "AUTO":
            flog("info", f"Mission complete — mode changed to {mode}")
            return True
        time.sleep(1.0)
    flog("warning", "wait_mission_complete: timeout")
    return False

# ───────────────────────────────────────────────────────────────
# LANDING SAFETY CHECK
#
# We don't have live terrain/obstacle sensing in this SITL setup, so the
# check is a simple radius/heading check against KNOWN locations:
#   - LANDING_HAZARDS: populated/restricted areas — never land within
#     HAZARD_CLEARANCE_M of these, no exceptions.
#   - SAFE_LANDING_ZONES: pre-approved open areas — landing here always
#     passes the check immediately.
#   - Anywhere else: assumed clear UNLESS within HAZARD_CLEARANCE_M of a
#     hazard, in which case it's blocked and the nearest safe zone is
#     suggested instead.
# ───────────────────────────────────────────────────────────────
LANDING_HAZARDS = ["hospital", "prison", "residence 1", "residence 2"]
SAFE_LANDING_ZONES = ["home", "airfield", "runway 35", "runway 17", "reserve"]
HAZARD_CLEARANCE_M = 200   # minimum distance from any hazard to land

def _nearest_safe_zone(lat, lon):
    """Return (name, distance_m) of the closest pre-approved safe landing zone."""
    best_name, best_dist = None, float("inf")
    for name in SAFE_LANDING_ZONES:
        loc = PRESET_LOCATIONS[name]
        d = _haversine(lat, lon, loc["lat"], loc["lon"])
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name, best_dist

def check_landing_safety(lat, lon):
    """
    Evaluate whether (lat, lon) is safe to land at.
    Returns a dict: {safe: bool, reason: str, nearest_safe_zone: str|None,
                      nearest_safe_zone_dist_m: float|None}
    """
    # 1. Pre-approved safe zone — always passes
    for name in SAFE_LANDING_ZONES:
        loc = PRESET_LOCATIONS[name]
        d = _haversine(lat, lon, loc["lat"], loc["lon"])
        if d <= HAZARD_CLEARANCE_M:
            return {"safe": True,
                    "reason": f"Within {d:.0f}m of pre-approved safe zone '{name}'.",
                    "nearest_safe_zone": name, "nearest_safe_zone_dist_m": round(d)}

    # 2. Hazard check — block if too close to a known restricted area
    for name in LANDING_HAZARDS:
        loc = PRESET_LOCATIONS[name]
        d = _haversine(lat, lon, loc["lat"], loc["lon"])
        if d <= HAZARD_CLEARANCE_M:
            safe_name, safe_dist = _nearest_safe_zone(lat, lon)
            return {"safe": False,
                    "reason": f"Only {d:.0f}m from '{name}' ({PRESET_LOCATIONS[name]['description']}) "
                              f"— minimum clearance is {HAZARD_CLEARANCE_M}m.",
                    "nearest_safe_zone": safe_name, "nearest_safe_zone_dist_m": round(safe_dist)}

    # 3. No known hazard nearby — assumed clear (no terrain data available)
    safe_name, safe_dist = _nearest_safe_zone(lat, lon)
    return {"safe": True,
            "reason": "No known hazard within clearance radius — assumed clear "
                      "(no live terrain/obstacle sensing in this simulation).",
            "nearest_safe_zone": safe_name, "nearest_safe_zone_dist_m": round(safe_dist)}

# ───────────────────────────────────────────────────────────────
# FILESYSTEM TOOLKIT
# ───────────────────────────────────────────────────────────────
class FilesystemToolkit(Toolkit):
    def __init__(self):
        self._md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missions")
        self._rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(self._md, exist_ok=True)
        os.makedirs(self._rd, exist_ok=True)
        super().__init__(name="filesystem",
            tools=[self.list_missions, self.read_mission, self.save_report, self.list_reports])

    def list_missions(self) -> str:
        "List mission files in missions/."
        f = [x for x in os.listdir(self._md) if x.endswith((".txt",".json"))]
        return "Missions:\n" + "\n".join(f"  {x}" for x in sorted(f)) if f else "No mission files."

    def read_mission(self, filename: str) -> str:
        "Read a mission file."
        p = os.path.join(self._md, os.path.basename(filename))
        if not os.path.exists(p):
            return f"Not found: {filename}"
        with open(p, encoding="utf-8") as f:
            return f.read()

    def save_report(self, content: str, filename: str = "") -> str:
        "Save a flight report."
        if not filename:
            filename = f"report_{SESSION_ID}.txt"
        filename = os.path.basename(filename)
        if not filename.endswith((".txt",".json",".md")):
            filename += ".txt"
        p = os.path.join(self._rd, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved: reports/{filename}"

    def list_reports(self) -> str:
        "List saved reports."
        f = [x for x in os.listdir(self._rd) if x.endswith((".txt",".json",".md"))]
        return "Reports:\n" + "\n".join(f"  {x}" for x in sorted(f,reverse=True)) if f else "No reports."

_fs = FilesystemToolkit()

# ───────────────────────────────────────────────────────────────
# COMMAND QUEUE + FLAGS
# ───────────────────────────────────────────────────────────────
command_queue = queue.Queue()
stop_flag     = threading.Event()
_hold_active  = threading.Event()   # True while plane is in LOITER hold

def _clear_queue():
    saved = []
    while not command_queue.empty():
        try:
            saved.append(command_queue.get_nowait())
            command_queue.task_done()
        except queue.Empty:
            break
    if saved:
        print(f"[QUEUE] Cleared {len(saved)} pending command(s).")
    mission_state["pending_mission"] = saved

# ───────────────────────────────────────────────────────────────
# CONDITION MONITOR
# ───────────────────────────────────────────────────────────────
_OPS = {
    "==": op_module.eq, "!=": op_module.ne,
    "<":  op_module.lt, "<=": op_module.le,
    ">":  op_module.gt, ">=": op_module.ge,
}
CONDITION_FIELDS = ["rel_alt","airspeed","groundspeed","battery_level",
                    "battery_voltage","armed","mode","airborne","yaw"]

class _Watch:
    def __init__(self, field, op, value, action, params=None, label=""):
        self.field   = field
        self.op      = op
        self.value   = value
        self.action  = action
        self.params  = params or {}
        self.label   = label or f"{field} {op} {value} -> {action}"
        self.triggered = False

    def check(self, state):
        sv = state.get(self.field)
        if sv is None: return False
        fn = _OPS.get(self.op)
        if not fn: return False
        try: return fn(sv, self.value)
        except Exception: return False

class ConditionMonitor:
    def __init__(self):
        self._watches = []
        self._lock    = threading.Lock()
        threading.Thread(target=self._tick, daemon=True).start()

    def add(self, w):
        with self._lock: self._watches.append(w)
        print(f"[COND] Watching: {w.label}")

    def clear(self):
        with self._lock: self._watches.clear()

    def list_all(self):
        with self._lock:
            return "\n".join(f"  {w.label}" for w in self._watches) or "None."

    def _state(self):
        try:
            loc  = vehicle.location.global_relative_frame
            batt = vehicle.battery
            return {
                "rel_alt":         round(loc.alt or 0, 1),
                "airspeed":        round(vehicle.airspeed or 0, 1),
                "groundspeed":     round(vehicle.groundspeed or 0, 1),
                "battery_level":   batt.level or 0,
                "battery_voltage": round(batt.voltage or 0, 2),
                "armed":           vehicle.armed,
                "mode":            vehicle.mode.name,
                "yaw":             round(math.degrees(vehicle.attitude.yaw), 1),
                "airborne":        (loc.alt or 0) > 5.0,
            }
        except Exception:
            return {}

    def _tick(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                if not self._watches: continue
                state = self._state()
                remaining = []
                for w in self._watches:
                    if w.triggered: continue
                    if w.check(state):
                        print(f"\n[COND TRIGGERED] {w.label}")
                        w.triggered = True
                        stop_flag.set()
                        _clear_queue()
                        cmd = {"action": w.action}
                        cmd.update(w.params)
                        command_queue.put(cmd)
                    else:
                        remaining.append(w)
                self._watches = remaining

cond_monitor = ConditionMonitor()

# ───────────────────────────────────────────────────────────────
# LOGGING HELPER
# ───────────────────────────────────────────────────────────────
def _log(action, details=None):
    loc = vehicle.location.global_relative_frame
    e = {
        "time":    datetime.datetime.now().strftime("%H:%M:%S"),
        "action":  action,
        "lat":     round(loc.lat, 6) if loc.lat else None,
        "lon":     round(loc.lon, 6) if loc.lon else None,
        "alt":     round(loc.alt, 1) if loc.alt else None,
        "details": details or {},
    }
    mission_state["flight_log"].append(e)
    if loc.alt:
        mission_state["max_altitude"] = max(mission_state["max_altitude"], loc.alt)
    try:
        mission_state["battery_current"] = vehicle.battery.level
    except Exception:
        pass

# ───────────────────────────────────────────────────────────────
# MISSION PLAN BUILDER
# The agent calls add_stop() for each destination, then
# execute_plan() once. This ensures the entire route is uploaded
# as a single mission — preventing mid-flight mission wipes.
# ───────────────────────────────────────────────────────────────
_plan_lock   = threading.Lock()
_plan_stops  = []   # list of dicts: {lat, lon, alt, dwell_s, name, is_final_loiter}

def _plan_clear():
    with _plan_lock:
        _plan_stops.clear()

def _plan_add(lat, lon, alt, dwell_s=0, name="", is_final=False):
    with _plan_lock:
        _plan_stops.append({
            "lat": lat, "lon": lon, "alt": alt,
            "dwell_s": dwell_s, "name": name, "is_final": is_final,
        })

def _plan_build_commands(end_with_land=False, end_with_rtl=False) -> list:
    """
    Convert _plan_stops into a MAVLink Command list.

    Stop types:
      • Normal stop (dwell_s == 0):  NAV_WAYPOINT + NAV_LOITER_TURNS(2)
        The plane flies through the WP and does 2 orbits before next stop.
      • Dwell stop (dwell_s > 0):    NAV_WAYPOINT + NAV_LOITER_TIME(dwell_s)
        The plane orbits for dwell_s seconds then continues.
      • Final stop (is_final, no land/rtl):  NAV_WAYPOINT + NAV_LOITER_UNLIM
        The plane orbits indefinitely until the operator intervenes.

    End options (mutually exclusive):
      end_with_land=True  → last stop uses NAV_LOITER_TURNS(1) then
                             DO_LAND_START + NAV_LAND at home coords.
      end_with_rtl=True   → last stop uses NAV_LOITER_TURNS(1) then
                             NAV_RETURN_TO_LAUNCH (loiters home, no landing).
    """
    with _plan_lock:
        stops = list(_plan_stops)

    if not stops:
        return []

    cmds = []
    loiter_r = 150   # m — default loiter radius

    for i, stop in enumerate(stops):
        lat  = stop["lat"]
        lon  = stop["lon"]
        alt  = stop["alt"]
        dw   = stop.get("dwell_s", 0)
        last = (i == len(stops) - 1)

        # NAV_WAYPOINT — fly to this coordinate
        cmds.append(_wp(lat, lon, alt))

        if last:
            if end_with_land or end_with_rtl:
                # One orbit over the last stop before heading home/landing
                cmds.append(_loiter_turns(lat, lon, alt, turns=1, radius=loiter_r))
            elif dw > 0:
                # Dwell then... there's nothing after, so orbit forever
                cmds.append(_loiter_time(lat, lon, alt, seconds=int(dw), radius=loiter_r))
            else:
                # Final destination — orbit forever
                cmds.append(_loiter_unlim(lat, lon, alt, radius=loiter_r))
        else:
            if dw > 0:
                cmds.append(_loiter_time(lat, lon, alt, seconds=int(dw), radius=loiter_r))
            else:
                # Single orbit at each intermediate stop before continuing
                cmds.append(_loiter_turns(lat, lon, alt, turns=1, radius=loiter_r))

    if end_with_rtl:
        cmds.append(_rtl())

    if end_with_land:
        _refresh_home()
        cmds.append(_do_land_start())
        cmds.append(_nav_land(_home_lat, _home_lon))

    return cmds

# ───────────────────────────────────────────────────────────────
# EXECUTOR THREAD
# ───────────────────────────────────────────────────────────────
def executor_loop():
    while True:
        cmd    = command_queue.get()
        action = cmd.get("action", "")
        try:
            # Nav actions: clear stop_flag unless hold is active
            if action in ("takeoff","execute_plan","fly_direction","nudge",
                          "rtl","land","land_here","set_mode","set_speed",
                          "racetrack","figure_eight","lawnmower"):
                if not _hold_active.is_set():
                    stop_flag.clear()
                mission_state["last_command"] = cmd
                mission_state["phase"] = "executing"

            # ── ARM ──────────────────────────────────────────────────────
            if action == "arm":
                if vehicle.mode.name not in ("MANUAL","STABILIZE","AUTO","TAKEOFF","FBWA"):
                    _set_mode("STABILIZE")
                vehicle.armed = True
                t0 = time.time()
                while not vehicle.armed and time.time()-t0 < 15:
                    vehicle.armed = True
                    time.sleep(0.5)
                _log("arm")
                print("[PLANE] Armed.")
                flog("info", "ARMED")

            # ── DISARM (ground only — hard safety guard) ───────────────────
            elif action == "disarm":
                # NEVER allow disarm in flight — a fixed-wing has no rotors to
                # cut; disarming kills the engine but the plane keeps falling
                # uncontrolled. ArduPlane's own arming-check (DISARM_PITCH etc.)
                # is not guaranteed configured in every SITL setup, so we
                # enforce this in software with a hard altitude/airspeed check.
                cur_alt = vehicle.location.global_relative_frame.alt or 0
                cur_spd = vehicle.airspeed or vehicle.groundspeed or 0
                if cur_alt > 3.0 or cur_spd > 3.0:
                    print(f"[PLANE] [BLOCKED] Cannot disarm — airborne "
                          f"(alt={cur_alt:.1f}m, speed={cur_spd:.1f}m/s). "
                          f"Land first, then disarm on the ground.")
                    flog("warning", f"DISARM BLOCKED — airborne alt={cur_alt:.1f}m speed={cur_spd:.1f}m/s")
                    mission_state["incidents"].append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                        f"disarm blocked — airborne at {cur_alt:.1f}m")
                else:
                    vehicle.armed = False
                    _log("disarm")
                    print("[PLANE] Disarmed (on ground).")
                    flog("info", "DISARMED on ground")

            # ── NUDGE (short obstacle-avoidance move) ──────────────────────
            elif action == "nudge":
                # A deliberately SMALL, SHORT move — for avoiding something in
                # the flight path right now. Unlike fly_direction (which is a
                # full mission leg ending in an orbit), nudge ends in a single
                # NAV_LOITER_TURNS(1) so the plane briefly stabilises and then
                # the operator/agent decides what happens next. This keeps the
                # plane under control instead of defaulting to an indefinite
                # loiter, which was disorienting during avoidance manoeuvres.
                direction = cmd["direction"]   # compass OR 'left'/'right'/'up'/'down'
                dist      = cmd.get("distance", 50)
                alt_delta = cmd.get("altitude_change", 0)
                cur = vehicle.location.global_relative_frame
                cur_alt = cur.alt or 60

                if direction.lower() in ("up", "climb", "higher"):
                    tlat, tlon = cur.lat, cur.lon
                    talt = cur_alt + (dist if dist else 20)
                elif direction.lower() in ("down", "descend", "lower"):
                    tlat, tlon = cur.lat, cur.lon
                    talt = max(30, cur_alt - (dist if dist else 20))
                elif direction.lower() in ("left", "right"):
                    # Relative to current heading (yaw), not compass.
                    yaw_deg = math.degrees(vehicle.attitude.yaw) % 360
                    rel_bearing = (yaw_deg - 90) % 360 if direction.lower() == "left" \
                                  else (yaw_deg + 90) % 360
                    tlat, tlon = _offset_coord(cur.lat, cur.lon, rel_bearing, dist)
                    talt = cur_alt + alt_delta
                else:
                    # Compass direction
                    tlat, tlon = _offset_coord(cur.lat, cur.lon, _bearing(direction), dist)
                    talt = cur_alt + alt_delta

                talt = max(30, min(120, talt))
                nudge_cmds = [_wp(tlat, tlon, talt), _loiter_turns(tlat, tlon, talt, turns=1, radius=120)]
                _upload_mission(nudge_cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Nudging {direction} {dist}m @ {talt:.0f}m — brief stabilise, "
                      f"then say where to go next.")
                flog("info", f"NUDGE {direction} {dist}m -> ({tlat:.5f},{tlon:.5f}) @ {talt:.0f}m")
                _log("nudge", {"direction": direction, "distance": dist, "alt": talt})

            # ── TAKEOFF ──────────────────────────────────────────────────
            elif action == "takeoff":
                alt = cmd.get("altitude", 60)
                # Arm if needed
                if not vehicle.armed:
                    if vehicle.mode.name not in ("MANUAL","STABILIZE","FBWA"):
                        _set_mode("STABILIZE")
                    vehicle.armed = True
                    t0 = time.time()
                    while not vehicle.armed and time.time()-t0 < 15:
                        vehicle.armed = True
                        time.sleep(0.5)
                # Set takeoff altitude parameter
                try:
                    vehicle.parameters["TKOFF_ALT"] = alt
                except Exception as e:
                    flog("warning", f"TKOFF_ALT write: {e}")
                _set_mode("TAKEOFF")
                print(f"[PLANE] TAKEOFF mode — climbing to {alt}m then auto-converts to LOITER.")
                flog("info", f"TAKEOFF — target {alt}m")
                # Wait for 80% of target altitude
                t0 = time.time()
                while time.time()-t0 < 120:
                    if stop_flag.is_set(): break
                    if (vehicle.location.global_relative_frame.alt or 0) >= alt*0.8:
                        print(f"[PLANE] Airborne at ~{round(vehicle.location.global_relative_frame.alt,1)}m.")
                        break
                    time.sleep(1.0)
                _log("takeoff", {"altitude": alt})

            # ── EXECUTE PLAN (main navigation command) ────────────────────
            elif action == "execute_plan":
                end_land = cmd.get("end_with_land", False)
                end_rtl  = cmd.get("end_with_rtl",  False)
                cmds     = _plan_build_commands(end_with_land=end_land, end_with_rtl=end_rtl)

                if not cmds:
                    print("[PLANE] execute_plan: no stops queued.")
                    command_queue.task_done()
                    continue

                _upload_mission(cmds)
                # Reset wp index to beginning for fresh start
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Mission started — {len(cmds)} item(s), "
                      f"{'landing' if end_land else 'RTL' if end_rtl else 'loiter'} at end.")
                flog("info", f"EXECUTE_PLAN: {len(cmds)} items | land={end_land} rtl={end_rtl}")
                _log("execute_plan", {"items": len(cmds),
                                      "land": end_land, "rtl": end_rtl})

                # Wait for mission completion only if we are landing/RTL so the
                # executor thread doesn't exit immediately and lose tracking.
                if end_land or end_rtl:
                    _wait_mission_complete(timeout=600)
                    if not stop_flag.is_set():
                        print("[PLANE] Mission complete.")
                        _log("mission_complete")
                _plan_clear()

            # ── FLY DIRECTION (single offset WP) ─────────────────────────
            elif action == "fly_direction":
                direction = cmd["direction"]
                dist      = cmd["distance"]
                alt       = cmd.get("altitude",
                                    vehicle.location.global_relative_frame.alt or 60)
                end_land  = cmd.get("end_with_land", False)
                end_rtl   = cmd.get("end_with_rtl",  False)
                cur       = vehicle.location.global_relative_frame
                tlat, tlon = _offset_coord(cur.lat, cur.lon, _bearing(direction), dist)

                _plan_clear()
                _plan_add(tlat, tlon, alt, name=f"{direction} {dist}m")
                cmds = _plan_build_commands(end_with_land=end_land, end_with_rtl=end_rtl)
                _upload_mission(cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Flying {direction} {dist}m → ({tlat:.5f},{tlon:.5f}) @ {alt}m.")
                flog("info", f"FLY_DIRECTION {direction} {dist}m -> ({tlat:.5f},{tlon:.5f})")
                _log("fly_direction", {"direction": direction, "dist_m": dist, "alt": alt})
                if end_land or end_rtl:
                    _wait_mission_complete(timeout=600)
                _plan_clear()

            # ── RACETRACK (holding-pattern oval) ──────────────────────────
            elif action == "racetrack":
                cur  = vehicle.location.global_relative_frame
                lat0 = cmd.get("center_lat") if cmd.get("center_lat") is not None else cur.lat
                lon0 = cmd.get("center_lon") if cmd.get("center_lon") is not None else cur.lon
                alt  = cmd.get("altitude") or cur.alt or 60
                cmds, last_lat, last_lon = _build_racetrack(
                    lat0, lon0, alt,
                    heading=cmd.get("heading", "north"),
                    leg_length_m=cmd.get("leg_length_m", 400),
                    turn_radius_m=cmd.get("turn_radius_m", 150),
                    turn_direction=cmd.get("turn_direction", "right"),
                    laps=cmd.get("laps", 3),
                )
                end_land = cmd.get("end_with_land", False)
                end_rtl  = cmd.get("end_with_rtl", False)
                if end_land:
                    _refresh_home()
                    cmds += [_do_land_start(), _nav_land(_home_lat, _home_lon)]
                elif end_rtl:
                    cmds.append(_rtl())
                else:
                    cmds.append(_loiter_unlim(last_lat, last_lon, alt,
                                               radius=cmd.get("turn_radius_m", 150)))
                _upload_mission(cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Racetrack pattern started — {cmd.get('laps',3)} lap(s) "
                      f"@ {alt:.0f}m, heading {cmd.get('heading','north')}.")
                flog("info", f"RACETRACK: laps={cmd.get('laps',3)} heading={cmd.get('heading')} "
                             f"leg={cmd.get('leg_length_m',400)}m radius={cmd.get('turn_radius_m',150)}m")
                _log("racetrack", {"laps": cmd.get("laps", 3), "heading": cmd.get("heading")})
                if end_land or end_rtl:
                    _wait_mission_complete(timeout=600)
                    if not stop_flag.is_set():
                        print("[PLANE] Racetrack + landing/RTL complete.")

            # ── FIGURE EIGHT ───────────────────────────────────────────────
            elif action == "figure_eight":
                cur  = vehicle.location.global_relative_frame
                lat0 = cmd.get("center_lat") if cmd.get("center_lat") is not None else cur.lat
                lon0 = cmd.get("center_lon") if cmd.get("center_lon") is not None else cur.lon
                alt  = cmd.get("altitude") or cur.alt or 60
                cmds, last_lat, last_lon = _build_figure_eight(
                    lat0, lon0, alt,
                    heading=cmd.get("heading", "north"),
                    lobe_radius_m=cmd.get("lobe_radius_m", 150),
                    loops=cmd.get("loops", 2),
                    first_lobe=cmd.get("first_lobe", "right"),
                )
                end_land = cmd.get("end_with_land", False)
                end_rtl  = cmd.get("end_with_rtl", False)
                if end_land:
                    _refresh_home()
                    cmds += [_do_land_start(), _nav_land(_home_lat, _home_lon)]
                elif end_rtl:
                    cmds.append(_rtl())
                else:
                    cmds.append(_loiter_unlim(last_lat, last_lon, alt,
                                               radius=cmd.get("lobe_radius_m", 150)))
                _upload_mission(cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Figure-eight started — {cmd.get('loops',2)} loop(s) "
                      f"@ {alt:.0f}m, heading {cmd.get('heading','north')}.")
                flog("info", f"FIGURE_EIGHT: loops={cmd.get('loops',2)} heading={cmd.get('heading')} "
                             f"lobe_radius={cmd.get('lobe_radius_m',150)}m")
                _log("figure_eight", {"loops": cmd.get("loops", 2), "heading": cmd.get("heading")})
                if end_land or end_rtl:
                    _wait_mission_complete(timeout=600)
                    if not stop_flag.is_set():
                        print("[PLANE] Figure-eight + landing/RTL complete.")

            # ── LAWNMOWER / SURVEY GRID ─────────────────────────────────────
            elif action == "lawnmower":
                cur  = vehicle.location.global_relative_frame
                lat0 = cmd.get("center_lat") if cmd.get("center_lat") is not None else cur.lat
                lon0 = cmd.get("center_lon") if cmd.get("center_lon") is not None else cur.lon
                alt  = cmd.get("altitude") or cur.alt or 60
                cmds, last_lat, last_lon = _build_lawnmower(
                    lat0, lon0, alt,
                    heading=cmd.get("heading", "north"),
                    length_m=cmd.get("length_m", 800),
                    width_m=cmd.get("width_m", 400),
                    track_spacing_m=cmd.get("track_spacing_m", 100),
                )
                end_land = cmd.get("end_with_land", False)
                end_rtl  = cmd.get("end_with_rtl", False)
                if end_land:
                    _refresh_home()
                    cmds += [_do_land_start(), _nav_land(_home_lat, _home_lon)]
                elif end_rtl:
                    cmds.append(_rtl())
                else:
                    cmds.append(_loiter_unlim(last_lat, last_lon, alt))
                _upload_mission(cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print(f"[PLANE] Lawnmower survey started — {cmd.get('length_m',800)}m x "
                      f"{cmd.get('width_m',400)}m grid @ {alt:.0f}m.")
                flog("info", f"LAWNMOWER: length={cmd.get('length_m',800)}m width={cmd.get('width_m',400)}m "
                             f"spacing={cmd.get('track_spacing_m',100)}m heading={cmd.get('heading')}")
                _log("lawnmower", {"length_m": cmd.get("length_m",800), "width_m": cmd.get("width_m",400)})
                if end_land or end_rtl:
                    _wait_mission_complete(timeout=600)
                    if not stop_flag.is_set():
                        print("[PLANE] Lawnmower survey + landing/RTL complete.")

            # ── RTL (return home, loiter — NO landing) ────────────────────
            elif action == "rtl":
                # Upload a single NAV_RETURN_TO_LAUNCH item.
                # ArduPlane will fly home and loiter at RTL_ALTITUDE (default 100m).
                _upload_mission([_rtl()])
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print("[PLANE] RTL — returning home and loitering (no landing).")
                flog("info", "RTL mission uploaded")
                _log("rtl")

            # ── LAND (full landing sequence) ──────────────────────────────
            elif action == "land":
                # DO_LAND_START + NAV_LAND at home coordinates.
                # ArduPlane executes an approach and touches down.
                _refresh_home()
                land_cmds = [_do_land_start(), _nav_land(_home_lat, _home_lon)]
                _upload_mission(land_cmds)
                vehicle.commands.next = 0
                _set_mode("AUTO")
                print("[PLANE] Landing sequence uploaded — approaching and touching down.")
                flog("info", f"LAND at ({_home_lat},{_home_lon})")
                _log("land")
                _wait_mission_complete(timeout=300)
                if not stop_flag.is_set():
                    print("[PLANE] Landed.")
                    mission_state["phase"] = "idle"
                # IMPORTANT: clear the mission immediately after landing.
                # ArduPlane locks arming when a DO_LAND_START mission is stored
                # on the flight controller. Clearing it now means the next
                # arm + takeoff works without the "In landing sequence" error.
                try:
                    _lcmds = vehicle.commands
                    _lcmds.download()
                    _lcmds.wait_ready()
                    _lcmds.clear()
                    _lcmds.upload()
                    flog("info", "Post-land mission cleared — arm lock released")
                    print("[PLANE] Mission cleared — ready to arm again.")
                except Exception as _le:
                    flog("warning", f"Post-land mission clear failed: {_le}")

            # ── LAND HERE (emergency — checks current position is safe) ────
            elif action == "land_here":
                # Unlike land(), this lands at the plane's CURRENT position,
                # not home. Used for emergencies where flying back home isn't
                # an option. We verify the spot is safe first — see
                # check_landing_safety() — and refuse + suggest the nearest
                # known safe zone if it's too close to a known hazard.
                cur = vehicle.location.global_relative_frame
                force = cmd.get("force", False)
                result = check_landing_safety(cur.lat, cur.lon)

                if not result["safe"] and not force:
                    print(f"[PLANE] [BLOCKED] Cannot land here — {result['reason']}")
                    if result["nearest_safe_zone"]:
                        print(f"           Nearest known safe zone: "
                              f"'{result['nearest_safe_zone']}' "
                              f"({result['nearest_safe_zone_dist_m']}m away). "
                              f"Say 'land at {result['nearest_safe_zone']}' or "
                              f"'land here anyway' to override.")
                    flog("warning", f"LAND_HERE blocked at ({cur.lat:.5f},{cur.lon:.5f}): {result['reason']}")
                    mission_state["incidents"].append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                        f"land_here blocked — {result['reason']}")
                else:
                    if not result["safe"] and force:
                        print(f"[PLANE] WARNING — landing here despite safety check: {result['reason']}")
                        flog("warning", f"LAND_HERE FORCED at ({cur.lat:.5f},{cur.lon:.5f}): {result['reason']}")
                    land_cmds = [_do_land_start(), _nav_land(cur.lat, cur.lon)]
                    _upload_mission(land_cmds)
                    vehicle.commands.next = 0
                    _set_mode("AUTO")
                    print(f"[PLANE] Emergency landing at current position "
                          f"({cur.lat:.5f},{cur.lon:.5f}) — approaching and touching down.")
                    flog("info", f"LAND_HERE at ({cur.lat:.5f},{cur.lon:.5f}) | safety={result['reason']}")
                    _log("land_here", {"lat": cur.lat, "lon": cur.lon, "safety": result["reason"]})
                    _wait_mission_complete(timeout=300)
                    if not stop_flag.is_set():
                        print("[PLANE] Landed.")
                        mission_state["phase"] = "idle"
                    # Same post-land arm-lock fix as the regular land action
                    try:
                        _lcmds2 = vehicle.commands
                        _lcmds2.download()
                        _lcmds2.wait_ready()
                        _lcmds2.clear()
                        _lcmds2.upload()
                        flog("info", "Post-land (land_here) mission cleared — arm lock released")
                        print("[PLANE] Mission cleared — ready to arm again.")
                    except Exception as _le2:
                        flog("warning", f"Post-land_here mission clear failed: {_le2}")

            # ── HOLD (LOITER at current position) ─────────────────────────
            elif action == "hold":
                _hold_active.set()
                stop_flag.set()
                _set_mode("LOITER")
                cur = vehicle.location.global_relative_frame
                print(f"[PLANE] LOITER — orbiting ({round(cur.lat,5)},{round(cur.lon,5)}) "
                      f"@ {round(cur.alt,1)}m.")
                flog("info", f"HOLD/LOITER @ {round(cur.alt,1)}m")
                _log("hold")

            # ── RESUME (back to AUTO) ─────────────────────────────────────
            elif action == "resume":
                _hold_active.clear()
                stop_flag.clear()
                _set_mode("AUTO")
                print("[PLANE] Resumed — AUTO mode, continuing mission.")
                flog("info", "RESUME -> AUTO")
                _log("resume")

            # ── SET SPEED ─────────────────────────────────────────────────
            elif action == "set_speed":
                speed = cmd.get("speed", _SPEED_CRZ)
                applied = _set_speed(speed)
                _log("set_speed", {"airspeed_ms": applied})
                print(f"[PLANE] Airspeed → {applied:.1f} m/s")

            # ── SET MODE ──────────────────────────────────────────────────
            elif action == "set_mode":
                allowed = {"LOITER","AUTO","RTL","MANUAL","STABILIZE",
                           "FBWA","FBWB","CRUISE","TAKEOFF"}
                name = cmd.get("mode","LOITER").upper()
                if name not in allowed:
                    print(f"[PLANE] Mode '{name}' not allowed.")
                elif name in ("MANUAL", "STABILIZE") and vehicle.armed:
                    # MANUAL/STABILIZE hand full control to RC sticks. In SITL
                    # (and most operator setups) there is no active RC input,
                    # so switching to these modes in flight means zero throttle
                    # and zero control surface deflection — an uncontrolled
                    # crash. Block it while airborne; only allow on the ground
                    # or when explicitly forced.
                    cur_alt = vehicle.location.global_relative_frame.alt or 0
                    if cur_alt > 3.0 and not cmd.get("force", False):
                        print(f"[PLANE] [BLOCKED] Refusing to switch to {name} while "
                              f"airborne (alt={cur_alt:.1f}m) — no active RC input means "
                              f"loss of control. Use 'land' to bring it down safely instead.")
                        flog("warning", f"SET_MODE {name} BLOCKED — airborne at {cur_alt:.1f}m")
                        mission_state["incidents"].append(
                            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                            f"set_mode({name}) blocked — airborne at {cur_alt:.1f}m")
                    else:
                        _set_mode(name)
                        _log("set_mode", {"mode": name})
                        print(f"[PLANE] Mode → {name}")
                else:
                    _set_mode(name)
                    _log("set_mode", {"mode": name})
                    print(f"[PLANE] Mode → {name}")

        except Exception as e:
            err = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {action}: {e}"
            mission_state["incidents"].append(err)
            print(f"[EXECUTOR ERROR] {e}")
            flog("error", f"EXECUTOR: {e}")

        command_queue.task_done()

threading.Thread(target=executor_loop, daemon=True).start()

# ───────────────────────────────────────────────────────────────
# LIVE CONTEXT  (fed to agent before every command)
# ───────────────────────────────────────────────────────────────
def get_live_context() -> str:
    try:
        loc  = vehicle.location.global_relative_frame
        att  = vehicle.attitude
        batt = vehicle.battery
        wp   = getattr(vehicle, "wp_dist", None)
        eta  = getattr(vehicle, "wp_eta",  None)
        plan_summary = ""
        with _plan_lock:
            if _plan_stops:
                names = [s.get("name") or f"({s['lat']:.4f},{s['lon']:.4f})"
                         for s in _plan_stops]
                plan_summary = f"\nQueued stops: {' -> '.join(names)}"
        return (
            f"Mode: {vehicle.mode.name} | Armed: {vehicle.armed}\n"
            f"GPS: lat={round(loc.lat,6)} lon={round(loc.lon,6)} alt={round(loc.alt,1)}m\n"
            f"Attitude: roll={round(math.degrees(att.roll),1)} "
            f"pitch={round(math.degrees(att.pitch),1)} "
            f"yaw={round(math.degrees(att.yaw),1)}\n"
            f"Airspeed: {round(vehicle.airspeed,1)}m/s (measured) | "
            f"Target: {_target_airspeed:.1f}m/s | "
            f"Groundspeed: {round(vehicle.groundspeed,1)}m/s\n"
            f"WP dist: {f'{wp:.0f}m' if wp is not None else 'N/A'} | "
            f"ETA: {f'{eta}s' if eta is not None else 'N/A'}\n"
            f"Battery: {batt.level}% | {batt.voltage}V\n"
            f"Phase: {mission_state['phase']}{plan_summary}"
        )
    except Exception:
        return "[telemetry unavailable]"

def _safety_check(altitude=None):
    if altitude is not None:
        if altitude > 120: return False, f"{altitude}m exceeds 120m AGL limit."
        if altitude < 30:  return False, f"{altitude}m below 30m minimum."
    return True, "ok"

# ───────────────────────────────────────────────────────────────
# PLANE TOOLKIT  —  tools exposed to the AI agent
# ───────────────────────────────────────────────────────────────
class PlaneToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="plane_toolkit", tools=[
            # Lifecycle
            self.arm_plane, self.disarm_plane, self.takeoff, self.return_home, self.land,
            self.land_here, self.check_safe_to_land,
            self.loiter_here, self.resume_flight,
            # Mission building
            self.add_stop, self.add_stop_by_name, self.execute_plan, self.clear_plan,
            # Single-shot nav
            self.fly_direction, self.nudge,
            self.fly_racetrack, self.fly_figure_eight, self.fly_lawnmower,
            # Utilities
            self.get_location, self.get_mission, self.set_speed, self.set_flight_mode,
            # Status
            self.get_status, self.get_position, self.get_flight_summary,
            # Conditions
            self.watch_condition, self.clear_conditions, self.list_conditions,
        ])

    # ── Lifecycle ────────────────────────────────────────────────

    def arm_plane(self) -> str:
        "Arm the plane. Call before takeoff if not already armed."
        command_queue.put({"action": "arm"})
        return "arm_plane queued."

    def disarm_plane(self) -> str:
        """
        Disarm the plane. ONLY works when the plane is on the ground
        (altitude <= 3m and speed <= 3m/s) — this is enforced as a hard
        safety check. If the plane is airborne, this will be BLOCKED and
        you must use land() first, then disarm_plane() once it has touched down.
        NEVER attempt to disarm or switch to MANUAL/STABILIZE mode as a way
        to stop a plane in flight — that causes an uncontrolled crash.
        """
        command_queue.put({"action": "disarm"})
        return ("disarm_plane queued — will only succeed if the plane is on the "
                "ground. If it's airborne, this will be blocked and you should "
                "call land() instead.")

    def takeoff(self, altitude: float = 60) -> str:
        "Take off to altitude metres (min 30, max 120). Plane climbs then auto-converts to LOITER."
        ok, msg = _safety_check(altitude)
        if not ok: return f"[BLOCKED] {msg}"
        command_queue.put({"action": "takeoff", "altitude": altitude})
        return f"takeoff({altitude}m) queued."

    def return_home(self) -> str:
        """
        Fly back to the launch point and LOITER there. Does NOT land.
        Use land() if you want an actual touchdown.
        Equivalent to 'go home', 'return', 'RTL'.
        """
        command_queue.put({"action": "rtl"})
        return "return_home queued — plane will fly home and orbit (no landing)."

    def land(self) -> str:
        """
        Upload a proper landing sequence (DO_LAND_START + NAV_LAND) and execute it.
        The plane will approach home and touch down. Use this for 'land', 'touch down',
        'full stop landing'. Do NOT use return_home() if the user wants to land.
        """
        command_queue.put({"action": "land"})
        return "land queued — plane will execute approach and touch down at home."

    def check_safe_to_land(self, latitude: float = None, longitude: float = None) -> str:
        """
        Check whether a location is safe to land at. If latitude/longitude are
        omitted, checks the plane's CURRENT position. Returns whether the spot
        is clear of known hazards (hospital, prison, residential areas) and,
        if not, the nearest known pre-approved safe zone.
        Call this BEFORE land_here() when the operator wants to land somewhere
        other than home, especially in an emergency.
        """
        if latitude is None or longitude is None:
            cur = vehicle.location.global_relative_frame
            latitude, longitude = cur.lat, cur.lon
        result = check_landing_safety(latitude, longitude)
        if result["safe"]:
            return (f"SAFE to land at ({latitude:.5f},{longitude:.5f}): {result['reason']}")
        else:
            return (f"NOT SAFE to land at ({latitude:.5f},{longitude:.5f}): {result['reason']} "
                    f"Nearest safe zone: '{result['nearest_safe_zone']}' "
                    f"({result['nearest_safe_zone_dist_m']}m away).")

    def land_here(self, force: bool = False) -> str:
        """
        EMERGENCY LANDING — land at the plane's CURRENT position instead of
        flying back home. Use when the operator says 'land here', 'land now',
        'emergency landing', or similar — when returning home is not an option.

        Before landing, this automatically checks the current position against
        known hazards (hospital, prison, residential areas). If unsafe, the
        landing is REFUSED and the nearest known safe zone is suggested instead
        — unless force=True is explicitly set (only use force=True if the
        operator explicitly overrides with something like 'land here anyway'
        or 'I understand the risk, land now').
        """
        command_queue.put({"action": "land_here", "force": force})
        return ("land_here queued — checking current position is safe before "
                "touching down. Will refuse and suggest an alternative if unsafe, "
                "unless force=True.")

    def loiter_here(self) -> str:
        """
        Switch to LOITER mode immediately — plane orbits its current position.
        Use for: stop, hold, wait, pause, loiter, freeze, orbit here.
        Say resume_flight() or 'resume' to continue the mission.
        """
        _hold_active.set()
        stop_flag.set()
        _clear_queue()
        command_queue.put({"action": "hold"})
        n = len(mission_state["pending_mission"])
        return f"loiter_here queued — plane will orbit current position. {n} saved command(s). Say 'resume' to continue."

    def resume_flight(self) -> str:
        """
        Resume AUTO mode after a loiter/hold, continuing the current mission.
        """
        last    = mission_state.get("last_command", {})
        pending = mission_state.get("pending_mission", [])
        skip    = {"hold","loiter_here","set_speed","set_mode"}
        if last.get("action") in skip: last = {}
        if not last and not pending:
            # Just switch back to AUTO with whatever mission is loaded
            _hold_active.clear()
            stop_flag.clear()
            command_queue.put({"action": "resume"})
            return "resume_flight queued — switching back to AUTO."
        _hold_active.clear()
        stop_flag.clear()
        queued = []
        if last:
            command_queue.put(last)
            queued.append(last.get("action","?"))
        for c in pending:
            command_queue.put(c)
            queued.append(c.get("action","?"))
        mission_state["pending_mission"] = []
        return f"resume_flight: re-queuing {' -> '.join(queued)}."

    # ── Mission building ─────────────────────────────────────────

    def add_stop(self, latitude: float, longitude: float,
                 altitude: float = 60, dwell_seconds: float = 0,
                 name: str = "") -> str:
        """
        Add a GPS coordinate as the next stop in the mission plan.
        dwell_seconds > 0 makes the plane orbit that point for that many seconds
        before continuing to the next stop (uses NAV_LOITER_TIME).
        dwell_seconds = 0 means 2 orbits then continue (default survey pass).
        Call execute_plan() after adding all stops.
        """
        ok, msg = _safety_check(altitude)
        if not ok: return f"[BLOCKED] {msg}"
        _plan_add(latitude, longitude, altitude, dwell_s=dwell_seconds, name=name or f"({latitude:.4f},{longitude:.4f})")
        with _plan_lock:
            n = len(_plan_stops)
        return f"add_stop: ({latitude:.5f},{longitude:.5f}) @ {altitude}m added. Plan now has {n} stop(s)."

    def add_stop_by_name(self, location_name: str,
                         altitude: float = 60, dwell_seconds: float = 0) -> str:
        """
        Add a named preset location as the next stop.
        Use for: hospital, prison, camp a, camp b, airfield, home, etc.
        Call execute_plan() after adding all stops.
        """
        key = location_name.lower().strip()
        matches = [k for k in PRESET_LOCATIONS if key in k or k in key]
        if not matches:
            return f"Unknown location '{location_name}'. Available: {', '.join(PRESET_LOCATIONS)}"
        ok, msg = _safety_check(altitude)
        if not ok: return f"[BLOCKED] {msg}"
        loc = PRESET_LOCATIONS[matches[0]]
        _plan_add(loc["lat"], loc["lon"], altitude,
                  dwell_s=dwell_seconds, name=matches[0])
        with _plan_lock:
            n = len(_plan_stops)
        return f"add_stop_by_name: '{matches[0]}' @ {altitude}m added. Plan has {n} stop(s)."

    def execute_plan(self, end_with_land: bool = False,
                     end_with_rtl: bool = False) -> str:
        """
        Upload and execute the queued mission plan.

        end_with_land = True  → after last stop: approach and LAND at home.
        end_with_rtl  = True  → after last stop: fly home and LOITER (no landing).
        Both False            → orbit the last stop indefinitely (default).

        ALWAYS call this after add_stop / add_stop_by_name.
        Do NOT call execute_plan() with end_with_land=True if the user only said
        'return home' — use end_with_rtl=True for that, or return_home() directly.
        """
        with _plan_lock:
            n = len(_plan_stops)
        if n == 0:
            return "[BLOCKED] No stops queued. Call add_stop or add_stop_by_name first."
        command_queue.put({
            "action": "execute_plan",
            "end_with_land": end_with_land,
            "end_with_rtl":  end_with_rtl,
        })
        suffix = " → LAND" if end_with_land else " → RTL/loiter home" if end_with_rtl else ""
        return f"execute_plan queued: {n} stop(s){suffix}. Mission uploading and starting AUTO."

    def clear_plan(self) -> str:
        "Clear all queued stops without executing. Use to start over."
        _plan_clear()
        return "Plan cleared."

    # ── Single-shot nav ──────────────────────────────────────────

    def fly_direction(self, direction: str, distance_m: float,
                      altitude: float = None,
                      end_with_rtl: bool = False,
                      end_with_land: bool = False) -> str:
        """
        Fly in a compass direction for distance_m metres.
        Directions: north, south, east, west, northeast, northwest, southeast, southwest.
        Plane will loiter at the destination unless end_with_rtl or end_with_land is set.
        """
        cur_alt = altitude or vehicle.location.global_relative_frame.alt or 60
        ok, msg = _safety_check(cur_alt)
        if not ok: return f"[BLOCKED] {msg}"
        command_queue.put({
            "action":       "fly_direction",
            "direction":    direction,
            "distance":     distance_m,
            "altitude":     cur_alt,
            "end_with_rtl": end_with_rtl,
            "end_with_land":end_with_land,
        })
        return f"fly_direction({direction}, {distance_m}m) queued."

    def nudge(self, direction: str, distance_m: float = 50,
              altitude_change: float = 0) -> str:
        """
        Make a SHORT, IMMEDIATE move to avoid an obstacle or adjust position
        RIGHT NOW, without committing to a full mission or an indefinite loiter.

        direction: compass (north/south/east/west/etc), OR relative to the
                   plane's CURRENT HEADING ('left', 'right'), OR vertical
                   ('up'/'climb', 'down'/'descend').
        distance_m: how far to move (default 50m). For up/down this is the
                   altitude change in metres (default 20m if not specified).
        altitude_change: optional altitude adjustment when nudging left/right.

        Use this instead of loiter_here() when the goal is to steer around
        something and keep moving — not to stop and hold position.
        The plane briefly stabilises after the nudge; follow up with another
        command (continue original heading, fly_direction, execute_plan, etc.)
        """
        ok, msg = _safety_check(None)  # altitude validated dynamically in executor
        command_queue.put({
            "action":          "nudge",
            "direction":       direction,
            "distance":        distance_m,
            "altitude_change": altitude_change,
        })
        return f"nudge({direction}, {distance_m}m) queued — brief avoidance manoeuvre."

    def fly_racetrack(self, heading: str, leg_length_m: float = 400,
                      turn_radius_m: float = 150, turn_direction: str = "right",
                      laps: int = 3, altitude: float = None,
                      location_name: str = None,
                      end_with_rtl: bool = False, end_with_land: bool = False) -> str:
        """
        Fly a racetrack / holding-pattern oval: two straight legs joined by
        two turns, like an aviation holding pattern. Use for 'racetrack',
        'holding pattern', 'oval', 'orbit back and forth', or loitering over
        a stretched-out area (e.g. a road or ridge) rather than a tight circle.

        heading: compass direction of the OUTBOUND leg (e.g. 'north', 90).
        leg_length_m: length of each straight leg (default 400m).
        turn_radius_m: radius of the two end turns (default 150m).
        turn_direction: 'right' (clockwise, standard) or 'left' (counter-clockwise).
        laps: number of full ovals to fly before ending (default 3).
        location_name: optional preset location to centre the pattern on;
                       defaults to the plane's current position.
        """
        alt = altitude or vehicle.location.global_relative_frame.alt or 60
        ok, msg = _safety_check(alt)
        if not ok: return f"[BLOCKED] {msg}"
        cmd = {
            "action": "racetrack", "heading": heading,
            "leg_length_m": leg_length_m, "turn_radius_m": turn_radius_m,
            "turn_direction": turn_direction, "laps": laps, "altitude": alt,
            "end_with_rtl": end_with_rtl, "end_with_land": end_with_land,
        }
        if location_name:
            key = location_name.lower().strip()
            matches = [k for k in PRESET_LOCATIONS if key in k or k in key]
            if not matches:
                return f"Unknown location '{location_name}'. Available: {', '.join(PRESET_LOCATIONS)}"
            loc = PRESET_LOCATIONS[matches[0]]
            cmd["center_lat"], cmd["center_lon"] = loc["lat"], loc["lon"]
        command_queue.put(cmd)
        return f"fly_racetrack({heading}, {laps} lap(s)) queued."

    def fly_figure_eight(self, heading: str, lobe_radius_m: float = 150,
                         loops: int = 2, first_lobe: str = "right",
                         altitude: float = None, location_name: str = None,
                         end_with_rtl: bool = False, end_with_land: bool = False) -> str:
        """
        Fly a figure-eight: two tangent orbit lobes, one clockwise and one
        counter-clockwise, crossing at a shared centre point. Use for
        'figure eight', 'figure 8', 'S-turns', or when the operator wants
        to cover both sides of a line/road/border without banking the
        same direction the whole time.

        heading: compass direction of the AXIS connecting the two lobes.
        lobe_radius_m: radius of each lobe (default 150m).
        loops: number of full figure-eights to fly (default 2).
        first_lobe: which lobe is flown clockwise first ('right' or 'left').
        location_name: optional preset location for the crossing point;
                       defaults to the plane's current position.
        """
        alt = altitude or vehicle.location.global_relative_frame.alt or 60
        ok, msg = _safety_check(alt)
        if not ok: return f"[BLOCKED] {msg}"
        cmd = {
            "action": "figure_eight", "heading": heading,
            "lobe_radius_m": lobe_radius_m, "loops": loops,
            "first_lobe": first_lobe, "altitude": alt,
            "end_with_rtl": end_with_rtl, "end_with_land": end_with_land,
        }
        if location_name:
            key = location_name.lower().strip()
            matches = [k for k in PRESET_LOCATIONS if key in k or k in key]
            if not matches:
                return f"Unknown location '{location_name}'. Available: {', '.join(PRESET_LOCATIONS)}"
            loc = PRESET_LOCATIONS[matches[0]]
            cmd["center_lat"], cmd["center_lon"] = loc["lat"], loc["lon"]
        command_queue.put(cmd)
        return f"fly_figure_eight({heading}, {loops} loop(s)) queued."

    def fly_lawnmower(self, heading: str, length_m: float = 800,
                      width_m: float = 400, track_spacing_m: float = 100,
                      altitude: float = None, location_name: str = None,
                      end_with_rtl: bool = False, end_with_land: bool = False) -> str:
        """
        Fly a lawnmower / boustrophedon survey grid: parallel back-and-forth
        legs covering a rectangular area. Use for 'survey', 'search pattern',
        'lawnmower', 'mow the lawn', 'sweep the area', or systematic area
        coverage requests (not a single-point loiter).

        heading: compass direction of the LONG survey legs.
        length_m: length of each leg (the long dimension of the area).
        width_m: total width of the area to cover (the short dimension).
        track_spacing_m: distance between adjacent parallel legs — smaller
                         spacing means denser coverage, more legs.
        location_name: optional preset location for the area centre;
                       defaults to the plane's current position.
        """
        alt = altitude or vehicle.location.global_relative_frame.alt or 60
        ok, msg = _safety_check(alt)
        if not ok: return f"[BLOCKED] {msg}"
        cmd = {
            "action": "lawnmower", "heading": heading,
            "length_m": length_m, "width_m": width_m,
            "track_spacing_m": track_spacing_m, "altitude": alt,
            "end_with_rtl": end_with_rtl, "end_with_land": end_with_land,
        }
        if location_name:
            key = location_name.lower().strip()
            matches = [k for k in PRESET_LOCATIONS if key in k or k in key]
            if not matches:
                return f"Unknown location '{location_name}'. Available: {', '.join(PRESET_LOCATIONS)}"
            loc = PRESET_LOCATIONS[matches[0]]
            cmd["center_lat"], cmd["center_lon"] = loc["lat"], loc["lon"]
        command_queue.put(cmd)
        return f"fly_lawnmower({heading}, {length_m}x{width_m}m) queued."

    # ── Utilities ────────────────────────────────────────────────

    def get_location(self, name: str) -> str:
        "Return coordinates of a named preset location."
        key = name.lower().strip()
        matches = [k for k in PRESET_LOCATIONS if key in k or k in key]
        if not matches:
            return f"Unknown '{name}'. Available: {', '.join(PRESET_LOCATIONS)}"
        loc = PRESET_LOCATIONS[matches[0]]
        return f"{matches[0]}: {loc['description']} — lat={loc['lat']}, lon={loc['lon']}"

    def get_mission(self) -> str:
        "Show the currently uploaded mission waypoints and current WP index."
        wps = _download_mission()
        if not wps: return "No mission uploaded."
        lines = [f"Uploaded mission ({len(wps)} items), current WP: {vehicle.commands.next}"]
        for w in wps:
            lines.append(f"  [{w['index']}] {w['command']} "
                         f"lat={w['lat']} lon={w['lon']} alt={w['alt']}m")
        return "\n".join(lines)

    def set_speed(self, speed_ms: float) -> str:
        "Set target airspeed in m/s (9–22). Applied immediately."
        if not (_SPEED_MIN <= speed_ms <= _SPEED_MAX):
            return f"[BLOCKED] {speed_ms} m/s out of range (9–22 m/s)."
        applied = _set_speed(speed_ms)
        return f"Airspeed set to {applied:.1f} m/s."

    def set_flight_mode(self, mode: str) -> str:
        """
        Switch ArduPlane mode: LOITER/AUTO/RTL/MANUAL/STABILIZE/FBWA/FBWB/CRUISE.
        SAFETY: MANUAL and STABILIZE hand control to the RC sticks. If the
        plane is airborne and you request either of these, the command will
        be BLOCKED automatically — there is no active pilot input in this
        simulation, so it would cause an uncontrolled crash. Use land() to
        bring the plane down instead.
        """
        allowed = {"LOITER","AUTO","RTL","MANUAL","STABILIZE","FBWA","FBWB","CRUISE","TAKEOFF"}
        m = mode.upper()
        if m not in allowed:
            return f"[BLOCKED] '{m}' not valid. Use: {', '.join(sorted(allowed))}"
        command_queue.put({"action": "set_mode", "mode": m})
        return f"set_flight_mode({m}) queued. (MANUAL/STABILIZE will be auto-blocked if airborne.)"

    # ── Status ───────────────────────────────────────────────────

    def get_status(self) -> str:
        "Full live plane telemetry."
        return get_live_context()

    def get_position(self) -> str:
        "Current GPS position and local NED frame."
        loc = vehicle.location.global_relative_frame
        ll  = vehicle.location.local_frame
        return (f"GPS: {round(loc.lat,6)}, {round(loc.lon,6)} @ {round(loc.alt,1)}m\n"
                f"Local: N={round(ll.north,2)}m E={round(ll.east,2)}m D={round(ll.down,2)}m")

    def get_flight_summary(self) -> str:
        "Session flight log."
        log = mission_state["flight_log"]
        if not log: return "No flight activity yet."
        lines = ["--- Flight Log ---"]
        for i,e in enumerate(log,1):
            det = ", ".join(f"{k}={v}" for k,v in e["details"].items()) if e["details"] else ""
            lines.append(f"{i}. [{e['time']}] {e['action'].upper()} {det} @ {e['alt']}m")
        lines.append(f"Max altitude: {round(mission_state['max_altitude'],1)}m")
        return "\n".join(lines)

    # ── Conditions ───────────────────────────────────────────────

    def watch_condition(self, field: str, operator: str, value: float,
                        then_action: str, then_params: str = "") -> str:
        "Register a background trigger. Fields: rel_alt/airspeed/groundspeed/mode/airborne/yaw."
        if field not in CONDITION_FIELDS:
            return f"[BLOCKED] Unknown field '{field}'."
        if operator not in _OPS:
            return f"[BLOCKED] Unknown operator '{operator}'."
        params = {}
        if then_params:
            try: params = json.loads(then_params)
            except Exception: return "[BLOCKED] then_params must be valid JSON."
        label = f"{field} {operator} {value} -> {then_action}"
        cond_monitor.add(_Watch(field, operator, value, then_action, params, label))
        return f"Condition registered: {label}"

    def clear_conditions(self) -> str:
        "Remove all active condition watches."
        cond_monitor.clear()
        return "All conditions cleared."

    def list_conditions(self) -> str:
        "List active condition watches."
        return cond_monitor.list_all()

# ───────────────────────────────────────────────────────────────
# SESSION DB + MODEL
# ───────────────────────────────────────────────────────────────
agent_db       = SqliteDb(db_file="plane_sessions.db")
_NO_THINK      = {"chat_template_kwargs": {"enable_thinking": False}}
ACTIVE_MODEL   = OpenRouter(id="google/gemini-3.1-flash-lite", extra_body=_NO_THINK)
_all_agents    = []

# ───────────────────────────────────────────────────────────────
# SYSTEM MESSAGE
# ───────────────────────────────────────────────────────────────
def _system_message():
    locs = "\n".join(
        f"  {k}: {v['description']} (lat={v['lat']}, lon={v['lon']})"
        for k,v in PRESET_LOCATIONS.items())
    return f"""/nothink
You control a real fixed-wing aircraft (ArduPlane) in SITL simulation.
This is NOT a quadcopter. The plane CANNOT hover — it orbits waypoints.

═══ CRITICAL RULES ═══════════════════════════════════════════════════
1. Call ALL required tools in correct order before responding.
2. NEVER stop after just one tool call for a multi-step request.
3. Respond with ONE short sentence after all tools are called.
4. NEVER ask for clarification — infer intent and execute.
5. Battery is fake in SITL — never mention it.
6. NEVER call get_status to monitor — DO NOT POLL.
7. The queue is async — commands run in background in order.

═══ MISSION BUILDING (most important) ════════════════════════════════
For any route with destinations, use this pattern:

  Single destination (loiter there):
    add_stop_by_name("hospital", altitude=60)
    execute_plan()                          ← loiters at hospital forever

  Single destination then return home (NO landing):
    add_stop_by_name("hospital", altitude=60)
    execute_plan(end_with_rtl=True)         ← orbits hospital then flies home

  Single destination then LAND:
    add_stop_by_name("hospital", altitude=60)
    execute_plan(end_with_land=True)        ← orbits hospital then lands

  Multi-stop then return home:
    add_stop_by_name("hospital", altitude=60)
    add_stop_by_name("prison",   altitude=70)
    execute_plan(end_with_rtl=True)

  Multi-stop then land:
    add_stop_by_name("camp a",   altitude=60)
    add_stop_by_name("camp b",   altitude=60)
    execute_plan(end_with_land=True)

  Dwell at a stop (orbit for N seconds):
    add_stop_by_name("hospital", altitude=60, dwell_seconds=120)
    add_stop_by_name("prison",   altitude=70)
    execute_plan(end_with_rtl=True)

RULE: ALWAYS call execute_plan() after add_stop / add_stop_by_name.
RULE: 'return home' / 'go back' / 'RTL' = execute_plan(end_with_rtl=True) OR return_home().
RULE: 'land' / 'touch down' / 'full stop' (no urgency, default) = execute_plan(end_with_land=True) OR land().
RULE: 'land here' / 'land now' / 'emergency landing' / 'land at my position' / land
      requests where flying back home is NOT mentioned or NOT desired = land_here().
      land_here() lands at the CURRENT position, NOT home, and automatically
      checks the spot is safe first (see SAFETY — LANDING below).
RULE: 'go to X' with no further instruction = add_stop_by_name + execute_plan() (no rtl, no land).
RULE: if the plane needs to take off first, call arm_plane() + takeoff() BEFORE any navigation.

═══ SAFETY — LANDING (land_here checks the spot first) ═════════════════
land_here() automatically calls a safety check before touching down:
  - Refuses to land within 200m of a known hazard (hospital, prison,
    residential areas) and instead tells the operator the nearest known
    pre-approved safe zone (home, airfield, runway 35/17, reserve).
  - If the operator explicitly overrides ('land here anyway', 'I understand
    the risk', 'force land'), call land_here(force=True).
  - You can check a spot in advance with check_safe_to_land(lat, lon) or
    check_safe_to_land() with no args to check the CURRENT position.
  - Pre-approved safe zones (home/airfield/runway 35/runway 17/reserve)
    always pass the check immediately — no need to double-check those.
  - This is a software check using KNOWN named locations only — there is no
    live terrain or obstacle sensing in this simulation. Treat 'not safe'
    results as informational guidance, and 'safe' results as 'no known
    hazard nearby', not as a guarantee of clear terrain.

═══ HOLD / RESUME ════════════════════════════════════════════════════
'stop' / 'hold' / 'loiter here' / 'wait' → loiter_here()
'resume' / 'continue' / 'carry on'        → resume_flight()

═══ OBSTACLE AVOIDANCE / SHORT MANOEUVRES ══════════════════════════════
For requests like 'go left to avoid that', 'go up a bit', 'move right',
'climb to avoid the hill' — use nudge(), NOT loiter_here().
  nudge('left', 80)             steer left of current heading 80m, then continue
  nudge('up', 20)                climb 20m, then continue
  nudge('east', 100, altitude_change=10)   move + climb together
nudge() does a brief single-orbit stabilise then waits for the next command —
it does NOT loiter indefinitely. Use loiter_here() only when the operator
explicitly wants to stop and hold position, not to dodge something while
continuing onward.

═══ NON-STANDARD ORBIT PATTERNS ══════════════════════════════════════
Use these instead of a plain loiter/execute_plan when the operator wants a
SHAPE, not just a point to orbit:

  'racetrack' / 'holding pattern' / 'oval' / loiter over something elongated:
    fly_racetrack(heading='north', leg_length_m=400, turn_radius_m=150,
                   turn_direction='right', laps=3)

  'figure eight' / 'figure 8' / S-turns / cover both sides of a line:
    fly_figure_eight(heading='north', lobe_radius_m=150, loops=2,
                      first_lobe='right')

  'survey' / 'search pattern' / 'lawnmower' / 'sweep the area':
    fly_lawnmower(heading='north', length_m=800, width_m=400,
                  track_spacing_m=100)

All three accept location_name (defaults to current position),
end_with_rtl, and end_with_land — same semantics as execute_plan().
RULE: a single named destination = add_stop_by_name + execute_plan (existing).
RULE: an elongated/shaped orbit over a point = fly_racetrack or fly_figure_eight.
RULE: area coverage / survey language = fly_lawnmower.

═══ SAFETY — DISARM AND MANUAL MODE (HARD RULES) ═══════════════════════
NEVER call disarm_plane() or set_flight_mode('MANUAL'/'STABILIZE') while
the plane is airborne. Both are blocked in software, but you must not even
attempt them as a way to 'stop' the plane — there is no active RC pilot
input in this simulation, so cutting the engine or releasing control to
MANUAL mid-flight causes an immediate uncontrolled crash.
  'stop the engine' / 'disarm' / 'kill it' while flying →
      explain you cannot disarm in flight, and offer land() instead.
  The ONLY safe way to stop a flying plane is loiter_here() (hold position)
  or land() (full landing sequence). disarm_plane() only works on the ground
  after landing.

═══ SPEED ════════════════════════════════════════════════════════════
Controls AIRSPEED via TRIM_ARSPD_CM. Range: 9–22 m/s. Cruise: 15 m/s.
Use set_speed(value) or instant modifiers in the input router.

═══ ALTITUDE ═════════════════════════════════════════════════════════
Min 30m AGL. Max 120m AGL. Default cruise 60m.
WP_LOITER_RAD = 300m (arrival threshold — plane orbits within this radius).

═══ NAMED LOCATIONS ══════════════════════════════════════════════════
{locs}

═══ TOOLS ════════════════════════════════════════════════════════════
arm_plane() | disarm_plane() | takeoff(alt) | return_home() | land()
land_here(force) | check_safe_to_land(lat,lon)
loiter_here() | resume_flight()
add_stop(lat,lon,alt,dwell_seconds) | add_stop_by_name(name,alt,dwell_seconds)
execute_plan(end_with_land,end_with_rtl) | clear_plan()
fly_direction(direction,distance_m,altitude,end_with_rtl,end_with_land)
nudge(direction,distance_m,altitude_change)
fly_racetrack(heading,leg_length_m,turn_radius_m,turn_direction,laps,location_name,end_with_rtl,end_with_land)
fly_figure_eight(heading,lobe_radius_m,loops,first_lobe,location_name,end_with_rtl,end_with_land)
fly_lawnmower(heading,length_m,width_m,track_spacing_m,location_name,end_with_rtl,end_with_land)
get_location(name) | get_mission() | set_speed(ms) | set_flight_mode(mode)
get_status() | get_position() | get_flight_summary()
watch_condition(field,op,val,action) | clear_conditions() | list_conditions()
list_missions() | read_mission(f) | save_report(content,f) | list_reports()
"""

# ───────────────────────────────────────────────────────────────
# LEARNING MACHINE
# ───────────────────────────────────────────────────────────────
_lm = LearningMachine(
    model=ACTIVE_MODEL,          # required — used by all stores for extraction/saving
    db=agent_db,
    user_profile=UserProfileConfig(
        mode=LearningMode.ALWAYS,
        additional_instructions=[
            "Extract operator preferences: preferred altitude, airspeed, favourite locations, "
            "typical mission patterns (survey, recon, patrol, land vs RTL preference).",
            "Ignore battery — it is fake in SITL.",
        ]),
    user_memory=UserMemoryConfig(
        mode=LearningMode.ALWAYS,
        additional_instructions=[
            "Capture flight patterns, mission preferences, and operator corrections.",
            "Ignore battery — fake in SITL.",
        ]),
    session_context=SessionContextConfig(
        mode=LearningMode.ALWAYS,
        enable_planning=True,
        additional_instructions=[
            "Track mission: goal, planned stops, completed stops, interruptions.",
        ]),
    decision_log=DecisionLogConfig(
        mode=LearningMode.AGENTIC,
        additional_instructions=[
            "Log tool selection decisions and mission interpretation choices.",
        ]),
)

def _build_agent():
    comp = CompressionManager(
        model=OpenRouter(id="google/gemini-2.0-flash-001", extra_body=_NO_THINK),
        compress_tool_results_limit=20,
        compress_tool_call_instructions=(
            "Summarise this plane tool result in one line. "
            "Keep: action, GPS, altitude, airspeed, mode, armed, errors. Remove boilerplate."),
    )
    skills_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
    skills = None
    if os.path.isdir(skills_path):
        try: skills = Skills(loaders=[LocalSkills(skills_path)])
        except Exception as e: print(f"[SKILLS] {e}")

    ag = Agent(
        name="PlaneAgent", model=ACTIVE_MODEL,
        tools=[PlaneToolkit(), _fs],
        db=agent_db, add_history_to_context=True, num_history_runs=10,
        tool_call_limit=50, compression_manager=comp,
        enable_agentic_state=False,
        learning=_lm, add_learnings_to_context=True,
        system_message=_system_message(),
        markdown=False,
    )
    if skills: ag.skills = skills
    return ag

def _switch_model(model_id: str):
    global ACTIVE_MODEL, flight_agent
    model_id = model_id.replace("openrouter:","").strip()
    ACTIVE_MODEL = OpenRouter(id=model_id, extra_body=_NO_THINK)
    # Update the LearningMachine model so memory stores keep working
    try:
        _lm.model = ACTIVE_MODEL
    except Exception:
        pass
    flight_agent = _build_agent()
    for a in [safety_agent, summary_agent, _planner, _safety_validator, _summariser]:
        try: a.model = ACTIVE_MODEL
        except Exception: pass
    print(f"[MODEL] Now using: {model_id}")

# ───────────────────────────────────────────────────────────────
# SUPPORTING AGENTS
# ───────────────────────────────────────────────────────────────
_loc_kb = "\n".join(
    f"  {k}: {v['description']} ({v['lat']},{v['lon']})"
    for k,v in PRESET_LOCATIONS.items())
_plane_kb = (
    "VEHICLE: Fixed-wing ArduPlane SITL. Cannot hover.\n"
    "ALTITUDE: 30–120m AGL. Cruise 60m.\n"
    "AIRSPEED: 9–22 m/s. Cruise 15 m/s.\n"
    "LOITER RADIUS: 150m. WP_LOITER_RAD 300m.\n"
    "Battery is fake — ignore.\n"
    f"LOCATIONS:\n{_loc_kb}"
)

safety_agent = Agent(
    name="Safety Agent", model=ACTIVE_MODEL,
    output_schema=SafetyAssessment, db=agent_db,
    add_history_to_context=True, num_history_runs=3,
    instructions=[
        "You are a UAV safety officer for fixed-wing aircraft.",
        f"Regulations:\n{_plane_kb}",
        "Check: altitude 30–120m, airspeed 9–22 m/s, loiter radius ≥ 150m.",
        "Battery is fake — never flag it.",
        "Return SafetyAssessment with is_safe, risk_level, issues, recommendations, approved.",
    ], markdown=False)

flight_agent = _build_agent()

summary_agent = Agent(
    name="Summary Agent", model=ACTIVE_MODEL, db=agent_db,
    add_history_to_context=True, num_history_runs=10,
    instructions=[
        "Produce a fixed-wing flight session summary.",
        "Write a clear narrative in plain English, like a pilot debrief.",
        "Include: all commands in order, altitudes, locations, incidents, duration.",
        "No JSON, no field names. Prose only.",
    ], markdown=False)

_planner = Agent(
    name="Mission Planner", model=ACTIVE_MODEL,
    output_schema=MissionPlan, db=agent_db,
    instructions=[
        "Convert a fixed-wing mission description into a MissionPlan.",
        "Use PlaneCommand steps. Plane CANNOT hover — it orbits waypoints.",
        "Time estimates: arm=5s takeoff=40s transit 1km=90s orbit=120s RTL=120s land=180s.",
        "Risk: LOW=<80m simple, MEDIUM=multi-stop, HIGH=near limits.",
        "Always start with arm_plane + takeoff if ground-based.",
        f"Locations:\n{_loc_kb}",
    ], markdown=False)

_safety_validator = Agent(
    name="Safety Validator", model=ACTIVE_MODEL,
    output_schema=SafetyAssessment, db=agent_db,
    instructions=[
        "Evaluate a MissionPlan for fixed-wing safety.",
        f"{_plane_kb[:600]}",
        "Return SafetyAssessment.",
    ], markdown=False)

_summariser = Agent(
    name="Post-Flight Summariser", model=ACTIVE_MODEL, db=agent_db,
    instructions=[
        "Generate a plain English flight debrief from the session log.",
        "Flowing prose, no JSON, no field names.",
        "Include: commands in order, altitudes, locations, incidents, duration.",
    ], markdown=False)

_all_agents.extend([safety_agent, flight_agent, summary_agent,
                    _planner, _safety_validator, _summariser])

# ───────────────────────────────────────────────────────────────
# 4-STEP MISSION WORKFLOW
# ───────────────────────────────────────────────────────────────
def run_mission(desc: str):
    SEP = "=" * 58
    print(f"\n{SEP}\n  MISSION: {desc}\n{SEP}")

    print("\n[1/4] Planning...")
    mission_state["phase"] = "planning"
    resp = _planner.run(f"[STATE]\n{get_live_context()}\n\nPlan: {desc}")
    plan = resp.content if isinstance(resp.content, MissionPlan) else None
    if not plan:
        print("[1/4] Planning failed."); mission_state["phase"]="idle"; return
    mission_state["current_mission"] = plan.model_dump()
    print(f"Plan: {plan.mission_name} | {len(plan.steps)} steps | Risk: {plan.risk_level}")

    print("\n[2/4] Safety check...")
    mission_state["phase"] = "safety_check"
    sr = _safety_validator.run(
        f"[STATE]\n{get_live_context()}\n\nPlan:\n{json.dumps(mission_state['current_mission'],indent=2)}")
    sa = sr.content if isinstance(sr.content, SafetyAssessment) else None
    if sa:
        print(f"Safety: {'APPROVED' if sa.approved else 'REJECTED'} | Risk: {sa.risk_level}")
        if sa.issues: print(f"Issues: {sa.issues}")
        if not sa.approved:
            mission_state["phase"]="idle"; print("MISSION ABORTED."); return
    else:
        print("[2/4] Safety check skipped.")

    print("\n[3/4] Executing...")
    mission_state["phase"] = "executing"
    steps_txt = "\n".join(
        f"- {s.get('action','?')}: "
        + ", ".join(f"{k}={v}" for k,v in s.items()
                    if k not in ("action","reason") and v is not None)
        for s in mission_state["current_mission"]["steps"]
    ) if mission_state["current_mission"].get("steps") else desc
    print("-"*40)
    flight_agent.print_response(
        f"Execute:\n{steps_txt}\nGoal: {plan.objective}",
        session_id=SESSION_ID, stream=True)
    mission_state["phase"] = "complete"
    print("-"*40)

    print("\n[4/4] Report...")
    elapsed = int((datetime.datetime.now()-SESSION_START).total_seconds())
    log_txt = "\n".join(
        f"[{e['time']}] {e['action'].upper()} {e['details']} @ {e['alt']}m"
        for e in mission_state["flight_log"]) or "No commands logged."
    print("-"*40)
    _summariser.print_response(
        f"Session {SESSION_ID} | Duration {elapsed}s\nMission: {desc}\n"
        f"Log:\n{log_txt}\nMax alt: {round(mission_state['max_altitude'],1)}m",
        stream=True)
    print(f"\n{SEP}")

# ───────────────────────────────────────────────────────────────
# ASYNC RUNNER
# ───────────────────────────────────────────────────────────────
def _run_async(fn, *args, **kwargs):
    if not _hold_active.is_set():
        if not command_queue.empty() or stop_flag.is_set():
            stop_flag.set(); _clear_queue()

    def _t():
        try: fn(*args, **kwargs)
        except Exception as e:
            print(f"\n[AGENT ERROR] {e}")
            flog("error", f"AGENT ERROR: {e}")
        print("\n>> ", end="", flush=True)

    threading.Thread(target=_t, daemon=True).start()

# ───────────────────────────────────────────────────────────────
# STOP DETECTION
# ───────────────────────────────────────────────────────────────
_STOP_EXACT = {
    "stop","hold","halt","loiter","loiter here","orbit here",
    "hold position","freeze","pause","stop now","abort",
}
_STOP_PARTIAL = ("stop","halt","freeze","loiter here","orbit here","abort mission")

def _is_stop(text: str) -> bool:
    if text in _STOP_EXACT: return True
    if len(text.split()) <= 4:
        return any(p in text for p in _STOP_PARTIAL)
    return False

# ───────────────────────────────────────────────────────────────
# INPUT ROUTER
# ───────────────────────────────────────────────────────────────
_SUMMARY_KW = (
    "summary","flight log","what did","what happened","recap",
    "flight summary","session summary","show log","history","what commands",
    "what did the plane do","everything we did",
)
_MODEL_Q = (
    "what model","which model","what ai","what llm","what are you using",
    "what version","are you gemini","are you gpt","are you qwen",
)
_RESUME_PHRASES = (
    "resume","continue","carry on","keep going","go on",
    "resume mission","continue mission","pick up where",
)

def _handle(user_input: str):
    low = user_input.lower().strip()

    # ── /model ──────────────────────────────────────────────────
    if low.startswith("/model"):
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2:
            print(f"Current model: {ACTIVE_MODEL.id}")
            print("Switch: /model <id>  e.g. /model google/gemini-2.0-flash-001")
        else:
            _switch_model(parts[1].strip())
        return

    if any(q in low for q in _MODEL_Q):
        print(f"Model: {ACTIVE_MODEL.id} (OpenRouter). Switch with /model <id>"); return

    # ── /status  ────────────────────────────────────────────────
    if low in ("/status","status","full status","what is the status"):
        loc  = vehicle.location.global_relative_frame
        att  = vehicle.attitude
        batt = vehicle.battery
        wp   = getattr(vehicle,"wp_dist",None)
        eta  = getattr(vehicle,"wp_eta",None)
        print(f"Mode: {vehicle.mode.name} | Armed: {vehicle.armed}")
        print(f"Alt: {round(loc.alt,1)}m | lat: {round(loc.lat,6)} | lon: {round(loc.lon,6)}")
        print(f"Roll: {round(math.degrees(att.roll),1)} "
              f"Pitch: {round(math.degrees(att.pitch),1)} "
              f"Yaw: {round(math.degrees(att.yaw),1)}")
        print(f"Airspeed: {round(vehicle.airspeed,1)} m/s (measured) | "
              f"Target: {_target_airspeed:.1f} m/s | "
              f"Groundspeed: {round(vehicle.groundspeed,1)} m/s")
        print(f"WP dist: {f'{wp:.0f}m' if wp else 'N/A'} | "
              f"ETA: {f'{eta}s' if eta else 'N/A'}")
        print(f"Battery: {batt.level}% | {batt.voltage}V")
        return

    if low in ("/battery","battery","battery level"):
        b = vehicle.battery; print(f"Battery: {b.level}% | {b.voltage}V | {b.current}A"); return

    if low in ("/position","position","where is the plane","where are we"):
        loc = vehicle.location.global_relative_frame
        ll  = vehicle.location.local_frame
        print(f"GPS: {round(loc.lat,6)}, {round(loc.lon,6)} @ {round(loc.alt,1)}m")
        print(f"Local: N={round(ll.north,2)}m E={round(ll.east,2)}m D={round(ll.down,2)}m")
        return

    if low in ("/mission","/mission_wps","show mission","current mission"):
        wps = _download_mission()
        if not wps: print("No mission uploaded."); return
        print(f"Uploaded mission ({len(wps)} items) | current WP: {vehicle.commands.next}")
        for w in wps:
            print(f"  [{w['index']}] {w['command']} "
                  f"lat={w['lat']} lon={w['lon']} alt={w['alt']}m")
        return

    if low == "/plan":
        with _plan_lock:
            stops = list(_plan_stops)
        if not stops: print("No stops queued."); return
        print(f"Queued plan ({len(stops)} stop(s)):")
        for i,s in enumerate(stops):
            print(f"  {i+1}. {s.get('name','?')} @ {s['alt']}m "
                  + (f"dwell={s['dwell_s']}s" if s['dwell_s'] else ""))
        return

    if low == "/state":
        p = {k:v for k,v in mission_state.items() if k != "current_mission"}
        print(json.dumps(p, indent=2, default=str))
        print(f"\nConditions: {cond_monitor.list_all()}")
        print(f"Hold: {_hold_active.is_set()} | Stop: {stop_flag.is_set()}")
        try:
            print(f"TRIM_ARSPD_CM: {vehicle.parameters.get('TRIM_ARSPD_CM')} cm/s")
            print(f"WP_LOITER_RAD: {vehicle.parameters.get('WP_LOITER_RAD')} m")
            print(f"TKOFF_ALT: {vehicle.parameters.get('TKOFF_ALT')} m")
        except Exception: pass
        return

    if low == "/locations":
        print("Preset locations:")
        for k,v in PRESET_LOCATIONS.items():
            print(f"  {k}: {v['description']} ({v['lat']}, {v['lon']})")
        return

    if low == "/memory":
        print("--- Learning Machine State ---")
        try:
            if hasattr(_lm,"user_profile_store") and _lm.user_profile_store:
                print("\n[User Profile]"); _lm.user_profile_store.print(user_id="operator")
            if hasattr(_lm,"user_memory_store") and _lm.user_memory_store:
                print("\n[User Memory]"); _lm.user_memory_store.print(user_id="operator")
            if hasattr(_lm,"session_context_store") and _lm.session_context_store:
                print("\n[Session Context]"); _lm.session_context_store.print(session_id=SESSION_ID)
            if hasattr(_lm,"decision_log_store") and _lm.decision_log_store:
                print("\n[Decision Log]"); _lm.decision_log_store.print(agent_id="PlaneAgent",limit=5)
        except Exception as e: print(f"[MEMORY] {e}")
        return

    if low == "/mcp":
        print(f"missions/ → {_fs._md}")
        print(f"reports/  → {_fs._rd}")
        print(f"Mission files: {[x for x in os.listdir(_fs._md)]}")
        print(f"Report files:  {[x for x in os.listdir(_fs._rd)]}")
        return

    if low == "/report":
        elapsed = int((datetime.datetime.now()-SESSION_START).total_seconds())
        log_txt = "\n".join(f"[{e['time']}] {e['action']} {e['details']}"
                            for e in mission_state["flight_log"]) or "No commands."
        _run_async(summary_agent.print_response,
            f"Session {SESSION_ID}. Duration {elapsed}s.\nLog:\n{log_txt}\n"
            f"Max alt: {mission_state['max_altitude']}m",
            session_id=SESSION_ID, stream=True)
        return

    if low.startswith("/mission "):
        desc = user_input[9:].strip()
        if desc: _run_async(run_mission, desc)
        else: print("Usage: /mission <describe your mission>")
        return

    # ── Summary keywords ────────────────────────────────────────
    if any(kw in low for kw in _SUMMARY_KW):
        log = mission_state["flight_log"]
        if not log: print("No flight activity recorded yet."); return
        elapsed = int((datetime.datetime.now()-SESSION_START).total_seconds())
        log_txt = "\n".join(
            f"[{e['time']}] {e['action'].upper()} "
            + (", ".join(f"{k}={v}" for k,v in e['details'].items()) if e['details'] else "")
            + f" @ {e['alt']}m" for e in log)
        _run_async(summary_agent.print_response,
            f"Session {SESSION_ID} | Duration {elapsed}s\n"
            f"Max alt: {round(mission_state['max_altitude'],1)}m\n"
            f"Incidents: {mission_state['incidents'] or 'None'}\n\nLog:\n{log_txt}",
            session_id=SESSION_ID, stream=True)
        return

    # ── DISARM (direct phrase — intercepted before reaching the agent) ──
    _DISARM_PHRASES = ("disarm", "kill the engine", "kill engine", "cut the engine",
                       "cut engine", "shut down the engine", "turn off the engine",
                       "kill it", "power off")
    if any(low == p or low.startswith(p) for p in _DISARM_PHRASES):
        cur_alt = vehicle.location.global_relative_frame.alt or 0
        cur_spd = vehicle.airspeed or vehicle.groundspeed or 0
        if cur_alt > 3.0 or cur_spd > 3.0:
            print(f"[BLOCKED] Cannot disarm while airborne (alt={cur_alt:.1f}m, "
                  f"speed={cur_spd:.1f}m/s). Disarming in flight cuts power with no "
                  f"glide control and will crash the plane.")
            print("           Say 'land' to bring it down safely, then 'disarm' once it has touched down.")
            flog("warning", f"DISARM phrase blocked — airborne alt={cur_alt:.1f}m")
        else:
            command_queue.put({"action": "disarm"})
            print("[PLANE] Disarm requested — plane is on the ground, proceeding.")
        return

    # ── STOP → LOITER ────────────────────────────────────────────
    if _is_stop(low):
        _hold_active.set()
        _clear_queue()
        stop_flag.set()
        command_queue.put({"action": "hold"})
        n = len(mission_state["pending_mission"])
        print(f"[STOP] LOITER mode — orbiting current position. "
              f"{n} saved command(s). Say 'resume' to continue.")
        flog("info", f"STOP '{user_input}': {n} saved")
        return

    # ── Speed modifiers (instant) ────────────────────────────────
    _MAX_S  = ("max speed","maximum speed","fly faster","go faster","full speed",
               "top speed","fastest","increase speed","speed up")
    _MIN_S  = ("slow down","slower","reduce speed","decrease speed","minimum speed","slowest")
    _CRZ_S  = ("normal speed","cruise speed","medium speed","default speed")

    if any(p in low for p in _MAX_S):
        a = _set_speed(_SPEED_MAX)
        print(f"[SPEED] Target airspeed → {a:.0f} m/s (max). Measured will catch up. Mission unaffected."); return
    if any(p in low for p in _MIN_S):
        a = _set_speed(_SPEED_MIN)
        print(f"[SPEED] Target airspeed → {a:.0f} m/s (min). Measured will catch up. Mission unaffected."); return
    if any(p in low for p in _CRZ_S):
        a = _set_speed(_SPEED_CRZ)
        print(f"[SPEED] Target airspeed → {a:.0f} m/s (cruise). Measured will catch up. Mission unaffected."); return

    m = re.search(r"(?:set speed|speed|at)\s+(\d+(?:\.\d+)?)\s*(?:m/?s)?", low)
    if m and any(k in low for k in ("speed","m/s","metres per","meters per")):
        spd = min(_SPEED_MAX, max(_SPEED_MIN, float(m.group(1))))
        a = _set_speed(spd)
        print(f"[SPEED] Target airspeed → {a:.1f} m/s. Measured will catch up. Mission unaffected."); return

    # ── RESUME ───────────────────────────────────────────────────
    if any(low == p or low.startswith(p) for p in _RESUME_PHRASES):
        last    = mission_state.get("last_command", {})
        pending = mission_state.get("pending_mission", [])
        skip    = {"hold","loiter_here","set_speed","set_mode","resume"}
        if last.get("action") in skip: last = {}
        _hold_active.clear()
        stop_flag.clear()
        if not last and not pending:
            # Just switch back to AUTO
            command_queue.put({"action": "resume"})
            print("[RESUME] Switching back to AUTO.")
            flog("info", "RESUME -> AUTO (no pending commands)")
            return
        queued = []
        if last:
            command_queue.put(last); queued.append(last.get("action","?"))
        for c in pending:
            command_queue.put(c); queued.append(c.get("action","?"))
        mission_state["pending_mission"] = []
        summary = " -> ".join(queued)
        print(f"[RESUME] Re-queuing: {summary}")
        flog("info", f"RESUME: {summary}")
        return

    # ── Fall through to flight agent ─────────────────────────────
    enriched = f"[LIVE PLANE STATE]\n{get_live_context()}\n\n[USER COMMAND]\n{user_input}"
    flog("info", f"USER: {user_input}")
    _run_async(flight_agent.print_response, enriched,
               user_id="operator", session_id=SESSION_ID, stream=True)

# ───────────────────────────────────────────────────────────────
# CLI BANNER
# ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  AGENTIC FIXED-WING PLANE CONTROL — READY  (ArduPlane / SITL)")
print("=" * 70)
print("""
  ── FLIGHT LIFECYCLE ─────────────────────────────────────────────────
  >> arm and take off to 60 meters
  >> take off to 80 meters
  >> return home                         fly home and LOITER (no landing)
  >> land                                approach and TOUCH DOWN at home
  >> land here / emergency landing       land at CURRENT position (checked)
  >> disarm                              GROUND ONLY — blocked if airborne
  >> stop / hold / loiter here           orbit current position (LOITER)
  >> resume / continue                   back to AUTO, continue mission

  ── EMERGENCY LANDING SAFETY CHECK ───────────────────────────────────
  'land here' checks the current spot before touching down:
    - refuses within 200m of hospital / prison / residential areas
    - suggests nearest known safe zone (home/airfield/runway/reserve)
    - override with 'land here anyway' if you accept the risk
  Pre-approved zones (home, airfield, runway 35/17, reserve) always pass.

  ── SAFETY ───────────────────────────────────────────────────────────
  disarm / MANUAL / STABILIZE are BLOCKED while airborne — there is no
  RC pilot input in this sim, so cutting power or releasing control
  mid-flight causes an uncontrolled crash. Use 'land' to come down safely.

  ── SINGLE DESTINATION ───────────────────────────────────────────────
  >> go to the hospital at 70 meters     fly there, loiter forever
  >> fly to camp a                       fly there, loiter forever
  >> fly north 2 kilometres              offset nav, loiter at target

  ── MULTI-STOP MISSIONS ──────────────────────────────────────────────
  >> survey camp a then camp b then return home
  >> fly to the hospital then the prison then land
  >> go to the airfield, loiter 60 seconds, then return home
  >> visit residence 1 and residence 2 then land

  ── OBSTACLE AVOIDANCE / SHORT MOVES ─────────────────────────────────
  >> go left 80 meters                   steer around something, then wait
  >> climb 20 meters                     short up-nudge, then wait
  >> move right and climb 10 meters      lateral + altitude nudge
  These are brief manoeuvres (single orbit) — not indefinite loiters.
  Follow up with your next instruction once clear.

  ── NON-STANDARD ORBIT PATTERNS ──────────────────────────────────────
  >> fly a racetrack heading north for 3 laps        holding-pattern oval
  >> fly a figure eight over the airfield             tangent lobes, crossing
  >> survey camp a in a lawnmower pattern 800 by 400  boustrophedon grid

  ── SPEED ────────────────────────────────────────────────────────────
  >> speed up / max speed                → 22 m/s
  >> slow down / minimum speed           →  9 m/s
  >> cruise speed / normal speed         → 15 m/s
  >> set speed to 18                     → 18 m/s (9–22 range)
  Note: airspeed (TRIM_ARSPD_CM), not groundspeed

  ── CONDITIONS / TRIGGERS ────────────────────────────────────────────
  >> if altitude exceeds 100 meters RTL
  >> watch airspeed < 10 then rtl

  ── STATUS (instant, no AI) ──────────────────────────────────────────
  /status       full telemetry           /battery      battery readout
  /position     GPS + local NED          /mission      uploaded WP list
  /plan         queued stops             /locations    all named locations
  /state        internal flags + params  /memory       learning machine
  /mcp          file toolkit paths       /report       AI session summary
  /mission <desc>   4-step plan→safety→execute→report
  /model        show AI model            /model <id>   switch model
  exit
""")
print("=" * 70 + "\n")

# ───────────────────────────────────────────────────────────────
# MAIN LOOP
# ───────────────────────────────────────────────────────────────
while True:
    try:
        user_input = input(">> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nShutting down.")
        flog("info", f"SESSION END — {SESSION_ID} | KeyboardInterrupt")
        break
    if not user_input:
        continue
    if user_input.lower() in ("exit","quit"):
        print("Exiting.")
        flog("info", f"SESSION END — {SESSION_ID} | user exit")
        break
    _handle(user_input)
