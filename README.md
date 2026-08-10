# AI Plane Flight Agent | Natural-Language Fixed-Wing Control

Control a simulated fixed-wing aircraft using plain English.

This project uses multiple AI agents to interpret natural-language instructions, plan missions, perform safety checks, execute flight commands, monitor the aircraft, and generate flight summaries.

Instead of manually programming every waypoint, you can simply describe what you want the aircraft to do and the system determines how to carry out the request.

Everything runs inside a simulated **ArduPlane SITL** environment, so **no physical aircraft is required**.

### Example Commands

```text
Arm and take off to 60 meters.

Fly north 2 kilometers.

Survey camp a then camp b then return home.

Fly a racetrack heading north for 3 laps.

Land on runway 35.

Stop and hold position.

Resume the mission.
```

---

## What Does This Project Do?

This project demonstrates how multiple AI agents can work together to control an autonomous fixed-wing aircraft in simulation.

Each agent has a specific responsibility, allowing complex flight requests to be broken down into smaller, structured, and safer actions.

Unlike a quadcopter, a fixed-wing aircraft **cannot hover**. Flight instructions therefore need to be translated into real MAVLink commands such as waypoints, orbit patterns, and landing sequences, which are uploaded to the autopilot and flown in `AUTO` mode.

The system can:

* Understand natural-language flight instructions
* Plan multi-step missions
* Perform safety checks before execution
* Execute flight commands through MAVLink
* Perform short obstacle-avoidance detours and automatically rejoin the active mission
* Fly non-standard orbit patterns such as racetracks, figure-eights, and lawnmower surveys
* Build runway-aligned landing approaches
* Measure how close the aircraft stops to the intended runway threshold
* Monitor aircraft state in the background
* Generate readable flight summaries
* Learn operator preferences over time
* Optionally provide browser-based control through the AVA WebUI

Everything runs in simulation, making the project suitable for experimentation, development, and learning without requiring a physical aircraft.

---

## How It Works

Whenever you enter an instruction, the system follows this workflow:

```text
User Instruction
       ↓
Flight Agent
       ↓
Mission Planning
       ↓
Safety Validation
       ↓
Command Queue
       ↓
Flight Execution
       ↓
Flight Summary
```

The aircraft can continue flying in the background while you provide additional instructions.

A single executor thread processes commands sequentially, preventing contradictory commands from being sent to the aircraft simultaneously.

---

## AI Agents

Several AI components work together behind the scenes.

### Flight Agent

The primary agent responsible for understanding natural-language instructions and determining what actions need to be performed.

For example:

```text
Survey camp a then camp b and return home.
```

The Flight Agent interprets the request and passes the required actions to the mission-planning system.

### Mission Planner

Breaks larger requests into smaller executable steps.

For example:

```text
Fly to the airfield, loiter for 60 seconds, then land.
```

can become:

```text
1. Arm and take off
2. Fly to the airfield
3. Loiter for 60 seconds
4. Approach and land
```

### Safety Validator

Validates missions before they are executed.

Current safety checks include:

* Maximum and minimum altitude limits
* Airspeed limits
* Landing-point safety checks
* Known hazard proximity checks

Missions that violate configured safety limits are rejected before execution.

### Session Summary Agent

Generates a readable summary when a mission is complete.

For example:

```text
The aircraft took off to 60 meters, flew to the airfield,
held position for 60 seconds, and then landed successfully.
```

### Learning Machine

The system can learn from previous interactions, including:

* Preferred altitude
* Preferred airspeed
* Frequently used locations
* Common mission patterns
* Preferences such as survey, patrol, landing, or return-to-home behavior

---

## Obstacle Avoidance

For a short, immediate detour without interrupting the current mission, you can give commands such as:

```text
Go left 80 meters.

Climb 20 meters.

Move right and climb 10 meters.
```

The aircraft performs the requested offset and then **automatically rejoins the active mission**.

This behaves more like a vehicle temporarily swerving around an obstacle and returning to its route than stopping the mission entirely.

If you want the aircraft to stop and hold position, use:

```text
Stop.
```

or:

```text
Hold position.
```

---

## Runway Landing

A command such as:

```text
Land on runway 35.
```

does not simply send the aircraft to a single landing point.

Instead, the system constructs a runway-aligned approach:

```text
Pre-approach waypoint
(on runway centerline and upwind of the threshold)
        ↓
Final approach waypoint
(lower altitude and shallow glide slope)
        ↓
Landing sequence
        ↓
Touchdown
        ↓
Rollout and stop
```

The approach heading is calculated from the coordinates of the runway's two named endpoints, such as `runway 35` and `runway 17`.

This allows the aircraft to align with the runway centerline instead of approaching from an arbitrary direction.

After landing, the script waits for the aircraft to finish rolling and then reports how far it stopped from the intended runway threshold.

> **Simulation note:** `runway 35` and `runway 17` are placeholder points near the default SITL home location. They are not surveyed real-world runway coordinates. Replacing those coordinates with different locations automatically updates the calculated landing approach without requiring changes to the landing logic.

---

## Orbit Patterns

The project supports several non-standard flight patterns built from real waypoint geometry rather than relying on a single circle command.

### Racetrack

```text
fly_racetrack
```

Creates a running-track or "stadium" pattern consisting of two straight legs connected by rounded ends.

Example:

```text
Fly a racetrack heading north for 3 laps.
```

### Figure Eight

```text
fly_figure_eight
```

Creates two circles that cross at a shared center point and are flown in opposite directions.

Example:

```text
Fly a figure eight over the airfield.
```

### Lawnmower Survey

```text
fly_lawnmower
```

Creates parallel back-and-forth flight legs covering a rectangular survey area.

Example:

```text
Survey camp a in a lawnmower pattern 800 by 400.
```

---

## Technologies

The project is built with:

* **Python**
* **DroneKit**
* **ArduPilot SITL / ArduPlane**
* **Agno**
* **OpenRouter**
* **PyMAVLink**

### Optional Components

* **Unreal Engine** — telemetry visualization
* **AVA WebUI** — browser-based chat control and live map visualization

---

## Environment Setup

### Requirements

Before running the project, you will need:

* Python 3.11 or newer
* An OpenRouter API key
* A working DroneKit environment
* ArduPilot SITL / ArduPlane
* Mission Planner or another SITL launcher on Windows

If you are setting up the environment from scratch, follow the installation guide:

[DroneKit + ArduPilot Windows Installation Guide](https://github.com/igsxf22/flight_manual/blob/main/win_install_dronekit_2026.md?utm_source=chatgpt.com)

The guide covers:

* Creating a Python virtual environment
* Installing DroneKit
* Setting up Mission Planner
* Configuring SITL
* Connecting the Python script to the simulator

Once the environment is working, continue with the steps below.

---

## Configuration

### 1. Create the Python File

Create a new Python file and copy the project script into it.

### 2. Configure Your OpenRouter API Key

At the beginning of the script, replace:

```text
your_api_key_here
```

with your OpenRouter API key.

### ⚠️ Keep Your API Key Secret

Never commit your real API key to a public repository.

If you share or push the script anywhere, replace your real key with:

```text
your_api_key_here
```

before doing so.

A leaked API key can be accessed by anyone who obtains the file or commit containing it. Simply deleting the key from a later version does **not** remove it from Git history.

For a real project, using an environment variable or `.env` file is recommended instead of hard-coding the key directly into the script.

---

## Running the Simulation

### 1. Start ArduPilot SITL

Start ArduPilot SITL through Mission Planner or your preferred launcher.

### 2. Verify the Connection

Make sure the simulator is available at:

```text
tcp:127.0.0.1:5763
```

### 3. Run the Agent

Run the Python script:

```bash
python filename.py
```

The system should now accept natural-language flight commands.

---

## Example Instructions

### Basic Flight

```text
Arm the plane.

Take off to 60 meters.

Return home.

Land.
```

### Landing

```text
Land.

Land on runway 35.

Land here.

Disarm.
```

### Navigation

```text
Fly north 70 meters.

Go to the hospital.

Fly to camp a.
```

### Multi-Step Missions

```text
Survey camp a then camp b then return home.

Fly to the hospital then the prison then land.

Go to the airfield, loiter for 60 seconds, then return home.
```

### Orbit Patterns

```text
Fly a racetrack heading north for 3 laps.

Fly a figure eight over the airfield.

Survey camp a in a lawnmower pattern 800 by 400.
```

### Obstacle Avoidance

```text
Go left 80 meters.

Climb 20 meters.

Move right and climb 10 meters.
```

### Continuous Flight

Stop the current mission:

```text
Stop.
```

Resume:

```text
Resume.
```

---

## Optional: Browser Control via AVA WebUI

The aircraft can also be controlled through a browser-based chat interface instead of the terminal using the **AVA WebUI** project.

The AVA WebUI repository is private, so you will need access before using this option.

Once you have it:

🔗 
[https://github.com/KashmirWorld/ava-webui/tree/cesiumm](https://github.com/KashmirWorld/ava-webui/tree/cesium)

Clone it and follow the setup instructions in its own README to get the app running locally (cesium branch).

When starting the WebUI, use:

```bash
python scripts/dev_stack.py --no-sim
```

### Why `--no-sim`?

The default launch also starts a fake telemetry generator.

Because this project already provides real telemetry from the SITL aircraft, running both telemetry sources at the same time can cause the map to flicker between the simulated data sources.

Using `--no-sim` prevents the fake telemetry generator from competing with the real aircraft telemetry.

### Connecting the WebUI

Once the WebUI is running:

1. Copy `backend/examples/chat_bridge.py` from the AVA WebUI repository into this project's directory.
2. Run the flight-agent script normally.
3. The script automatically connects to the WebUI when it is available.
4. If the WebUI is not running, the script falls back to terminal input.

Once connected, commands entered into the WebUI's **Flight** tab are processed in the same way as terminal commands.

The aircraft's live position is also streamed to the map.

---

## Files Created Automatically

The first time the script runs, it creates several files and directories used to store missions, logs, reports, and session data.

```text
missions/
logs/
reports/
plane_sessions.db
```

These are created automatically and do not require manual setup.

---

## Available Commands

| Command           | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `/status`         | Show live aircraft telemetry                               |
| `/position`       | Show the current GPS position                              |
| `/mission`        | Show the currently uploaded waypoints                      |
| `/plan`           | Show queued stops that have not yet been flown             |
| `/locations`      | Show available predefined locations                        |
| `/model`          | Show the currently active AI model                         |
| `/model <model>`  | Switch to a different AI model                             |
| `/state`          | Show mission state and internal flags                      |
| `/report`         | Generate a flight report                                   |
| `/memory`         | Show learned operator preferences                          |
| `/mission <text>` | Run the complete plan → safety → execute → report workflow |

---

## Available Locations

The simulation includes the following predefined locations:

```text
home
airfield
runway 35
runway 17
hospital
prison
camp a
camp b
reserve
residence 1
residence 2
creek south
location 1
```

> `runway 35` and `runway 17` are placeholder points near the default SITL home location. They are not surveyed real-world runway coordinates. The landing approach is calculated dynamically from their configured coordinates.

---

## Safety Notes

This project is designed for **simulation, experimentation, and research**.

The current configured flight limits are:

```text
Maximum altitude: 120 meters

Minimum altitude: 30 meters

Airspeed range: 9–22 m/s
```

These limits are enforced by the project's safety-validation system.

### Airborne Disarming and Manual Modes

While airborne, the aircraft cannot be:

* Disarmed
* Switched to `MANUAL`
* Switched to `STABILIZE`

There is no active RC pilot input in this simulation. Allowing the aircraft to be disarmed or switched to a manually controlled mode while airborne could therefore result in an uncontrolled crash.

The supported ways to stop a flying aircraft are to:

* Hold position
* Land

### Battery Simulation

Battery values in the simulator are simulated and **should not be used for real-world flight decision-making**.

### Real Aircraft Use

This project is currently intended for simulation.

If adapting the system for a real aircraft, thoroughly review and redesign the safety architecture, command validation, failsafes, communications handling, geofencing, flight termination behavior, and human-override mechanisms before attempting any real-world operation.

**Never assume that behavior which is safe in SITL is automatically safe on a physical aircraft.**

---

## Project Status

This project is an experimental demonstration of combining:

**Natural Language + Multi-Agent AI + Mission Planning + MAVLink + Fixed-Wing Simulation**

The goal is to explore how an operator can interact with a simulated autonomous aircraft using natural language while still maintaining structured mission planning and safety validation between the user's request and the aircraft.

---

## Repository

[AI Plane Flight Agent on GitHub](https://github.com/Mohamed-MT/Agentic-AI-Plane-Script-Project?utm_source=chatgpt.com)
