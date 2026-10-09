# CityFlow: Zheng et al., WWW 2019
*CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario*

**Read for:** how traffic simulators model intersections and signal phases for reinforcement learning. Background for the optional Phase 8.

## What it is
An open-source **microscopic traffic simulator** built for RL research on signal control. It has a C++ core with a Python API and is designed for city-scale scenarios. The paper reports it is more than 20× faster than SUMO for these workloads, because RL needs millions of simulation steps.

## Concepts to write down
- **Road network:** roads → lanes → intersections. Vehicles follow a car-following model (keep a safe gap) and change lanes.
- **Signal phase:** a set of movements (e.g. north–south straight + right) that get green at the same time. A **signal plan** is a sequence of phases with durations (cycle length, splits). Between phases: yellow / all-red clearance.
- **RL formulation (per intersection = one agent):**
  - *State:* vehicle counts / queue lengths per incoming lane, current phase.
  - *Action:* which phase to switch to next (or keep the current one).
  - *Reward:* usually negative queue length, negative waiting time, or **pressure** (vehicles on incoming lanes minus outgoing lanes). Max-pressure is also a strong *non-learned* controller.
  - *Evaluation:* average travel time of all vehicles (lower is better), plus throughput.
- **Multi-agent:** many intersections act at once and influence each other. That is why the graph view matters for control too.

## How it connects to this project
- Phase 8 uses **SUMO** (the brief names it) with TraCI. CityFlow is the stretch option for a DQN agent whose **state includes our 15-min-ahead forecasts**.
- Our baselines map directly onto the paper's vocabulary: fixed-time (static plan), max-pressure (reactive), forecast-aware (proactive).
- **Brief rule:** everything here is simulation. Never claim live traffic control.

## Write-down for interviews
- "A phase is a set of non-conflicting movements that get green together."
- "Reward = −queue length or −pressure; metric = average travel time."
- "Why simulate? You can't A/B test signal policies on real intersections; simulation gives repeatable seeds and counterfactuals."
