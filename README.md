# whole-body-mpc

Whole-body model-predictive control for a simulated humanoid, written from the
optimal control problem up rather than by configuring an existing stack.

Robot: Unitree G1, 29 DoF, torque controlled. Dynamics with Pinocchio, OCP with
Crocoddyl (FDDP), plant in MuJoCo. Simulation only.

## Research question

How do formulation choices — centroidal versus full rigid-body dynamics, contact
model, horizon length, cost structure — trade off tracking quality, robustness and
solve time in whole-body MPC for tasks where the arms generate significant
momentum?

The headline task is bimanual loco-manipulation: walk to a table and lift a box,
with the object carried explicitly in the optimisation and the hand–object forces
as decision variables, so the controller anticipates the payload instead of
reacting to it.

## Status

| Milestone | State |
|---|---|
| M0 model: G1 verified Pinocchio ↔ MuJoCo | done |
| M1 plant: spawn, step, query, render, log | done |
| M2 standing balance and reaching under MPC | next |
| M3 stepping | |
| M4 loco-manipulation + ablation | |

## Install

Pinocchio and Crocoddyl are C++ libraries with Python bindings and must come from
conda-forge — pip-installing them into a conda environment links them against a
different numpy and boost.

```bash
conda env create -f environment.yml
conda activate wb-mpc
```

`environment.lock.yml` records the resolved versions.

## Run

```bash
python   scripts/run_sim.py --duration 5 --name m1_fall   # headless, logs to runs/
mjpython scripts/run_sim.py --viewer --rtf 0.25           # macOS needs mjpython
python   scripts/run_sim.py --no-log
```

Each run writes `runs/<timestamp>_<name>/` with `log.h5`, `meta.yaml` and the
flattened `model.xml`, so a run can be replayed and re-rendered without the
original process.

```python
from wbc.recording import RunReader

run = RunReader("runs/<dir>/log.h5")
run.channels()
run["sim.state"].q
run.at("sim.centroidal", 2.5)["com"]
```

## Layout

```
src/wbc/
  types.py       shared dataclasses, no dependencies beyond numpy
  model.py       Pinocchio model, geometry, limits
  sim.py         MuJoCo plant, MuJoCo↔Pinocchio conversion, viewer
  recording.py   multi-rate logger, HDF5 backend, run reader
  ocp.py         optimal control problem
  control.py     MPC loop and low-level tracker
model/           G1 URDF, MJCF, meshes
scripts/         run drivers
tests/
```

Two import rules keep the measurement honest: `ocp.py` never touches MuJoCo, and
`sim.py` never touches Crocoddyl. Model mismatch between plant and formulation is
an experiment, not an accident.

## Model

`model/g1_29dof_mpc.xml` derives from MuJoCo Menagerie's `unitree_g1`
(BSD-3-Clause), with position servos replaced by torque motors, per-joint armature
from `g1_mjx.xml`, an elliptic friction cone, named four-point foot contacts and
centroidal sensors. `model/model_comments.md` documents every change.

Ankle and waist roll/pitch effort limits in the URDF were raised from 35 to 50 N·m
to match the MJCF: 35 N·m cannot drive the centre of pressure to the toe in single
support (327 N × 0.12 m ≈ 39 N·m), which would silently truncate the support
polygon.

Cite MuJoCo Playground (Zakka et al., arXiv:2502.08844) if the MJX-derived
collision geometry or armature values appear in a write-up.

## License

Apache-2.0. Robot model files retain their upstream BSD-3-Clause licence.
