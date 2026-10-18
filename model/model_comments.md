# Robot model: Unitree G1, prepared for torque-level whole-body MPC

**Status:** done — Phase 1 model selection closed
**Location:** `~/Workspace/projects/whole-body-control/model/g1_description/`

---

## Decision

| Role | File |
|---|---|
| MuJoCo plant | `g1_29dof_mpc.xml` (+ `scene_mpc.xml` with the floor) |
| Pinocchio model | `g1_29dof_rev_1_0.urdf` |
| Reproducible checks | `check_model.py` |

29 actuated DoF: 6 per leg, 3 waist, 7 per arm. 33.341 kg. Floating base (`freejoint`
on `pelvis`), so `nq=36`, `nv=35`, `nu=29`.

**Why 29dof_rev_1_0 and not the alternatives.** Unitree's README deprecates the
plainly-named `g1_29dof` / `g1_23dof` files (old hip gear ratios). The `mode_1X`
variants ship URDF only — no MJCF — so they cannot back a MuJoCo + Pinocchio pair.
`g1_23dof` has 5-DoF arms and waist-yaw only, too thin for a bimanual lift. Locking
the waist would remove the joints that mediate arm-momentum/base coupling, which is
the subject of the research question; `g1_29dof_lock_waist_rev_1_0` is kept in reserve
as a free ablation axis instead (identical in every other respect).

## Model consistency (verified, `check_model.py`)

- MuJoCo Menagerie `unitree_g1/g1.xml` is derived from `g1_29dof_rev_1_0.xml`; the two
  compile to **bit-identical** body poses, masses and inertias (max diff 0.0).
- The URDF and the MJCF have the **same 29 joints in the same order**; forward
  kinematics agree to **2.8e-7 m** at a random configuration (float32 storage
  precision). Pinocchio joint index `i` maps to MuJoCo `qpos[7+i]` / `ctrl[i]`.
- Known, negligible export difference: `torso_link` mass 7.817 kg (URDF) vs 7.818 kg
  (MJCF), with ipos differing by 3e-5 m. Wrist joint limits differ by 2e-6 rad.
- Menagerie's 51 STL files are byte-identical to the package's own `meshes/`, so the
  MPC model points at `meshes/` and `unitree_g1/assets/` is a redundant 35 MB copy.

## What was changed relative to Menagerie `g1.xml`

1. **`<position>` → `<motor>` (gear=1).** `ctrl` is joint torque in N·m; `ctrlrange`
   mirrors each joint's `actuatorfrcrange`, so it doubles as the OCP input-bound set.
   Torque limits: hip roll / knee ±139, hip pitch / hip yaw / waist yaw ±88, ankle and
   waist roll/pitch ±50, shoulders / elbow / wrist roll ±25, wrist pitch/yaw ±5 N·m.
   Verified **bit-exact**: replaying the position servo's `actuator_force` through these
   motors reproduces `qpos` to 0.0 under explicit Euler.
2. **Per-joint armature** from `g1_mjx.xml` (0.0036–0.0251 kg·m²) replacing Menagerie's
   flat 0.01 placeholder. Rotor inertia is not negligible at 14.3–22.5:1.
3. **`frictionloss` 0.3 → 0**, so the plant matches a Pinocchio model with no Coulomb
   term. Set 0.1 for robustness tests.
4. **Explicit `<option>`**: `timestep=0.002`, `implicitfast`, Newton, **elliptic cone,
   `impratio=100`**. Measured 0.00 mm foot slip over 5 s standing vs 0.28 mm for the
   pyramidal default, at the same ~50k steps/s.
5. **Self-collision off** (`contype=2`/`conaffinity=1` on the collision class): bodies
   still collide with the world but not with each other. One attribute to re-enable.
6. **Hand collision capsules + `left_palm`/`right_palm` sites** (from `g1_mjx.xml`) —
   the rubber hands are visual-only in `g1.xml`, which would have blocked the Phase 3
   box lift.
7. **Named foot contacts** (`*_heel_in/out`, `*_toe_in/out`) and `*_sole` site frames.
   Four point contacts per foot map directly onto a 4-point contact-wrench cone.
8. **Centroidal + contact sensors**: `subtreecom`, `subtreelinvel`, `subtreeangmom`,
   per-foot force/torque, sole and palm frames.
9. **Keyframes carry `qpos` only** — for a torque model `ctrl` is a torque, not a
   posture. `home` (z=0.7837, knees slightly bent) is the intended MPC nominal;
   `squat` (z=0.755) for perturbation work; `stand` has fully extended knees and is
   kinematically singular — do not use as a reference.

## Two gotchas worth remembering

**Explicit-PD stability bound.** A `motor` actuator contributes no ∂force/∂qvel term,
so `implicitfast` has nothing to make implicit and joint damping is 0. A hand-written
joint-space PD is then limited to `kd < 2·M_ii/dt`. The smallest hinge inertia is
0.00398 kg·m² (wrist roll), i.e. **kd < 4.0 at dt = 2 ms**. A textbook kp=500/kd=43 PD
stands at dt ≤ 0.5 ms and falls at 1–2 ms. This constrains baseline controllers, not
the MPC, but it cost an afternoon of debugging once already.

**Foot force sensors are not the GRF.** A site `force` sensor reports the
body/parent interaction wrench — here the ankle reaction, i.e. the ground reaction
minus the foot link's own weight and inertia (~6 N per foot at rest). For the true GRF,
sum `mj_contactForce` over the named `*_heel_*`/`*_toe_*` geoms. Using the sensor
directly in a contact-force cost would bias it.

## Licence

`g1_29dof_mpc.xml` derives from MuJoCo Menagerie's `unitree_g1` (BSD-3-Clause).
Cite MuJoCo Playground (Zakka et al., arXiv:2502.08844) if the MJX-derived collision
geometry or armature values end up in the write-up.