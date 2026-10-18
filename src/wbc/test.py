from wbc.recording import RunReader

run = RunReader("runs/2026-10-17T21-58-45_m1_fallpython scripts/render_run.py runs//log.h5")
print(run.channels())
print(run["sim.state"].q.shape, run["sim.state"].t[-1])
print(run["sim.grf"].left_foot_heel_in[:5, 2])
print(run.at("sim.centroidal", 2.5)["com"])
print(run.meta["git"]["sha"], run.meta["versions"]["mujoco"])
run.close()