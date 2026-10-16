"""CPU physics only; the XML deliberately uses zero gravity."""
from pathlib import Path
import json,csv,mujoco
model=mujoco.MjModel.from_xml_string(Path(__file__).with_name('one_joint.xml').read_text(encoding='utf-8'))
data=mujoco.MjData(model)
data.ctrl[0]=.5
trace=[(0.,0.,.5)]
for i in range(1500):
    mujoco.mj_step(model,data)
    if (i+1)%10==0:trace.append((data.time,float(data.qpos[0]),.5))
with Path(__file__).with_name('joint_response.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['time_s','qpos_rad','target_rad']);writer.writerows(trace)
assert abs(data.time-3.)<1e-9
assert abs(data.qpos[0]-.5)<.002
print(json.dumps({'mujoco_version':mujoco.__version__,'seconds':data.time,'qpos':float(data.qpos[0]),'qvel':float(data.qvel[0]),'target':.5}))
