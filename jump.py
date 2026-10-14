"""Educational 12-actuator, floating-base quadruped jump. SI units, world z up.

No base pose edits after initialization, no external force injection, no welds.
Controller uses simulator state (not a deployable state estimator).
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import mujoco

LEGS = ('FL', 'FR', 'RL', 'RR')
HIPS = np.array([[.18,.12,0],[.18,-.12,0],[-.18,.12,0],[-.18,-.12,0]])
L = .22
RADIUS = .025
DT = .001


def model_xml(height=.15):
    legs, motors = [], []
    for name, (x,y,z) in zip(LEGS,HIPS):
        sign = np.sign(y)
        legs.append(f'''
        <body name="{name}_hip" pos="{x} {y} 0">
          <joint name="{name}_roll" axis="1 0 0" range="-.7 .7"/>
          <geom type="capsule" fromto="0 0 0 0 {sign*.04} 0" size=".022" mass=".1"/>
          <body name="{name}_thigh" pos="0 {sign*.04} 0">
            <joint name="{name}_pitch" axis="0 1 0" range="-2.5 2.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{L}" size=".016" mass=".25" rgba=".1 .5 .8 1"/>
            <body name="{name}_calf" pos="0 0 -{L}">
              <joint name="{name}_knee" axis="0 1 0" range="-2.9 -.05"/>
              <geom type="capsule" fromto="0 0 0 0 0 -{L}" size=".012" mass=".15"/>
              <body name="{name}_foot" pos="0 0 -{L}">
                <geom name="{name}_foot_geom" type="sphere" size="{RADIUS}" mass=".04" rgba=".08 .08 .08 1"/>
                <site name="{name}_site" size=".008"/>
              </body>
            </body>
          </body>
        </body>''')
        for joint in ('roll','pitch','knee'):
            motors.append(f'<motor name="{name}_{joint}_motor" joint="{name}_{joint}" gear="1" ctrllimited="true" ctrlrange="-45 45"/>')
    return f'''<mujoco model="teaching_quadruped">
    <compiler angle="radian"/>
    <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="60"/>
    <visual><global offwidth="960" offheight="640"/><headlight ambient=".4 .4 .4"/></visual>
    <default>
      <joint damping=".12" armature=".015" limited="true"/>
      <geom friction=".9 .01 .001" condim="3" solref=".008 1" rgba=".35 .4 .45 1"/>
    </default>
    <worldbody>
      <light pos="0 -2 4" dir="0 .3 -1"/>
      <geom name="floor" type="plane" size="4 3 .1" rgba=".85 .87 .89 1"/>
      <geom name="step" type="box" pos="1.3 0 {height/2}" size="1 .65 {height/2}" rgba=".4 .58 .4 1"/>
      <body name="trunk" pos="0 0 .345">
        <freejoint/>
        <geom name="trunk_geom" type="box" size=".22 .10 .055" mass="7" rgba=".16 .23 .32 1"/>
        {''.join(legs)}
      </body>
    </worldbody>
    <actuator>{''.join(motors)}</actuator>
    </mujoco>'''


def ik(p, side):
    """Foot relative to hip-roll origin. Knee-back branch; raises if unreachable."""
    x,y,z = np.asarray(p)
    d = side*.04
    yz2 = y*y+z*z-d*d
    if yz2 <= 0:
        raise ValueError('Foot too close to abduction axis')
    zs = -np.sqrt(yz2)
    q0 = np.arctan2(z,y)-np.arctan2(zs,d)
    q0 = (q0+np.pi)%(2*np.pi)-np.pi
    c = (x*x+zs*zs-2*L*L)/(2*L*L)
    if abs(c)>1+1e-7:
        raise ValueError(f'Unreachable foot target: {p}')
    q2 = -np.arccos(np.clip(c,-1,1))
    q1 = np.arctan2(-x,-zs)-np.arctan2(L*np.sin(q2),L+L*np.cos(q2))
    return np.array([q0,q1,q2])


def fk(q, side):
    q0,q1,q2 = q
    x = -L*np.sin(q1)-L*np.sin(q1+q2)
    zs = -L*np.cos(q1)-L*np.cos(q1+q2)
    d = side*.04
    return np.array([x,d*np.cos(q0)-zs*np.sin(q0),d*np.sin(q0)+zs*np.cos(q0)])


def skew(r):
    x,y,z = r
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


class Controller:
    def __init__(self,m,d,height):
        self.m,self.d,self.height=m,d,height
        self.mass=float(m.body_mass.sum())
        self.trunk=m.body('trunk').id
        self.sites=[m.site(n+'_site').id for n in LEGS]
        self.foot_geoms=[m.geom(n+'_foot_geom').id for n in LEGS]
        self.ground={m.geom('floor').id,m.geom('step').id}
        self.phase='settle'
        self.liftoff=None
        self.touchdown=None
        self.target_x=.70
        self.anchors=np.array([[x,y+np.sign(y)*.04,RADIUS] for x,y,z in HIPS])
        self.last_tau=np.zeros(12)
        self.jac=np.zeros((4,3,m.nv))
        self.rotjac=np.zeros((3,m.nv))
        self.history=[]
        self.commands=0
        self.saturated_commands=0
        self.takeoff_duration=.12
        self.vlaunch=np.array([1.95,0,3.3])
        self.launch_start=1.3

    def contacts(self):
        active=np.zeros(4,dtype=bool)
        loads=np.zeros(4)
        nonfoot=False
        cf=np.zeros(6)
        for k in range(self.d.ncon):
            c=self.d.contact[k]
            a,b=int(c.geom1),int(c.geom2)
            for i,g in enumerate(self.foot_geoms):
                if (a==g and b in self.ground) or (b==g and a in self.ground):
                    mujoco.mj_contactForce(self.m,self.d,k,cf)
                    if cf[0]>.5:
                        active[i]=True
                        loads[i]+=cf[0]
            if (a in self.ground and b not in self.ground and b not in self.foot_geoms) or (b in self.ground and a not in self.ground and a not in self.foot_geoms):
                nonfoot=True
        return active,loads,nonfoot

    def compute(self):
        m,d=self.m,self.d
        # Deliberate simple sample-and-hold loop: mj_step leaves derived
        # kinematics at the previous integration sample (about 1 ms latency).
        # qpos/qvel are newer. This fixed timing is part of the tuned example;
        # replacing it with mj_forward / mj_step1 requires retuning.
        mujoco.mj_subtreeVel(m,d)
        t=d.time
        active,loads,bad=self.contacts()
        pos=d.qpos[:3].copy()
        vel=d.qvel[:3].copy()
        rot=d.xmat[self.trunk].reshape(3,3)
        omega=rot@d.qvel[3:6]
        # Rotation error with respect to world-aligned trunk, expressed in world.
        err=.5*np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])
        feet=d.site_xpos[self.sites].copy()
        for i,sid in enumerate(self.sites):
            mujoco.mj_jacSite(m,d,self.jac[i],self.rotjac,sid)
        if t<.7:
            self.phase='settle'
            des=np.array([0,0,.345]); vdes=np.zeros(3); acc=np.zeros(3)
        elif t<self.launch_start:
            self.phase='crouch'
            u=(t-.7)/(.6)
            s=10*u**3-15*u**4+6*u**5
            sd=(30*u**2-60*u**3+30*u**4)/.6
            sdd=(60*u-180*u**2+120*u**3)/.6**2
            des=np.array([0,0,.345-.10*s]);vdes=np.array([0,0,-.10*sd]);acc=np.array([0,0,-.10*sdd])
        elif self.liftoff is None:
            self.phase='push'
            u=t-self.launch_start
            acc=self.vlaunch/self.takeoff_duration
            des=np.array([0,0,.245])+.5*acc*u*u
            vdes=acc*u
            if (u>.06 and not active.any()) or u>=self.takeoff_duration:
                self.liftoff=t
                self.launch_pos=pos.copy()
                self.launch_vel=vel.copy()
                self.phase='flight'
        if self.liftoff is not None:
            since=t-self.liftoff
            if self.touchdown is None and since>.12 and active.any():
                self.touchdown=t
                self.land_pos=pos.copy()
                self.land_feet=feet.copy()
                self.anchors=feet.copy()
                self.anchors[:,2]=self.height+RADIUS
                self.target_x=float(feet[:,0].mean())
            if self.touchdown is None:
                self.phase='flight'
            else:
                self.phase='land' if t-self.touchdown<.35 else 'stand'
                des=np.array([self.target_x,0,self.height+.345]);vdes=np.zeros(3);acc=np.zeros(3)
        tau=np.zeros(12)
        if self.phase=='flight':
            since=t-self.liftoff
            # Retract initially; deploy before descending onto the platform.
            extension=.20 if since<.16 else .30
            for i,hip in enumerate(HIPS):
                ext=extension if since<.16 else (.27 if i<2 else .33)
                desired=ik(np.array([.04,np.sign(hip[1])*.04,-ext]),np.sign(hip[1]))
                sl=slice(i*3,i*3+3)
                tau[sl]=70*(desired-d.qpos[7+3*i:10+3*i])-2.5*d.qvel[6+3*i:9+3*i]
        else:
            kp=np.array([90,90,180]);kd=np.array([18,18,24])
            if self.phase in ('land','stand'):
                kp=np.array([65,90,140]);kd=np.array([18,18,23])
            force=self.mass*(acc+kp*(des-pos)+kd*(vdes-vel)+np.array([0,0,9.81]))
            moment=-np.array([100,100,60])*err-np.array([12,12,8])*omega
            # Wrench distribution: least-squares, then unilateral/friction clipping.
            support=np.arange(4) if self.phase not in ('land','stand') else np.flatnonzero(active)
            if len(support)==0:support=np.arange(4)
            A=np.vstack([np.tile(np.eye(3),(1,len(support))),np.hstack([skew(feet[i]-pos) for i in support])])
            f=(A.T@np.linalg.solve(A@A.T+1e-5*np.eye(6),np.r_[force,moment])).reshape(-1,3)
            for j,i in enumerate(support):
                f[j,2]=np.clip(f[j,2],0,180)
                xy=np.linalg.norm(f[j,:2]); limit=.75*f[j,2]
                if xy>limit:f[j,:2]*=limit/max(xy,1e-9)
                sl=slice(3*i,3*i+3)
                # f is ground-on-robot force; actuator command has a minus sign.
                tau[sl]=-self.jac[i,:,6+3*i:9+3*i].T@f[j]
            # Keep stance feet near their world anchors; low-gain task-space spring.
            for i in range(4):
                sl=slice(3*i,3*i+3)
                j=self.jac[i,:,6+3*i:9+3*i]
                footvel=self.jac[i]@d.qvel
                gain=800 if self.phase in ('land','stand') and not active[i] else 100
                tau[sl]+=j.T@(gain*(self.anchors[i]-feet[i])-4*footvel)
        self.saturated_commands+=int(np.count_nonzero(np.abs(tau)>45))
        self.commands+=12
        tau=np.clip(tau,-45,45)
        d.ctrl[:]=tau
        self.last_tau=tau
        self.history.append([t,*pos,*vel,*err,*active.astype(int),*loads,float(np.max(np.abs(tau))),self.phase,int(bad),*feet.reshape(-1),*d.subtree_com[self.trunk],*d.subtree_linvel[self.trunk],*tau,float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1))))])


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    xml=model_xml(args.height);(out/'model.xml').write_text(xml,encoding='utf-8')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    d.qpos[:3]=[0,0,.345];d.qpos[3:7]=[1,0,0,0]
    for i,hip in enumerate(HIPS):
        d.qpos[7+3*i:10+3*i]=ik([0,np.sign(hip[1])*.04,-.32],np.sign(hip[1]))
    mujoco.mj_forward(m,d)
    c=Controller(m,d,args.height)
    viewer=None;renderer=None;writer=None;frames=[]
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer=mjviewer.launch_passive(m,d)
        viewer.cam.lookat[:]=[.4,0,.35];viewer.cam.distance=2.5;viewer.cam.azimuth=135;viewer.cam.elevation=-15
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=640,width=960)
        cam=mujoco.MjvCamera();cam.lookat[:]=[.4,0,.32];cam.distance=2.1;cam.azimuth=105;cam.elevation=-15
        writer=imageio.get_writer(str(out/'jump.mp4'),fps=50,codec='libx264',quality=8)
    start=time.perf_counter()
    for k in range(round(args.duration/DT)):
        c.compute()
        if renderer and k%20==0:
            renderer.update_scene(d,camera=cam);frame=renderer.render();writer.append_data(frame)
            if k in (400,1200,1400,1560,1680,1820,2200,3800):frames.append((k*DT,frame.copy(),c.phase))
        mujoco.mj_step(m,d)
        if viewer:
            if not viewer.is_running():break
            if k%10==0:viewer.sync()
            wait=d.time-(time.perf_counter()-start)
            if wait>0:time.sleep(wait)
    if viewer:viewer.close()
    if writer:writer.close();renderer.close()
    fields=['t','x','y','z','vx','vy','vz','rx','ry','rz',*[f'contact_{n}' for n in LEGS],*[f'normal_{n}' for n in LEGS],'max_tau','phase','nonfoot_contact',*[f'{n}_{axis}' for n in LEGS for axis in 'xyz'],*[f'com_{a}' for a in ('x','y','z','vx','vy','vz')],*[f'tau_{n}_{j}' for n in LEGS for j in ('roll','pitch','knee')],'trunk_tilt_deg']
    with (out/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(c.history)
    mujoco.mj_forward(m,d)
    active,loads,bad=c.contacts();feet=d.site_xpos[c.sites]
    tail=[r for r in c.history if r[0]>=d.time-.5]
    def stable(r):
        pts=np.array(r[21:33]).reshape(4,3)
        return all(r[10:14]) and (pts[:,0]>.30+RADIUS).all() and np.max(np.abs(pts[:,2]-(args.height+RADIUS)))<.015 and np.linalg.norm(r[4:7])<.1 and np.linalg.norm(r[7:10])<.1 and not r[20]
    air=[r for r in c.history if r[19]=='flight' and not any(r[10:14]) and not r[20]]
    nonfoot=sum(r[20] for r in c.history)
    success=bool(c.liftoff is not None and c.touchdown is not None and len(tail)>=490 and all(stable(r) for r in tail) and len(air)*DT>.15 and nonfoot==0 and d.xmat[c.trunk].reshape(3,3)[2,2]>.98)
    summary={'success':success,'mass_kg':c.mass,'step_height_m':args.height,'final_xyz_m':d.qpos[:3].tolist(),'final_speed_m_s':float(np.linalg.norm(d.qvel[:3])),'liftoff_time_s':c.liftoff,'touchdown_time_s':c.touchdown,'airborne_samples_duration_s':len(air)*DT,'stable_last_half_second':bool(tail and all(stable(r) for r in tail)),'final_contacts':active.tolist(),'final_feet_xyz_m':feet.tolist(),'peak_joint_torque_Nm':float(max(row[18] for row in c.history)),'saturated_joint_command_fraction':c.saturated_commands/max(c.commands,1),'nonfoot_contact_steps':nonfoot,'mujoco_version':mujoco.__version__,'numpy_version':np.__version__}
    if c.liftoff is not None:summary.update(launch_xyz_m=c.launch_pos.tolist(),launch_velocity_m_s=c.launch_vel.tolist())
    summary['peak_trunk_tilt_deg']=max(r[-1] for r in c.history)
    summary['peak_foot_normal_force_N']=max(max(r[14:18]) for r in c.history)
    if len(air)>100:
        mid=air[20:-20]; t0=mid[0][0]; p0=np.array(mid[0][33:36]); v0=np.array(mid[0][36:39])
        residual=[np.linalg.norm(np.array(r[33:36])-(p0+v0*(r[0]-t0)+np.array([0,0,-4.905*(r[0]-t0)**2]))) for r in mid]
        summary['com_ballistic_max_error_m']=float(max(residual))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
    if frames:
        from PIL import Image,ImageDraw,ImageFont
        strip=Image.new('RGB',(960*2,640*4),'white')
        try:font=ImageFont.truetype('arial.ttf',30)
        except OSError:font=ImageFont.load_default(size=30)
        for idx,(t,frame,phase) in enumerate(frames):
            im=Image.fromarray(frame);ImageDraw.Draw(im).text((25,25),f'{phase.upper()}   t = {t:.2f} s',fill='white',font=font)
            strip.paste(im,((idx%2)*960,(idx//2)*640))
        strip.save(out/'sequence.png')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height',type=float,default=.15)
    p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--out',default='results')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if not 0<args.height<=.25:p.error('height must be in (0, .25] m; only default is tuned')
    if args.duration<=0:p.error('duration must be positive')
    raise SystemExit(0 if run(args)['success'] else 2)
