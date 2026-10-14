"""Independent kinematics checks and actual dynamic jump acceptance test."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import numpy as np
import mujoco
import jump
from plan_jump import ballistic_plan


def main():
    m=mujoco.MjModel.from_xml_string(jump.model_xml())
    d=mujoco.MjData(m)
    d.qpos[:3]=[0,0,.5];d.qpos[3:7]=[1,0,0,0]
    rng=np.random.default_rng(42)
    worst=0.;jworst=0.
    for i,(name,hip) in enumerate(zip(jump.LEGS,jump.HIPS)):
        side=np.sign(hip[1]);site=m.site(name+'_site').id
        for _ in range(100):
            q=np.array([rng.uniform(-.3,.3),rng.uniform(.4,1.1),rng.uniform(-2.2,-.8)])
            p=jump.fk(q,side)
            inv=jump.ik(p,side)
            assert np.max(np.abs(inv-q))<1e-10
            d.qpos[7+3*i:10+3*i]=q
            mujoco.mj_forward(m,d)
            expected=d.qpos[:3]+hip+p
            worst=max(worst,float(np.max(np.abs(d.site_xpos[site]-expected))))
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacSite(m,d,jp,jr,site)
            numerical=np.zeros((3,3))
            for k in range(3):
                dq=np.eye(3)[k]*1e-6
                numerical[:,k]=(jump.fk(q+dq,side)-jump.fk(q-dq,side))/(2e-6)
            jworst=max(jworst,float(np.max(np.abs(jp[:,6+3*i:9+3*i]-numerical))))
    assert worst<1e-10 and jworst<1e-7
    p=ballistic_plan(.359,.471,.511,.11)
    assert abs(.359+p['vz']*p['flight_time']-4.905*p['flight_time']**2-.471)<1e-10
    assert abs(p['vx']*p['flight_time']-.511)<1e-10
    result={'fk_vs_mujoco_max_error_m':worst,'jacobian_max_error':jworst,'random_leg_configurations':400,'ballistic_endpoint_check':True}
    summaries={}
    for height in [.10,.15,.20]:
        out=Path('verification')/f'height_{height:.2f}'
        with contextlib.redirect_stdout(io.StringIO()):
            s=jump.run(argparse.Namespace(height=height,out=out,duration=4.,viewer=False,video=False))
        summaries[str(height)]=s
        print(f'height={height:.2f} m, success={s["success"]}, nonfoot_contacts={s["nonfoot_contact_steps"]}')
    assert summaries['0.15']['success'], 'Default jump did not meet acceptance criteria'
    assert summaries['0.15']['com_ballistic_max_error_m']<.005
    result['height_trials']={h:{k:s[k] for k in ['success','stable_last_half_second','nonfoot_contact_steps']} for h,s in summaries.items()}
    Path('verification').mkdir(exist_ok=True)
    Path('verification/checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
