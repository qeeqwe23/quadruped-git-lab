"""Ballistic first estimate, NOT an automatic retuner for jump.py."""
import argparse
import math


def ballistic_plan(z0, z_land, dx, apex_clearance, gravity=9.81):
    if apex_clearance<=0 or dx<=0 or gravity<=0:
        raise ValueError('clearance, dx and gravity must be positive')
    apex=max(z0,z_land)+apex_clearance
    vz=math.sqrt(2*gravity*(apex-z0))
    discriminant=vz*vz-2*gravity*(z_land-z0)
    flight_time=(vz+math.sqrt(discriminant))/gravity
    return {'vx':dx/flight_time,'vz':vz,'flight_time':flight_time,'apex':apex}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--z0',type=float,default=.359)
    p.add_argument('--z-land',type=float,default=.471)
    p.add_argument('--dx',type=float,default=.511)
    p.add_argument('--clearance',type=float,default=.11)
    a=p.parse_args()
    print(ballistic_plan(a.z0,a.z_land,a.dx,a.clearance))
