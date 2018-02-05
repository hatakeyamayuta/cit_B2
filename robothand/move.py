import numpy as np
import math as mt


def kinematics(x,y,z,l1,l2,l3,th):
    print('l1={0} l2={1} l3={2}'.format(l1,l2,l3))
    print('th={0}'.format(th))
    th = mt.radians(th)
    th4 = mt.atan2(y, x)
    print('th4={0}'.format(str(mt.degrees(th4))))
    tx = mt.cos(-th4)*x - mt.sin(-th4)*y
    ty = mt.sin(-th4)*x + mt.cos(-th4)*y
    print(tx,int(ty))
    X = tx - l3*mt.cos(th)
    Z = z - l3*mt.sin(th)
    print(X,Z)
    th2 = -mt.acos((X**2 + Z**2 - l1**2 - l2**2)/(2*l1*l2))
    print('th2={0}'.format(mt.degrees(th2)))
    #th1 = mt.atan2(Y/X)-mt.atan((l2*mt.sin(th2))/(l1 + l2*mt.cos(th2)))
    th1 = mt.atan2(Z, X) + mt.atan2((l2*mt.sin(th2)), (l1+l2*mt.cos(th2)))
    print('th1={0}'.format(mt.degrees(th1)))
    th3=th - th1 - th2
    print('th3={0}'.format(mt.degrees(th3)))





if __name__ == '__main__':
    l = [250,160,72]
    ps = [200, 0, -100]
    if ps[0]**2+ps[1]**2+ps[2]**2 > 482**2:
        print('over point')
    kinematics(ps[0],ps[1],ps[2],l[0],l[1],l[2],-90)
