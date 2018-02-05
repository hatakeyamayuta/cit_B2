import numpy as np
"""
def g_yakobi(dx,dy,th,l1,l2,l3):
    dv = np.array([[dx],[dy]])
    th1=th[0]
    th2=th[1]
    J = np.array([[-l1*np.sin(th1) - l2*np.sin(th1+th2) - l3*np.sin(th1+th2+th3),
                   -l2*np.sin(th1+th2) - l3*np.sin(th1+th2+th3),
                   -l3*np.sin(th1+th2+th3)],
                   [l1*np.cos(th1) + l2*np.cos(th1+th2) + l3*np.cos(th1+th2+th3),
                   l2*np.cos(th1+th2) + l3*np.cos(th1+th2+th3),
                   l3*np.cos(th1+th2+th3)]])
    gJ = np.linalg.pinv(J)
    print(gJ)
    #dth = np.dot(J,np.array([[np.radians(90)],[np.radians(90)],[np.radians(0)]]))
    dth = np.dot(gJ, dv)
    print(np.degrees(dth[0]),np.degrees(dth[1]),np.degrees(dth[2]))
"""
def g_yakobi2(dx,dy,th,l1,l2,l3):
    th1=th[0]
    th2=th[1]
    th3=th[2]
    dv = np.array([[dx],[dy]])
    J = np.array([[-l1*np.sin(th1) - l2*np.sin(th1+th2) - l3*np.sin(th1+th2+th3),
                   -l2*np.sin(th1+th2) - l3*np.sin(th1+th2+th3),
                   -l3*np.sin(th1+th2+th3)],
                   [l1*np.cos(th1) + l2*np.cos(th1+th2) + l3*np.cos(th1+th2+th3),
                   l2*np.cos(th1+th2) + l3*np.cos(th1+th2+th3),
                   l3*np.cos(th1+th2+th3)]])
    gJ = np.linalg.pinv(J)
    #print(gJ)
    #dth = np.dot(J,np.array([[np.radians(90)],[np.radians(90)],[np.radians(0)]]))
    dth = np.dot(gJ, dv)
    #print(np.degrees(dth[0]),np.degrees(dth[1]),np.degrees(dth[2]))
    return dth


def g_y2(dx,dy,th1,th2,l1,l2):
    dv=np.array([[dx], [dy]])
    J = np.array([[-l1*np.sin(th1) - l2*np.sin(th1+th2), -l2*np.sin(th1+th2)],
                   [l1*np.cos(th1) + l2*np.cos(th1+th2),
                   l2*np.cos(th1 + th2)]])
    gJ=np.linalg.pinv(J)
    print(gJ)
    
    dth = np.dot(gJ,dv)
    print(np.degrees(dth[0]),np.degrees(dth[1]))
if __name__=='__main__':
    #初期値
    th = np.array([[np.radians(101), np.radians(60), np.radians(-110)]])
    print(th)
    #3軸逆ヤコビ
    
    dth = g_yakobi2(0, 0, th[0], 200, 160, 72)
    th = th-dth.T
    deg = np.degrees(th[0]).astype(np.int64)
    with open('angle_t.txt','a')as f:
        f.write('90,{0},{1},{2},0,200'.format(deg[0], deg[1], deg[2]) + '\n')
    print(deg)
    for i in range(20):
         dth = g_yakobi2(5, 0, th[0], 200, 160, 72)
         th = th -dth.T
         deg = np.degrees(th[0]).astype(np.int64)
         with open('angle_t.txt','a')as f:
            f.write('90,{0},{1},{2},0,200'.format(deg[0], deg[1], deg[2]) + '\n')
    
    #g_y2(30,-30,0,np.radians(90),200,150)
