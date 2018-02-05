import numpy as np
import math as mat
def inverse(x, y, z, fh, l1, l2, l3, l4, l5):
    #th1 = np.arctan2(y,x)
    th1 = 0
    print(th1)

    p = np.array([[x],[y],[z]])
    a = np.array([[np.sin(th1)*np.sin(fh[1])],
                  [np.cos(th1)*np.sin(fh[1])],
                  [np.cos(fh[1])]])
    print(p)

    p3 = p - (l4 + l5)*a
    print(p3)

    p3x = p3[0]**2
    p3y = p3[1]**2
    p3z = (p3[2]-l1)**2
    c3 = (p3x + p3y + p3z - l2**2 - l3**2)/(2*l2*l3)
    th3 = np.arctan2(np.sqrt(1-c3**2), c3)
    A = np.sqrt(p3x + p3y)
    B = p3[2] - l1
    M = c3*l3+l2
    N = np.sin(th3)*l3
    th2 = np.arctan2(M*A-N*B, N*A + M*B)
    th4 = fh[1] - th3 - th2
    print(p3,np.degrees(th2),np.degrees(th3),np.degrees(th4))
    th2, th3 ,th4 = mat.degrees(th2),mat.degrees(th3),mat.degrees(th4)
    with open('angle_4.txt', 'a')as f:
        f.write('0,{0},{1},{2},0'.format(int(th2),int(th3),int(th4))+'\n')
    
if __name__ =='__main__':
    pr = np.array([[50],[250],[350]])
    dr = np.array([[0],[-5],[5]])
    th = ([[0, np.radians(90), 0]])
    inverse(0,pr[1],pr[2],th[0],300,200,160,72,0)
    for i in range(10):
        pr += dr
        inverse(0,pr[1],pr[2],th[0],300,200,160,72,0)

