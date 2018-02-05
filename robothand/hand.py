import paramiko
import time
def hand_write(angle,fg):
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('',port=,username='htk',password='')
        stdin,stdout,stderr=ssh.exec_command("> angles echo {0}".format(angle))
        ssh.exec_command("echo {0} > tes ".format(fg))

fg = 0
with open('angle_1.txt','r') as f:
    angles = f.readlines()
#a = 10.232.169.107
for i,angle in enumerate(angles):
    print('step{0}'.format(i))
    if i == 50:
        fg = 1
    elif i == 93:
        fg = 0
    hand_write(str(angle),fg)
    time.sleep(0) 
