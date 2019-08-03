## Best Angle To Cross River
###############################
import numpy as np
import random as rd

def solve(v_river,v_person,s_river,T,lamda,max_iter):
    a=0.
    for i in range(max_iter):
        a -= -v_person*np.cos(a)-lamda*s_river/v_person*np.sin(a)/(np.cos(a)**2)
    dh = (v_river+v_person*np.sin(a))*T
    return dh

def gen():
    v_river =rd.randint(0,2)+rd.random()
    v_person = rd.randint(2,3)+rd.random()
    s_river = rd.randint(50,100)+rd.random()
    T = rd.randint(50,100)+rd.random()
    return v_river,v_person,s_river,T

def run():
    print ("1")
    v_river,v_person,s_river,T=gen()
    print("初始条件为[河流速度:{0}m/s,人游泳的速度:{1}m/s,河流宽度:{2}m,游泳限定时间:{3}s]".format(v_river,v_person,s_river,T))
    lamda=0.01
    max_iter=10000
    print("游泳最长深度为:{0}".format(solve(v_river,v_person,s_river,T,lamda,max_iter)))
    
if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()