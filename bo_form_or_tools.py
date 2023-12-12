# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:58:29 2023

@author: Delhivery
"""

import numpy as np
import pandas as pd
from itertools import chain
from ortools.linear_solver import pywraplp

all_combined3=pd.read_csv('all_combined3.csv')
load_matrix=pd.read_csv('load_matrix.csv')
splits_matrix=pd.read_csv('splits_matrix.csv')
y_fix=pd.read_csv('y_fix.csv')
upper_cap=pd.read_csv('upper_cap.csv')

max_check=9

"""
PART 4: RUNNING THE MODEL AND COMPILING THE RESULT
"""
all_combined3 = all_combined3.replace(r'^\s*$', np.nan, regex=True)
all_combined3 = all_combined3.fillna(value=np.nan)

def model_pre_processing():
    def per_match(a, b):
        if a[:2] == 'S_' or a[:2] == 'E_':
            a = a[2:]
        if b[:2] == 'S_' or b[:2] == 'E_':
            b = b[2:]
        n = 0
        if len(a) <= len(b):
            for i in range(len(a)):
                if a[i] == b[i]: n += 1
        if len(a) > len(b):
            for i in range(len(b)):
                if a[i] == b[i]: n += 1
        return n

    split = splits_matrix[['Scan_Location', 'Splits']].values.tolist()
    load = load_matrix[['Source', 'Destination', 'Load', 'Mot']].values.tolist()
    k = max_check
    #max_check=10
    col_read = []
    col_read.append('oc_name')
    col_read.append('cn_name')
    col_read.append('mot')
    for i in range(1, k + 1):
        col_read.append('stop_name_' + str(i))
        if i != 1: col_read.append('mot' + str(i))
        col_read.append('time_diff' + str(i))
        col_read.append('route_flag' + str(i))
        col_read.append('ifdct' + str(i))
    data2 = all_combined3.copy()
    hop = data2[col_read].values.tolist()
    print('upload done')
    # check surface and express mot
    # file=open('mot_issue_pairs.txt', 'w')
    for i in range(len(hop)):
        if hop[i][2] == 'surface' and str(hop[i][2]) != 'nan':
            for j in range(7, len(hop[i]), 5):
                if hop[i][j + 1] != 'surface' and str(hop[i][j + 1]) != 'nan':
                    #               file.write(str(i)+'  ::  '+str(hop[i]))
                    #               file.write('error surface mot but bag type express at'+ hop[i][j])
                    hop[i][j + 1] = 'surface'
        if hop[i][2] == 'express' and str(hop[i][2]) != 'nan':
            k = 0
            for j in range(7, len(hop[i]), 5):
                if str(hop[i][j + 4]) != 'nan': k += 1
                if hop[i][j + 1] == 'surface' and str(hop[i][j + 1]) != 'nan':
                    if k != 0:
                        #                   file.write(str(i)+ '  ::  '+ str(hop[i]))
                        #                   file.write('error express mot but bag type surface after flight at'+ hop[i][j])
                        hop[i][j + 1] = 'express'
                if hop[i][j + 1] == 'express' and str(hop[i][j + 1]) != 'nan':
                    if k == 0:
                        #                   file.write(str(i)+ '  ::  '+ str(hop[i]))
                        #                   file.write('error express mot but bag type express before flight at'+ hop[i][j])
                        hop[i][j + 1] = 'surface'

    fin = []
    for i in range(len(hop)):
        # print('i :',i)
        ad = []
        ad.append(hop[i][0])
        if (hop[i][2] == 'surface'): ad.append('S_' + hop[i][1])
        if (hop[i][2] == 'express'): ad.append('E_' + hop[i][1])
        k = 0
        for j in range(len(load)):
            if load[j][0] == hop[i][0] and load[j][1] == hop[i][1] and load[j][3] == hop[i][2]:
                ad.append(load[j][2])
                k += 1
        if k == 0: continue
        if k >= 2:
            print('issue of OD pair repitition')
            break
        k = 0
        for j in range(0, 8):
            if hop[i][10 + k] == 1 and hop[i][7 + k] != hop[i][0]:
                if hop[i][7 + k] != hop[i][1] and 'S_' + hop[i][7 + k] not in ad and 'E_' + hop[i][7 + k] not in ad:
                    if (hop[i][8 + k] == 'surface'): ad.append('S_' + hop[i][7 + k])
                    if (hop[i][8 + k] == 'express'): ad.append('E_' + hop[i][7 + k])
                if hop[i][7 + k] == hop[i][1]:
                    if (hop[i][8 + k] == 'surface'): ad.append('S_' + hop[i][7 + k])
                    if (hop[i][8 + k] == 'express'): ad.append('E_' + hop[i][7 + k])

            k += 5
        # for j in range(len(ad),11):
        #     ad.append('')
        fin.append(ad)
    # print(fin)
    temp = list(chain.from_iterable(fin))  # convert to one dimensional array
    temp = list(filter(None, temp))  # removing blank entries
    temp = set(temp)  # remove repetitions
    sp = []
    dont = []
    for i in temp:
        if type(i) != type(1) and type(i) != type(1.0):
            k = i.replace('S_', '')
            k = k.replace('E_', '')
            if dont.count(k) != 0: continue
            ad = []
            for n in range(len(split)):
                if split[n][0] == k:
                    ad = [k, split[n][1]]
                    dont.append(k)
                    break
            if len(ad) > 0: sp.append(ad)
    sp.sort()

    # file= open("test_fin.txt",'w')
    # for i in fin:
    #     for j in i:
    #         file.write(str(j)+', ')
    #     file.write('\n')

    def find_max_list(list):
        list_len = [len(i) for i in list]
        return max(list_len)

    # print output#
    k = find_max_list(fin)

    edg = []
    for k in range(len(fin)):
        fl = []
        ab = []
        for i in range(11, len(hop[k]), 5):
            if str(hop[k][i]) != 'nan' and str(hop[k][i]) != '':
                if i == 11:
                    for j in range(10, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 16:
                    for j in range(15, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 21:
                    for j in range(20, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 26:
                    for j in range(25, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 31:
                    for j in range(30, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 36:
                    for j in range(35, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 41:
                    for j in range(40, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
                if i == 46:
                    for j in range(45, len(hop[k]), 5):
                        if hop[k][j] == 1:
                            fl.append(fin[k].index('E_' + hop[k][j - 3], 3, len(fin[k])))
                            break
        if len(fin[k]) == 4:
            ad = []
            ad = [fin[k][0], fin[k][len(fin[k]) - 1]]
            ab.append(ad)
            edg.append(ab)
            continue
        # if k==16116 or k==7679 or k== 16690:
        #     print(fl)
        #     print(fin[k])
        #     print(hop[k])
        ad = []
        ad = [fin[k][0], fin[k][len(fin[k]) - 1]]
        ab.append(ad)
        for i in range(3, len(fin[k]) - 1):
            ad = []
            ad = [fin[k][0], fin[k][i]]
            ab.append(ad)
            for j in range(i + 1, len(fin[k])):
                if len(fl) > 0:
                    if i < fl[len(fl) - 1]:
                        ad = []
                        ad.append(fin[k][i][2:])
                        ad.append(fin[k][j])
                        ab.append(ad)
                        continue
                    if i >= fl[len(fl) - 1]:
                        ad = []
                        ad.append(fin[k][i][2:])
                        ad.append('S_' + fin[k][j][2:])
                        ab.append(ad)
                        continue
                ad = []
                ad.append(fin[k][i][2:])
                ad.append(fin[k][j])
                ab.append(ad)
        # if k==16116 or k==7679 or k== 16690:
        #     print(k,' :: ', ab)
        edg.append(ab)
    # print(edg)
    # exit()
    # B making
    B = []
    for k in range(len(fin)):
        ab = []
        ad = list(np.zeros(len(edg[k])))
        for m in range(len(edg[k])):
            if edg[k][m][0] == fin[k][0]:
                ad[m] = 1
        ab.append(ad)
        for i in range(3, len(fin[k])):
            ad = list(np.zeros(len(edg[k])))
            for m in range(len(edg[k])):
                if edg[k][m][0] == fin[k][i][2:]: ad[m] = 1
                if edg[k][m][1][2:] == fin[k][i][2:]: ad[m] = -1
            ab.append(ad)
        B.append(ab)
    # print(B)
    b = []
    for k in range(len(fin)):
        ab = list(np.zeros(len(B[k])))
        ab[0] = fin[k][2]
        ab[len(ab) - 1] = -fin[k][2]
        b.append(ab)
    # print(b)
    ori = []
    des = []
    for i in fin:
        if ori.count(i[0]) == 0:
            ori.append(i[0])
        for j in range(3, len(i)):
            if des.count(i[j]) == 0:
                des.append(i[j])

    # adding new hop cities as origins
    for i in fin:
        for j in range(3, len(i) - 1):
            if ori.count(i[j][2:]) == 0:
                ori.append(i[j][2:])
    ori.sort()
    # print(ori)
    # print(des)
    sp[6][1]=25
    return edg, sp, ori, des, fin, B, b


edg, sp, ori, des, fin, B, b = model_pre_processing()

model = pywraplp.Solver.CreateSolver('SCIP')

infinity = model.infinity()

x = []
for k in range(len(edg)):
    x.append([])
    for m in range(len(edg[k])):
        x[k].append(
            model.NumVar(0, infinity, str(k) + ',' + edg[k][m][0] + ',' + edg[k][m][1]))
        
y = []
for k in range(len(ori)):
    y.append([])
    for m in range(len(des)):
        y[k].append(model.BoolVar('y_' + ori[k] + ' -> ' + des[m]))
        
        
Q = []
cap_extra=[]
for i in range(len(ori)):
    Q.append(model.NumVar(0, infinity,'q_' + ori[i]))
    cap_extra.append(model.NumVar(0, infinity, 'cap_extra_' + ori[i]))

ndont=[]
for i in y_fix:
    if i[0] in ori and i[1] in des:
        model.Add(y[ori.index(i[0])][des.index(i[1])]==1)
    else:
        ndont.append(i)
print('con 0 is made')


for k in range(len(fin)):
    for i in range(len(B[k])):
        exp1=sum(x[k][m] * B[k][i][m] for m in range(len(B[k][i])))      
        model.Add(exp1 == b[k][i],'bal' + str(k))
print('con 1 made')


exp = []
for i in range(len(ori)):
    exp.append([])
    for j in range(len(des)):
        exp[i].append(0)
print('expr is made')

load_max = 0
for i in fin: 
    load_max += i[2]
print('load_max', load_max)
for k in range(len(edg)):
    for m in range(len(edg[k])):
        i = ori.index(edg[k][m][0])
        j = des.index(edg[k][m][1])
        exp[i][j] += x[k][m]

print('expr seted')
for i in range(len(ori)):
    for j in range(len(des)):
        model.Add(exp[i][j] <= load_max * y[i][j])

print('con 2 made')
for i in range(len(ori)):
    exp2 =0
    for j in range(len(des)):
        exp2 += y[i][j]
        
    for k in range(len(sp)):
        if sp[k][0] == ori[i]: break
    model.Add(exp2 <= sp[k][1], "lim" + ori[i])
    
    # if ori[i]!=sp[i][0] :print('issue', ori[i], ' :: ',sp[i][0])

for i in upper_cap['center_name']:
    cap = upper_cap.loc[upper_cap['center_name']==i,'load']
    cap=cap.reset_index(drop=True)
    k = ori.index(i)
    exp3 =0
    for j in range(len(des)):
        exp3 += exp[k][j]
    model.Add(exp3 <= cap[0]+cap_extra[k], "max_processing_" + i)

total =0
for k in range(len(edg)):
    for m in range(len(edg[k])):
        total+=x[k][m]

T2 = 0
for i in range(len(ori)):
    T2 += Q[i]

T3 = 0
for i in range(len(ori)):
    T3 += cap_extra[i]

    
runtime=900000
objective=total

model.set_time_limit(runtime)

model.Minimize(objective)

status = model.Solve()
#model.Objective().Value()

#### Get Cap_Extra Values ###

cap_extra_values={}

for i in range(len(ori)):
    cap_extra_values[ori[i]]=cap_extra[i].solution_value()

cap_extra_df=pd.DataFrame({'center_name':list(cap_extra_values.keys()),
                          'load':list(cap_extra_values.values())})
new_cap_limi=pd.merge(upper_cap,cap_extra_df,on='center_name')

new_cap_limi['load']=new_cap_limi['load_x']+new_cap_limi['load_y']
new_cap_limi=new_cap_limi[['center_name','load']]

############ RUN AGAIN WITH REVISED CAPACITY ###########

model2 = pywraplp.Solver.CreateSolver('SCIP')

infinity = model2.infinity()

x = []
for k in range(len(edg)):
    x.append([])
    for m in range(len(edg[k])):
        x[k].append(
            model2.NumVar(0, infinity, str(k) + ',' + edg[k][m][0] + ',' + edg[k][m][1]))
        
y = []
for k in range(len(ori)):
    y.append([])
    for m in range(len(des)):
        y[k].append(model2.BoolVar('y_' + ori[k] + ' -> ' + des[m]))
        
        
Q = []
cap_extra=[]
for i in range(len(ori)):
    Q.append(model2.NumVar(0, infinity,'q_' + ori[i]))
    cap_extra.append(model2.NumVar(0, infinity, 'cap_extra_' + ori[i]))

ndont=[]
for i in y_fix:
    if i[0] in ori and i[1] in des:
        model2.Add(y[ori.index(i[0])][des.index(i[1])]==1)
    else:
        ndont.append(i)
print('con 0 is made')


for k in range(len(fin)):
    for i in range(len(B[k])):
        exp1=sum(x[k][m] * B[k][i][m] for m in range(len(B[k][i])))      
        model2.Add(exp1 == b[k][i],'bal' + str(k))
print('con 1 made')

exp
exp = []
for i in range(len(ori)):
    exp.append([])
    for j in range(len(des)):
        exp[i].append(0)
print('expr is made')

load_max = 0
for i in fin: 
    load_max += i[2]
print('load_max', load_max)
for k in range(len(edg)):
    for m in range(len(edg[k])):
        i = ori.index(edg[k][m][0])
        j = des.index(edg[k][m][1])
        exp[i][j] += x[k][m]

print('expr seted')
for i in range(len(ori)):
    for j in range(len(des)):
        model2.Add(exp[i][j] <= load_max * y[i][j])

print('con 2 made')
for i in range(len(ori)):
    exp2 =0
    for j in range(len(des)):
        exp2 += y[i][j]
        
    for k in range(len(sp)):
        if sp[k][0] == ori[i]: break
    model2.Add(exp2 <= sp[k][1], "lim" + ori[i])
    
    # if ori[i]!=sp[i][0] :print('issue', ori[i], ' :: ',sp[i][0])

for i in new_cap_limi['center_name']:
    cap = new_cap_limi.loc[new_cap_limi['center_name']==i,'load']
    cap=cap.reset_index(drop=True)
    k = ori.index(i)
    exp3 =0
    for j in range(len(des)):
        exp3 += exp[k][j]
    model2.Add(exp3 <= cap[0], "max_processing_" + i)

total =0
for k in range(len(edg)):
    for m in range(len(edg[k])):
        total+=x[k][m]

T2 = 0
for i in range(len(ori)):
    T2 += Q[i]

    
runtime=900000
objective=total

model2.set_time_limit(runtime)

model2.Minimize(objective)
status = model2.Solve()

###### GET OUTPUT VALUES #####

y_sol={}

for i in range(len(ori)):
    for j in range(len(des)):
        if y[i][j].solution_value()>0:
            y_sol[y[i][j]]=y[i][j].solution_value()


x_sol={}
for k in range(len(edg)):
    for m in range(len(edg[k])):
       if x[k][m].solution_value()>0.001:
        x_sol[x[k][m]]=x[k][m].solution_value()
        
dd=[]
for nam_load in x_sol.items():
    nam=str(nam_load[0])
    load=nam_load[1]
    if load <=0.001:continue
    nam = nam.split(',')
    ad=[]
    ad.append(nam[1])
    ad.append(fin[int(nam[0])][1][2:])
    if fin[int(nam[0])][1][:2] == 'S_': ad.append('surface')
    if fin[int(nam[0])][1][:2] == 'E_': ad.append('express')
    ad.append(nam[2][2:])
    if nam[2][:2] == 'S_': ad.append('surface')
    if nam[2][:2] == 'E_': ad.append('express')
    ad.append(int(load))
        
    dd.append(nam[0])

#### MODEL POST PROCESSING ####

df345 = pd.DataFrame(dd, columns=['oc', 'cn', 'mot', 'bag_des', 'bag_type', 'load'])
# df345.to_csv("com_model_output.csv", index=False)
# data35=pd.read_csv('oc_cn_output_flag_feb.csv')
# dew=data35[['oc_name','cn_name', 'stop_name_1', 'time_diff1', 'route_flag1', 'stop_name_2', 'time_diff2', 'route_flag2','stop_name_3', 'time_diff3', 'route_flag3','stop_name_4', 'time_diff4', 'route_flag4','stop_name_5', 'time_diff5', 'route_flag5','stop_name_6', 'time_diff6', 'route_flag6','stop_name_7', 'time_diff7', 'route_flag7','stop_name_8', 'time_diff8', 'route_flag8','stop_name_9', 'time_diff9', 'route_flag9']].values.tolist()
# exit()
col_new = []
col_new.append('oc_name')
col_new.append('cn_name')
for i in range(1, max_check + 1):
    col_new.append('stop_name_' + str(i))
    col_new.append('time_diff' + str(i))
    col_new.append('route_flag' + str(i))
data35 = all_combined3.copy()
dew = data35[col_new].values.tolist()

sorter = []
for k in range(len(sp)):
    if sp[k][1] >= 100: sorter.append(sp[k][0])
dd.sort(key=lambda x: (x[0], x[1], x[2]))
pin = []
# mes=[]

dd_new = df345.copy()
dd_new = dd_new.drop(columns=["load"])
dd_new = dd_new.drop_duplicates()
dd_new = dd_new.groupby(["oc", "cn", "mot"], as_index=False).size()
dd_new = dd_new[dd_new["size"] != 1]
dd_new["Index"] = dd_new["oc"] + "_" + dd_new["cn"] + "_" + dd_new["mot"]
df345["Index"] = df345["oc"] + "_" + df345["cn"] + "_" + df345["mot"]
dd_new = dd_new.merge(df345[["Index", "bag_des", "bag_type", "load"]], on="Index", how="left")       
dd_new['oc']=dd_new['oc'].astype(str)

dd_new = dd_new.drop(columns=["size", "Index"])
pin = dd_new.values.tolist()

print(pin)
print('pin size', len(pin))
# df=pd.DataFrame(pin,columns=['oc','cn','mot','bag_dest','bag_type', 'load'])
# df.to_csv("pin.csv", index=False)

# bad = ['[', ']']
rep = []

for it, i in enumerate(pin):
    for jt, j in enumerate(pin):
        if i == j: continue
        if i[0] != j[0] or i[1] != j[1] or i[2] != j[2]: continue
        if i[3] == j[3] and i[4] == j[4]: continue
        # sorter condition
        if sorter.count(i[3]) == 0 and sorter.count(j[3]) != 0:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(j[3])
            re.append('sorter')
            rep.append(re)
            pin[it][3] = j[3]
            pin[it][4] = j[4]
            continue
        if sorter.count(i[3]) != 0 and sorter.count(j[3]) == 0:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(i[3])
            re.append('sorter')
            rep.append(re)
            pin[jt][3] = i[3]
            pin[jt][4] = i[4]
            continue
        ln = dew[i[5]][dew[i[5]].index(i[3], 2, len(dew[i[5]]) - 1) + 1]            
        lm = dew[j[5]][dew[j[5]].index(j[3], 2, len(dew[j[5]]) - 1) + 1]

        if str(ln)== 'nan': ln=[]
        if str(lm)== 'nan': lm=[]

        print(dew[i[5]])
        print(i[5], ' ', ln)
        print(dew[j[5]])
        print(j[5], ' ', lm)
        if all([not elem for elem in ln ]) == True:
            if all([not elem for elem in lm ]) == False:
                re = []
                re.append(i[5])
                re.append(j[5])
                re.append(i[3])
                re.append(j[3])
                re.append(j[3])
                re.append('single_dwell')
                rep.append(re)
                pin[it][3] = j[3]
                pin[it][4] = j[4]
                continue
        if all([not elem for elem in lm ]) == True:
            if all([not elem for elem in ln ]) == False:
                re = []
                re.append(i[5])
                re.append(j[5])
                re.append(i[3])
                re.append(j[3])
                re.append(i[3])
                re.append('single_dwell')
                rep.append(re)
                pin[jt][3] = i[3]
                pin[jt][4] = i[4]
                continue
        if all([not elem for elem in ln ]) == True:
            if all([not elem for elem in lm ]) == True:
                print('in')
                nvol = 0
                mvol = 0
                for k in pin:
                    if k[0] == i[0] and k[1] == i[1] and k[2] == i[2] and k[3] == i[3] and k[4] == i[4]:
                        nvol += fin[k[5]][2]
                    if k[0] == j[0] and k[1] == j[1] and k[2] == j[2] and k[3] == j[3] and k[4] == j[4]:
                        mvol += fin[k[5]][2]

                if nvol >= mvol:
                    re = []
                    re.append(i[5])
                    re.append(j[5])
                    re.append(i[3])
                    re.append(j[3])
                    re.append(i[3])
                    re.append('vol')
                    re.append(nvol)
                    re.append(mvol)
                    rep.append(re)
                    pin[jt][3] = i[3]
                    pin[jt][4] = i[4]

                    continue
                if nvol < mvol:
                    re = []
                    re.append(i[5])
                    re.append(j[5])
                    re.append(i[3])
                    re.append(j[3])
                    re.append(j[3])
                    re.append('vol')
                    re.append(nvol)
                    re.append(mvol)
                    rep.append(re)
                    pin[it][3] = j[3]
                    pin[it][4] = j[4]
                    continue
        print('out')
        lm = [item for sublist in lm for item in sublist]
        ln = [item for sublist in ln for item in sublist]
        # lm = str(lm)
        # ln= str(ln)
        # for k in bad: ln = ln.replace(k, '')
        # for k in bad: lm = lm.replace(k, '')
        # ln = ln.split(",")
        # lm = lm.split(",")
        print('ln', ln)
        ln = [float(k) for k in ln]
        lm = [float(k) for k in lm]
        # print('for', i[3], ln, '  ',sum(ln))
        # print('for', j[3], lm,'  ',sum(lm))
        for (k, o) in enumerate(ln):
            if o < 3: ln[k] = 24 - 3
        for (k, o) in enumerate(lm):
            if o < 3: lm[k] = 24 - 3
        a = 0
        b = 0
        for k in ln:
            if k >= 3 and k < 6:
                a = 1
                break
        for k in lm:
            if k >= 3 and k < 6:
                b = 1
                break
        if a == 1 and b == 0:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(i[3])
            re.append('dwell_bet_3_6')
            rep.append(re)
            pin[jt][3] = i[3]
            pin[jt][4] = i[4]
            continue
        if a == 0 and b == 1:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(j[3])
            re.append('dwell_bet_3_6')
            rep.append(re)
            pin[it][3] = j[3]
            pin[it][4] = j[4]
            continue
        lnn = 100000
        for k in ln:
            if lnn >= k and k >= 6: lnn = k
        lmm = 100000
        for k in lm:
            if lmm >= k and k >= 6: lmm = k
        if lnn > lmm:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(i[3])
            re.append('dwell time')
            re.append(lnn)
            re.append(lmm)
            rep.append(re)
            pin[jt][3] = i[3]
            pin[jt][4] = i[4]

            continue
        if lnn < lmm:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(j[3])
            re.append('dwell time')
            re.append(lnn)
            re.append(lmm)
            rep.append(re)
            pin[it][3] = j[3]
            pin[it][4] = j[4]

            continue
        nvol = 0
        mvol = 0
        for k in pin:
            if k[0] == i[0] and k[1] == i[1] and k[2] == i[2] and k[3] == i[3] and k[4] == i[4]:
                nvol += fin[k[5]][2]
            if k[0] == j[0] and k[1] == j[1] and k[2] == j[2] and k[3] == j[3] and k[4] == j[4]:
                mvol += fin[k[5]][2]
        if nvol >= mvol:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(i[3])
            re.append('vol')
            re.append(nvol)
            re.append(mvol)
            rep.append(re)
            pin[jt][3] = i[3]
            pin[jt][4] = i[4]

            continue
        if nvol < mvol:
            re = []
            re.append(i[5])
            re.append(j[5])
            re.append(i[3])
            re.append(j[3])
            re.append(j[3])
            re.append('vol')
            re.append(nvol)
            re.append(mvol)
            rep.append(re)
            pin[it][3] = j[3]
            pin[it][4] = j[4]
            continue
# print(pin)
print('pin size', len(pin))
for i in pin:
    # print('which ',i)
    if (len(i) == 6): del i[5]
    # print('after',i)
# print (pin,'\n','\n')
pina = list(set(tuple(element) for element in pin))
pinal = np.array([*pina])
print('pinal :', pinal)
additions = []
for k in range(len(pinal)):
    for n in range(len(dd)):
        if dd[n][0] == pinal[k][0] and dd[n][1] == pinal[k][1] and dd[n][2] == pinal[k][2]:
            # check whether m is presnt in the route
            if pinal[k][4] == 'express': nam = 'E_' + pinal[k][3]
            if pinal[k][4] == 'surface': nam = 'S_' + pinal[k][3]
            # print(dd[n][5])
            # print(fin[dd[n][5]])
            if fin[dd[n][5]].count(nam) == 0: continue
            ca = dd[n][3]
            cf = dd[n][4]
            dd[n][3] = pinal[k][3]
            dd[n][4] = pinal[k][4]
            for p in range(len(dd)):
                if p == n: continue
                if dd[p][5] != dd[n][5]: continue
                if dd[p].count(ca) == 0: continue
                if dd[p].index(ca) == 0:
                    dd[p][0] = pinal[k][3]
                    for q in edg[dd[p][5]]:
                        # if dd[p][5] == 7679 or dd[p][5] == 16116 or dd[p][5] == 16690: print(q)
                        if q[0] == pinal[k][3] and q[1] == 'E_' + dd[p][3]:
                            dd[p][3] = q[1][2:]
                            dd[p][4] = 'express'
                            # if dd[p][5] == 7679 or dd[p][5] == 16116 or dd[p][5] == 16690: print(q,' :: ',dd[p])
                            break
                        if q[0] == pinal[k][3] and q[1] == 'S_' + dd[p][3]:
                            dd[p][3] = q[1][2:]
                            dd[p][4] = 'surface'
                            # if dd[p][5] == 7679 or dd[p][5] == 16116 or dd[p][5] == 16690: print(q, ' :: ', dd[p])
                            break
                    # if dd[p][5]==7679 or dd[p][5]==16116 or dd[p][5]==16690: print('final: ',dd[p][5],' :: ',dd[p])
                    continue
                if dd[p].index(ca) == 3:
                    dd[p][3] = pinal[k][3]
                    dd[p][4] = pinal[k][4]
                if dd[p][1] == ca:
                    ad = []
                    ad.append(pinal[k][3])
                    ad.append(pinal[k][1])
                    ad.append(pinal[k][2])
                    ad.append(ca)
                    ad.append(cf)
                    ad.append(dd[n][5])
                    print(ad)
                    additions.append(ad)
print('additions', len(additions))
print(additions)
for i in range(len(additions)):
    dd.append(additions[i][:])
n = 1
while n == 1:
    for k in range(len(dd)):
        if dd[k][0] == dd[k][3]:
            print(dd[k])
            del (dd[k])
            break
    print(k)
    if k == len(dd) - 1: break

df2 = pd.DataFrame(rep, columns=['load_1', 'load_2', 'bag_des_1', 'bag_dest_2', 'selected', 'remark', '1_dwell/vol',
                                 '2_dwell/vol'])
final = pd.DataFrame(dd, columns=['oc', 'cn', 'mot', 'Bag_Destination', 'Bag_Type', 'load'])




            