import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from kalman import Kalman
import utils
from quaternions import Quaternion

dataroot = './beverse_val_visualize/'
# --------------------------------Kalman参数---------------------------------------
# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]])
# 控制输入矩阵B
B = None
# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性,
# 在跟踪任务当中，过程噪声来自于目标移动的不确定性（突然加速、减速、转弯等）
Q = np.eye(A.shape[0]) * 100
# 状态观测矩阵
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0]])
# 观测噪声协方差矩阵R，p(v)~N(0,R)
# 观测噪声来自于检测框丢失、重叠等
R = np.eye(H.shape[0]) * 0.01
# 状态估计协方差矩阵P初始化
P = np.eye(A.shape[0])
# -------------------------------------------------------------------------------
def get_img2ego_N_instance(file_name):
    csv = file_name+'logs.csv'
    f = pd.read_csv(csv,header=None)
    lines = f.values.tolist()
    return lines

def get_ego_post(file_name):
    ego = pd.read_csv(file_name+'ego_pose.csv',header=None)
    ego_pose = ego.values.tolist()
    ego_pose_rotation = np.array(ego_pose[:4]).reshape(1,4)[0]
    ego_pose_rotation = Quaternion(ego_pose_rotation[0], ego_pose_rotation[1],
                ego_pose_rotation[2], ego_pose_rotation[3]
    )
    ww, xx, yy, zz = ego_pose_rotation
    ego_pose_rotation_matrix = np.array([
        [1-yy**2*2-zz**2*2, 2*xx*yy-2*ww*zz, 2*xx*zz+2*ww*yy],
        [2*xx*yy+2*ww*zz, 1-xx**2*2-zz**2*2, 2*yy*zz-2*ww*xx],
        [2*xx*zz-2*ww*yy, 2*yy*zz+2*ww*xx, 1-xx**2*2-yy**2*2]
    ])[:2,:2]
    ego_pose_translation = np.array(ego_pose[4:]).reshape(3,1)
    return ego_pose_rotation_matrix, ego_pose_translation

def get_sensor_args(file_name):
    sensor = pd.read_csv(file_name+'sensor.csv',header=None)
    sensor = sensor.values.tolist()
    sensor_rotation = np.array(sensor[:4]).reshape(2,2)
    sensor_translation = np.array(sensor[4:]).reshape(3,1)
    h_camera = float(sensor_translation[-1][0])
    return h_camera

def get_scene_token(file_name):
    txt_file = open(file_name+'scene_token.txt','r')
    scene_token = txt_file.readline()
    return scene_token

def get_location(file_name):
    txt_file = open(file_name+'location.txt','r')
    location = txt_file.readline()
    return location

def cv_poly(x, coeff, img, color):
    x = list(set(np.int32(x)))
    x.sort()
    for i in range(len(x)-1):
        y1 = coeff[0]*x[i]**3 + coeff[1]*x[i]**2 +coeff[2]*x[i] +coeff[3]
        y2 = coeff[0]*x[i+1]**3 + coeff[1]*x[i+1]**2 +coeff[2]*x[i+1] +coeff[3]
        point1 = [x[i], int(y1)]
        point2 = [x[i+1], int(y2)]
        #cv2.line(img, point1, point2, color, 2, 4)
    return x
    

def get_curve(corr, img, dir):
    line = []    #分类以后的散点
    CX_K = []    #c0-c3
    xs_list = [] #
    
    corr=np.array(corr)
    ##################聚类并拟合#################################
    from sklearn.cluster import DBSCAN
    pred = DBSCAN(eps=50).fit(corr)
    #排除干扰点（标签值=-1）
    others = False
    #print(pred.labels_)
    for i in pred.labels_:
        if i == -1:
            others = True
    line_num = len(set(pred.labels_))
    if others:
        line_num = line_num - 1
    
    
    for i in range(line_num):
        lin = []
        fitting = {}
        for j in range(len(pred.labels_)):
            if pred.labels_[j] == i:
                lin.append(corr[j]) 
        line.append(lin)
        lin = np.array(lin)
        coeff = np.polyfit(lin[:,0],lin[:,1],3) #c0-c3
        CX_K.append(coeff)
        y_fig = lin[:,0]**3*coeff[0] + lin[:,0]**2*coeff[1]+coeff[2]*lin[:,0] + coeff[3]
        x_list = cv_poly(lin[:,0], coeff, img, (0,255,255))
        xs_list.append(x_list)
        # plt.plot(lin[:,0], y_fig)

    
    return CX_K, img, xs_list, line

def per_frame(dir):
    ego2img = get_img2ego_N_instance(dir)[:3]
    #################读入gt获得车道线投影############################
    gt_path = dir + 'gt.png'
    fig = cv2.imread(gt_path)     #fig -> gt
    fig_HSV = cv2.cvtColor(fig, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(fig_HSV, np.array([0,43,46]), np.array([10,255,255])) 
    mask2 = cv2.inRange(fig_HSV, np.array([156,43,46]), np.array([180,255,255]))
    b = mask1 + mask2
    b = cv2.bitwise_not(b)
    w, h = b.shape
    #################投影############################
    lam = []
    for i in range(int(w/2)):
        for j in range(h):
            if b[i][j] == 0 and i>0 and i<w-1 and j>0 and j<h-1:
                me = np.array([0.15*(-i+w/2),0.15*(-j+h/2),  0, 1.]).T
                m = np.array(ego2img) @ me
                m[2] = np.clip(m[2], a_min=1e-5, a_max=1e5) #lambda
                x1 = m[0]/m[2]
                x2 = m[1]/m[2]
                x1=np.clip(x1, a_min=-1e4, a_max=1e4)
                x2=np.clip(x2, a_min=-1e4, a_max=1e4)
                lam.append([x1, x2])

    img = cv2.imread(dir+'det_gt_CAM_FRONT.png')        #img -> front img
    W, H, _= img.shape
    
        
    #############拟合散点的曲线###############################
    corr = []
    for i in lam:
        if i[0]>0 and i[0]<1600 and i[1]>0 and i[1]<900:
            corr.append(i)
    if len(corr) > 0:
        CX_K, img, xs_list, points = get_curve(corr, img, dir)
    else:
        CX_K = None
        xs_list = None
        points = None
    return CX_K, img, xs_list, points

def main():
    '''
    观测:CX_K
    状态:state
    '''
    state_list = []
    ################VIDEO ARGS########################################
    # fps = 1
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video_path='/home/riku/workspace/BEVerse/Kalman/test_scene_4.mp4'
    # videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (640,480))
    bev_img_dir = './test4/'
    if not os.path.exists(bev_img_dir):
        os.mkdir(bev_img_dir)

    imgs = plt.figure(figsize=(19.6,12.8),dpi=1000)
    
    dir_name = '/home/riku/workspace/BEVerse/beverse_train_visualize/'
    for i in range(34149):        
        CX_K, img, xs_list, lines = per_frame(dir_name+str(i)+'/')
        location = get_location(dir_name+str(i)+'/')
        if location == 'singapore-queensto':

            # 预测
            if CX_K is not None:
                for target in state_list:
                    target.predict()
                # 最大权值匹配
                mea_list = [np.array([mea]).T for mea in CX_K]
                state_rem_list, mea_rem_list, match_list = Kalman.association(state_list, mea_list,xs_list)

                # 状态没匹配上的，更新一下，如果触发终止就删除
                state_del = list()
                for idx in state_rem_list:
                    status, _, _ = state_list[idx].update()
                    if not status:
                        state_del.append(idx)
                state_list = [state_list[i] for i in range(len(state_list)) if i not in state_del]
                # 量测没匹配上的，作为新生目标进行航迹起始
                for idx in mea_rem_list:
                    state_list.append(Kalman(A, B, H, Q, R, utils.mea2state(mea_list[idx]), P, xs_list[idx]))
            else:
                state_list = []
            
            # 画全局地图
            if lines is not None:
                #################BEV ARGS##############################
                datas = get_img2ego_N_instance(dir_name+str(i)+'/')
                img2ego = np.linalg.inv(np.array(datas[:4]))
                instance = np.array(datas[8:])
                instance = instance[:,:3]
                fy = float(instance[1][1])       #焦距
                v0 = float(instance[1][2])

                ego_pose_rotation, ego_pose_translation = get_ego_post(dir_name+str(i)+'/')
                h_camera = get_sensor_args(dir_name+str(i)+'/')
                # plt.scatter(ego_pose_translation[0], ego_pose_translation[1],c='r', marker='s')

                #################二次投影，前视->BEV##########################
                # plt.subplot(2,2,2)
                for state in state_list:
                    global_x = []
                    global_y = []
                    for j in state.x_list:
                        y = state.X_posterior[0]*(j**3) + state.X_posterior[1]*(j**2) +state.X_posterior[2]*j +state.X_posterior[3]
                        Zc = fy * h_camera / (y-v0)     #这里会出现深度Zc的估计误差，x范围缩小，y范围变大
                        ego_point = img2ego[:,:3] @ np.array([j, y, 1]).T *Zc
                        # print(ego_point)
                        global_point = ego_pose_rotation @ np.matrix(ego_point[:2]).T + ego_pose_translation[:2]
                        global_x.append(np.array(global_point.T)[0][0])
                        global_y.append(np.array(global_point.T)[0][1])
                        det_x = abs(np.array(global_point.T)[0][0] - ego_pose_translation[0])
                        det_y = abs(np.array(global_point.T)[0][1] - ego_pose_translation[1])
                        color_deep = np.exp(-1*(np.sqrt(det_x**2 + det_y**2) / 28.))
                        # plt.scatter(np.array(global_point.T)[0][0],np.array(global_point.T)[0][1],s=0.8,c='b',alpha=color_deep)
                    plt.plot(np.array(global_x),np.array(global_y),c='b')
            # plt.savefig(bev_img_dir+str(i)+'.png')

    plt.savefig('./singapore-queensto_train.png')
    plt.close()
        # for i in range(40):
        #     image = cv2.imread('/home/riku/workspace/BEVerse/Kalman/test4/'+str(i+120)+'.png')
        #     videoWriter.write(image)
        # videoWriter.release()


if __name__ == '__main__':
    main()
