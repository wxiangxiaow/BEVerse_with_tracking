'''
Edit Date: 2022.09.21
Editor: wxiangxiaow
Input args: sample data token
Output args: ego pose and calibrate(camera) sensor in csv file
'''

from nuscenes.nuscenes import NuScenes
import pandas as pd
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/riku/workspace/nuscenes/trainval/', verbose=True)

for i in range(34149):
    data_path = '/home/riku/workspace/BEVerse/beverse_train_visualize/'+str(i)+'/'
    sample_data_token_path = 'takens.txt'
    path = data_path + sample_data_token_path

    token_file = open(path,'r')
    sample_data_token = token_file.readline()

    sample_data = nusc.get('sample_data', sample_data_token)

    calibrate_sensor_token = sample_data['calibrated_sensor_token']
    calibrate_sensor = nusc.get('calibrated_sensor', calibrate_sensor_token)
    sensor_translation = calibrate_sensor['translation']
    sensor_rotation = calibrate_sensor['rotation']

    ego_pose_token = sample_data['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    ego_translation = ego_pose['translation']
    ego_rotation = ego_pose['rotation']

    sample_token = sample_data['sample_token']
    sample = nusc.get('sample', sample_token)

    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)

    log_token = scene['log_token']
    log = nusc.get('log', log_token)
    location = log['location']

    ego = []
    for i in ego_rotation:
        ego.append(i)
    for i in ego_translation:
        ego.append(i)
    egos = pd.DataFrame(ego)
    egos.to_csv(data_path+'ego_pose.csv',columns=None, header=None, index=None)

    sensor = []
    for i in sensor_rotation:
        sensor.append(i)
    for i in sensor_translation:
        sensor.append(i)
    sensors = pd.DataFrame(sensor)
    sensors.to_csv(data_path+'sensor.csv',columns=None,index=None, header=None)

    with open(data_path+'location.txt','w') as f:
        f.write(location)
    
    with open(data_path+'scene_token.txt','w') as f:
        f.write(scene_token)
