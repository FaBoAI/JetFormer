#!/usr/bin/env python3
# sim_dart_20261014 の写し(ベストラップの verify_rc_teacher_dataset_rl.py から、教師の id の行だけを D1〜D3 の 3 つの id に変えた)。
# 元: 記録の段(2026-10-14)の写し: ユーザーの qa/verify-rc-teacher-dataset.py を 1 行だけ変えたもの。
# 変えたのは「Teacher marker」の行だけ(元: context.teacher.not_learned_policy が真 → RL 版: context.teacher.id が rl-bestlap-ppo-v1 で
# uses_ground_truth が真)。センサー・時刻・IMU・PWM・未来ラベル・画像の重複の検査はすべて元のまま。元のファイルは変えていない。
"""Independent saved-image/sensor/timestamp audit. Does not claim pixel re-render parity."""
from pathlib import Path
import argparse, hashlib, json, math
import cv2
import numpy as np

def require(condition, message):
    if not condition: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def near(a,b,label,tol=1e-9):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    error=float(np.max(np.abs(a-b),initial=0)); require(error<=tol,label+f' error {error}');return error

def verify(root):
    root=Path(root).resolve()
    if not (root/'simulation/session.json').is_file():
        roots=list(root.glob('*/simulation/session.json'));require(len(roots)==1,'Select one scene');root=roots[0].parent.parent
    def read(p):return json.loads((root/p).read_text())
    manifest=read('manifest.json'); entries=manifest['files'];names=[v['path'] for v in entries]
    require(len(names)==len(set(names)),'Manifest duplicate')
    for entry in entries:
        f=(root/entry['path']).resolve();require(f.is_relative_to(root),'Manifest path escaped');require(f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],'Manifest integrity '+entry['path'])
    s=read('simulation/session.json');r=read('simulation/evaluation-report.json');card=read('dataset_card.json');rows=[json.loads(l) for l in (root/'simulation/frames.jsonl').read_text().splitlines()]
    require(s['controller']==r['context']['controller']=='demo' and s['synthetic'],'Not synthetic teacher');require(s['context']['teacher'].get('id') in ('rl-bestlap-ppo-dart-v1','rl-line-margin-c03-v1','rl-line-margin-c03-dart-v1') and s['context']['teacher'].get('uses_ground_truth') is True,'Teacher marker')  # RL版: 元は not_learned_policy(台本の RC 教師の印)。RL の教師は学習した方策なので、教師の id と正解の状態を使う印を確かめる
    require(card['num_modes']==9 and card['mode_frames']=={'8':600} and card['training_label_mode']=='synthetic_controller','Genre/label contract')
    files=list((root/'aligned').glob('*.npz'));require(len(files)==1,'One NPZ');a=np.load(files[0],allow_pickle=False)
    require(len(rows)==len(r['actions'])==len(r['trace'])==manifest['frames']==600,'Coverage')
    require(all(len(a[k])==600 for k in a.files),'NPZ row coverage');require((a['mode']==8).all() and a['synthetic'].all() and not a['human_control'].any(),'Synthetic genre8 flags')
    require(a['vehicle_valid'].all() and a['erpm_valid'].all(),'Labels/eRPM validity');require(np.array_equal(a['future_valid'],np.arange(600)<590),'Future validity')
    require((a['tof_age_ms']==0).all() and (a['vehicle_age_ms']==0).all(),'Synchronous ages')
    inv=next(root.glob('v1.0-*')); sd={x['filename']:x for x in json.loads((inv/'sample_data.json').read_text())};ep={x['token']:x for x in json.loads((inv/'ego_pose.json').read_text())};sam={x['token']:x for x in json.loads((inv/'sample.json').read_text())}
    require(len(sd)==1200 and len(ep)==len(sam)==600,'nuScenes coverage')
    rawimu={v['utime']:v for v in json.loads(next((root/'can_bus').glob('*_ms_imu.json')).read_text())};rawvehicle=json.loads(next((root/'can_bus').glob('*_vehicle_monitor.json')).read_text())
    epoch=rows[0]['timestamp_us'];history=[((i-9)/100,[0,0,9.80665,0,0,0]) for i in range(10)];next_t=.01;image_hashes=set();tof_hashes=set();imu_error=0.;previous={'x':rows[0]['pose_pre_action']['x'],'z':rows[0]['pose_pre_action']['z'],'yaw':rows[0]['pose_pre_action']['yaw'],'speed':0,'sim_time':0}
    c=s['calibration'];m=s['model_config'];durations=[];confidence=set()
    for i,row in enumerate(rows):
        require(row['frame']==i,'Contiguous frame');require(row['timestamp_us']==int(a['timestamps_us'][i])==epoch+round(row['sim_time']*1e6),'Camera timestamp');near(row['sim_time'],i/30,'30Hz simtime')
        near([row['pose_pre_action'][k] for k in ['x','z','yaw']],[previous[k] for k in ['x','z','yaw']],'Preaction pose')
        for channel,rel,shape in [('CAM_FRONT',str(a['cam_files'][i]),(224,224,3)),('TOF_FRONT',str(a['cam_files'][i]).replace('CAM_FRONT','TOF_FRONT').replace('.jpg','.png'),(42,54))]:
            require(rel in names and rel in sd,'Covered sensor file');im=cv2.imread(str(root/rel),cv2.IMREAD_UNCHANGED);require(im is not None and im.shape==shape,'Sensor shape '+rel)
            entry=sd[rel];pose=ep[entry['ego_pose_token']];sample=sam[entry['sample_token']]
            require(entry['timestamp']==pose['timestamp']==sample['timestamp']==row['timestamp_us'],'Image/time/pose mapping')
            p=row['pose_pre_action'];near(pose['translation'],[p['x'],-p['z'],0],'Ego translation');near(pose['rotation'],[math.cos(p['yaw']/2),0,0,math.sin(p['yaw']/2)],'Ego orientation');require(pose['synthetic_ground_truth'],'Groundtruth pose marker')
            h=hashlib.sha256(im.tobytes()).hexdigest()
            if channel=='CAM_FRONT':image_hashes.add(h)
            else:
                require(im.dtype==np.uint16 and np.array_equal(im,a['tof'][i]),'Packed ToF exact PNG/NPZ parity');tof_hashes.add(h);confidence.update(int(x) for x in np.unique(im>>13))
        imu_expected=np.asarray([v for t,v in history],np.float32);times_expected=np.asarray([epoch+round(t*1e6) for t,v in history],np.int64)
        require(np.array_equal(times_expected,a['imu_timestamps_us'][i]),'Causal IMU timestamp synthesis');imu_error=max(imu_error,near(a['imu'][i],imu_expected,'IMU independently derived from motion',2e-6))
        require(np.max(times_expected)<=row['timestamp_us']+1 and row['timestamp_us']-max(times_expected)<=10001,'Causal fresh IMU')
        require(row['imu_padding_before_episode']==sum(t<0 for t,v in history),'IMU reset-padding marker')
        for t,v in zip(times_expected,a['imu'][i]):
            raw=rawimu[int(t)];near(v,raw['linear_accel']+raw['rotation_rate'],'Raw/aligned IMU',0)
        require(np.float32(previous['speed']*c['erpm_per_mps']*c['erpm_forward_sign'])==a['vehicle'][i,3],'Preaction speed/eRPM')
        fraction=np.clip(row['command_si']['steering']/c['max_steering_rad']/c['steering_sign'],-1,1);center=m['steer_center_us'];neutral=m['throttle_neutral_us'];speed=row['command_si']['speed']
        expected=[center+fraction*(c['pwm_max_us']-center if fraction>=0 else center-c['pwm_min_us']),neutral-min(1,speed/c['max_speed_mps'])*(neutral-c['pwm_min_us']) if speed>=0 else neutral+min(1,-speed/c['max_reverse_speed_mps'])*(c['pwm_max_us']-neutral)]
        near(row['applied_pwm_us'],expected,'Actual SI inverse PWM');require(np.array_equal(np.asarray(expected,np.float32),a['vehicle'][i,:2]),'Serialized PWM')
        rv=rawvehicle[i];require(rv['utime']==row['timestamp_us'] and rv['controller']=='demo' and rv['synthetic'] and not rv['human_control'],'Raw vehicle flags/time');near([rv[k] for k in ['steering_us','throttle_us','mode_us','erpm']],a['vehicle'][i],'Raw vehicle',0)
        require(np.array_equal(a['future_actions'][i],a['vehicle'][np.minimum(np.arange(i+1,i+11),599),:2]),'Future labels next1..10')
        current=row['state_post_action'];dt=current['sim_time']-previous['sim_time'];dyaw=math.atan2(math.sin(current['yaw']-previous['yaw']),math.cos(current['yaw']-previous['yaw']));yawrate=dyaw/dt;acc=(current['speed']-previous['speed'])/dt
        while next_t<=current['sim_time']+1e-9:
            history.append((next_t,[-current['speed']*yawrate,acc,9.80665,0,0,yawrate]));next_t+=.01
        history=history[-10:];previous=current
    require(len(image_hashes)>590 and len(tof_hashes)>590,'Unexpected frozen/repeated sensor frames')
    return {'schema':'rc-teacher-dataset-sensor-integrity/v1','pass':True,'scope':'Manifest integrity, image timestamp/ego-pose linkage, all JPEG dimensions/content variety, exact PNG/NPZ packed ToF equality, independently motion-derived causal100Hz IMU, actual PWM/eRPM and future labels. Does not independently re-render WebGL RGB/depth pixels or establish real sensor fidelity. Ground-truth teacher, not learned AI.', 'dataset':str(root),'manifest_sha256':sha(root/'manifest.json'),'manifest_files_checked':len(entries),'frames':600,'seconds':20,'genre_id':8,'num_modes':9,'future_valid_rows':590,'unique_decoded_camera_images':len(image_hashes),'unique_packed_tof_frames':len(tof_hashes),'confidence_codes':sorted(confidence),'raw_imu_rows':len(rawimu),'max_imu_float32_error':imu_error,'capture_times_us':[int(a['timestamps_us'][0]),int(a['timestamps_us'][-1])],'no_imu_reset_after_start':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--output',default='qa/rc-pilot-sensor-integrity.json');args=p.parse_args();result=verify(args.dataset);Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
