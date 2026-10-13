#!/usr/bin/env node
// RC教師(単周・停止から)の各ゲート通過時刻と速度、路面ごとのフレーム数を表示する(高速版コア使用)。
// 使い方: node probe/teacher_splits.mjs
const simRoot=new URL('../sim/',import.meta.url).pathname;
const {loadCourseWorld}=await import(simRoot+'qa/load-course-world.mjs');
const fast=await import(simRoot+'fast-sim-core.mjs');
const {createRcTrainingTeacher}=await import(simRoot+'dist/rc-speed-controller.js');
const world=await loadCourseWorld();
const RC={dt:1/30,maxSpeed:7.24,maxSteer:.52,acceleration:1.2,steerRate:1.2,vehicleLength:.42,vehicleWidth:.19,wheelbase:.26,stopOnCollision:true,frictionMultiplier:1,longitudinalModel:'real-drive-response-estimate-v1',driveTau:.4,driveAccelerationCap:6,brakeDeceleration:1.2};
const sim=fast.createSimulation(world,{...RC,seed:1,timeLimit:20,noiseStd:0});
const t=createRcTrainingTeacher(world,sim.snapshot().config,{gate:0,left_offset_m:0,left_yaw_deg:0});
const targets=sim.targets;let prev=0;const rows=[];let maxv=0,surf={},cmds=[];
let s=sim.snapshot();
while(!s.state.terminal){const a=t.action(s);cmds.push(a);sim.step(a,true);s=sim.peek();maxv=Math.max(maxv,s.state.speed);surf[s.state.surface]=(surf[s.state.surface]||0)+1;
 if(s.metrics.route_index!==prev){rows.push([prev,s.state.sim_time.toFixed(3),s.state.speed.toFixed(2),s.state.x.toFixed(2),s.state.z.toFixed(2)]);prev=s.metrics.route_index;}}
const f=x=>x.toFixed(3);
console.log('gate|target(x,z)|from|seg_m|teacher_t|v');
let len=0;
for(let i=0;i<targets.length;i++){const b=i?targets[i-1]:world.route[0],p=targets[i],L=Math.hypot(p[0]-b[0],p[1]-b[1]);len+=L;const r=rows.find(r=>r[0]===i);console.log(`${i}|(${f(p[0])},${f(p[1])})|${L.toFixed(3)}|${r?r[1]:'-'}|${r?r[2]:'-'}`);}
console.log('route length',len.toFixed(3),'max speed',maxv.toFixed(3),'surfaces frames',JSON.stringify(surf),'shortcut',JSON.stringify(sim.peek().metrics.required_crossing_events));
console.log('max cmd speed',Math.max(...cmds.map(c=>c.speed)).toFixed(3),'mean speed',(s.metrics.distance/s.metrics.elapsed).toFixed(3),'distance',s.metrics.distance.toFixed(3),'max lat acc',s.metrics.max_lateral_acceleration.toFixed(3),'traction-limited substeps',s.metrics.traction_limited_steps);
