// 確認 r3 の反事実用のコピー(2026-10-14)。元: course-simulator/dist/sim-core.js(sha256 aaeb2379…549e)。変えたのは横の上限の 1 行だけ(摩擦円・縦を優先)。
// Deterministic, planar kinematic model. SI units; yaw=0 points +x,
// positive yaw/steering turns toward -z. This approximates traction, not tyre dynamics.
export const DEFAULTS = Object.freeze({
  vehicleLength: .42, vehicleWidth: .19, wheelbase: .26,
  maxSpeed: 1.6, maxSteer: .52, acceleration: 1.2, steerRate: 1.2,
  frictionMultiplier: 1, noiseStd: 0, seed: 12345,
  timeLimit: 120, stopOnCollision: true,
  dt: 1/30, substep: .01, rayCount: 13, rayRange: 3, sensorHeight: .08,
  gateHalfWidth: .65,
});

const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const copy = value => JSON.parse(JSON.stringify(value));
const dist = (a, b) => Math.hypot(b[0]-a[0], b[1]-a[1]);
function number(value, fallback, lo, hi) {
  return finite(value) ? clamp(value, lo, hi) : fallback;
}
// Opt-in forward-drive estimate. Missing/explicit legacy keeps the historical
// normalized configuration byte-for-byte (including its keys).
export function normalizeLongitudinalConfig(input = {}) {
  const model = Object.hasOwn(input,'longitudinalModel') ? input.longitudinalModel : 'constant-acceleration-v1';
  const fields = {driveTau:[.40,.05,2],driveAccelerationCap:[6,.01,20],brakeDeceleration:[1.2,.01,20]};
  if (!['constant-acceleration-v1','real-drive-response-estimate-v1'].includes(model))
    throw new Error('Unknown longitudinalModel');
  if (model === 'constant-acceleration-v1') {
    if (Object.keys(fields).some(key => Object.hasOwn(input,key))) throw new Error('RC response parameters require real-drive-response-estimate-v1');
    return {};
  }
  const result = {longitudinalModel:model};
  for (const [key,[fallback,lo,hi]] of Object.entries(fields)) {
    const value = Object.hasOwn(input,key) ? input[key] : fallback;
    if (!finite(value) || value < lo || value > hi) throw new Error(`${key} must be finite in ${lo}..${hi}`);
    result[key] = value;
  }
  return result;
}
function normalizedConfig(input) {
  input = input && typeof input === 'object' ? input : {};
  const result = { ...DEFAULTS };
  const limits = {
    vehicleLength: [.05, 3], vehicleWidth: [.03, 2], wheelbase: [.03, 2],
    maxSpeed: [.05, 10], maxSteer: [.01, 1.2], acceleration: [.01, 20],
    steerRate: [.01, 20], frictionMultiplier: [.01, 5], noiseStd: [0, 2],
    timeLimit: [1/120, 3600], gateHalfWidth: [.05, 2], rayRange: [.1, 50],
    dt: [1/120, .1], sensorHeight: [.001, 3],
  };
  for (const [key, [lo, hi]] of Object.entries(limits)) {
    result[key] = number(input[key], DEFAULTS[key], lo, hi);
  }
  result.wheelbase = Math.min(result.wheelbase, result.vehicleLength);
  result.seed = finite(input.seed) ? Math.trunc(input.seed) >>> 0 : DEFAULTS.seed;
  result.stopOnCollision = typeof input.stopOnCollision === 'boolean' ? input.stopOnCollision : DEFAULTS.stopOnCollision;
  result.rayCount = Math.round(number(input.rayCount, DEFAULTS.rayCount, 3, 61));
  // Configured camera/ToF cadence is fixed per episode; motion intervals stay <=10ms.
  result.substep = Math.min(DEFAULTS.substep,result.dt);
  Object.assign(result,normalizeLongitudinalConfig(input));
  if (Object.hasOwn(input,'episodeMode')) {
    if (!['single-lap-v1','continuous-laps-v1'].includes(input.episodeMode)) throw new Error('Unknown episodeMode');
    if (input.episodeMode === 'continuous-laps-v1') result.episodeMode = input.episodeMode;
  }
  return result;
}
function point(value) {
  return Array.isArray(value) && value.length >= 2 && finite(value[0]) && finite(value[1]) ? [value[0], value[1]] : null;
}
function normalizedWorld(input = {}) {
  input = input && typeof input === 'object' ? input : {};
  const segments = (Array.isArray(input.segments) ? input.segments : []).flatMap((s, i) => {
    if (!s || typeof s !== 'object') return [];
    const a = point(s.a), b = point(s.b);
    return a && b ? [{ id: `segment-${i}`, a, b, radius: number(s.radius, .01, 0, 2), height: number(s.height, .1, 0, 10), kind: String(s.kind || 'rail') }] : [];
  }).filter(s => s.height > 0);
  const disks = (Array.isArray(input.disks) ? input.disks : []).flatMap((s, i) => {
    if (!s || typeof s !== 'object' || !finite(s.x) || !finite(s.z)) return [];
    const obstacle = { id: `disk-${i}`, x: s.x, z: s.z, radius: number(s.radius, .02, 0, 2), height: number(s.height, .1, 0, 10), kind: String(s.kind || 'post') };
    // scene.js historically stores every post assembly in `disks`. The foot is
    // actually an unrotated 0.16 x 0.15 m box; its old .11 m circumcircle stopped
    // cars before their bodies reached the rendered edges. Preserve IDs/height.
    if (obstacle.kind === 'post-foot') Object.assign(obstacle, { shape: 'rectangle', w: .16, d: .15 });
    return [obstacle];
  }).filter(s => s.height > 0);
  const surfaces = (Array.isArray(input.surfaces) ? input.surfaces : []).flatMap((s, i) => s && typeof s === 'object' && finite(s.x) && finite(s.z) && finite(s.w) && finite(s.d) && s.w > 0 && s.d > 0 ? [{ id: String(s.id || `surface-${i}`), x: s.x, z: s.z, w: s.w, d: s.d, mu: number(s.mu, .8, .01, 3) }] : []);
  const route = (Array.isArray(input.route) ? input.route : []).map(point).filter(Boolean).filter((p, i, all) => !i || dist(p, all[i-1]) > 1e-8);
  // The final gate returns to route[0]. Do not create an extra zero-length gate.
  if (route.length > 1 && dist(route[0], route[route.length-1]) < 1e-8) route.pop();
  const spawn = { x: number(input.spawn?.x, route[0]?.[0] || 0, -1e6, 1e6), z: number(input.spawn?.z, route[0]?.[1] || 0, -1e6, 1e6), yaw: wrap(number(input.spawn?.yaw, 0, -1e6, 1e6)) };
  const boundsInput = input.bounds || {};
  const bounds = {
    minX: number(boundsInput.minX, -20, -1e6, 1e6), maxX: number(boundsInput.maxX, 20, -1e6, 1e6),
    minZ: number(boundsInput.minZ, -20, -1e6, 1e6), maxZ: number(boundsInput.maxZ, 20, -1e6, 1e6),
  };
  if (bounds.maxX <= bounds.minX) { bounds.minX = -20; bounds.maxX = 20; }
  if (bounds.maxZ <= bounds.minZ) { bounds.minZ = -20; bounds.maxZ = 20; }
  const requiredCrossings = (Array.isArray(input.requiredCrossings) ? input.requiredCrossings : []).flatMap((g,i) => {
    if (!g || !finite(g.minX) || !finite(g.maxX) || !finite(g.z) || g.minX >= g.maxX) return [];
    return [{id:String(g.id || `crossing-${i}`),surface_id:String(g.surface_id || ''),minX:g.minX,maxX:g.maxX,z:g.z,direction:g.direction === 1 ? 1 : -1}];
  });
  return { segments, disks, surfaces, bounds, route, spawn, requiredCrossings };
}
// A recovery starts a new episode, not a checkpoint resume. The target cursor
// is independent of this episode's observed gate count and crossing evidence.
export function validateInitialState(input, worldInput, configInput = {}) {
  const keys = ['x','z','yaw','speed','steering','routeIndex'];
  if (!input || typeof input !== 'object' || Array.isArray(input)
      || Object.keys(input).length !== keys.length || keys.some(k => !Object.hasOwn(input,k) || !finite(input[k]))) {
    throw new Error('initial-state requires exactly finite x, z, yaw, speed, steering and routeIndex');
  }
  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);
  if (input.x < world.bounds.minX || input.x > world.bounds.maxX || input.z < world.bounds.minZ || input.z > world.bounds.maxZ)
    throw new Error('initial-state position is outside world bounds');
  if (input.yaw < -Math.PI || input.yaw > Math.PI) throw new Error('initial-state yaw must be radians in [-pi, pi]');
  if (Math.abs(input.speed) > config.maxSpeed || Math.abs(input.steering) > config.maxSteer)
    throw new Error('initial-state speed/steering exceed episode limits');
  if (!Number.isInteger(input.routeIndex) || world.route.length < 2 || input.routeIndex < 0 || input.routeIndex >= world.route.length)
    throw new Error('initial-state routeIndex must index targets=[...route.slice(1),route[0]]');
  return Object.fromEntries(keys.map(k => [k,input[k]]));
}
function hash(value) {
  let h = 2166136261;
  for (const c of JSON.stringify(value)) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
  return h.toString(16).padStart(8, '0');
}
function pointSegmentDistanceSquared(p, a, b) {
  const dx = b[0]-a[0], dz = b[1]-a[1], len2 = dx*dx+dz*dz;
  const t = len2 ? clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dz)/len2, 0, 1) : 0;
  return (p[0]-a[0]-t*dx)**2+(p[1]-a[1]-t*dz)**2;
}
function localPoint(p, state) {
  const dx = p[0]-state.x, dz = p[1]-state.z, c = Math.cos(state.yaw), s = Math.sin(state.yaw);
  return [dx*c-dz*s, dx*s+dz*c];
}
function pointRectDistanceSquared(p, hx, hz) {
  return Math.max(0, Math.abs(p[0])-hx)**2+Math.max(0, Math.abs(p[1])-hz)**2;
}
function segmentIntersectsRect(a, b, hx, hz) {
  let low = 0, high = 1;
  for (const [origin, delta, half] of [[a[0], b[0]-a[0], hx], [a[1], b[1]-a[1], hz]]) {
    if (Math.abs(delta) < 1e-12) { if (Math.abs(origin) > half) return false; }
    else {
      let t0 = (-half-origin)/delta, t1 = (half-origin)/delta;
      if (t0 > t1) [t0, t1] = [t1, t0];
      low = Math.max(low, t0); high = Math.min(high, t1);
      if (low > high) return false;
    }
  }
  return true;
}
function segmentRectDistanceSquared(a, b, hx, hz) {
  if (segmentIntersectsRect(a, b, hx, hz)) return 0;
  let minimum = Math.min(pointRectDistanceSquared(a, hx, hz), pointRectDistanceSquared(b, hx, hz));
  for (const corner of [[-hx,-hz],[-hx,hz],[hx,-hz],[hx,hz]]) minimum = Math.min(minimum, pointSegmentDistanceSquared(corner, a, b));
  return minimum;
}
function carCorners(state, config) {
  const c = Math.cos(state.yaw), s = Math.sin(state.yaw), hx = config.vehicleLength/2, hz = config.vehicleWidth/2;
  return [[-hx,-hz],[-hx,hz],[hx,-hz],[hx,hz]].map(([x,z]) => [state.x+x*c+z*s, state.z-x*s+z*c]);
}
function carIntersectsRectangle(state, config, rectangle, padding = 0) {
  const dx = rectangle.x-state.x, dz = rectangle.z-state.z;
  const c = Math.cos(state.yaw), s = Math.sin(state.yaw), ac = Math.abs(c), as = Math.abs(s);
  const hx = config.vehicleLength/2, hz = config.vehicleWidth/2;
  const rx = rectangle.w/2+padding, rz = rectangle.d/2+padding, epsilon = 1e-12;
  // Separating-axis test on both car axes and both world-aligned foot axes.
  return Math.abs(dx*c-dz*s) <= hx+rx*ac+rz*as+epsilon
    && Math.abs(dx*s+dz*c) <= hz+rx*as+rz*ac+epsilon
    && Math.abs(dx) <= rx+hx*ac+hz*as+epsilon
    && Math.abs(dz) <= rz+hx*as+hz*ac+epsilon;
}
function collisionAt(state, world, config) {
  const hx = config.vehicleLength/2, hz = config.vehicleWidth/2;
  for (const segment of world.segments) {
    if (segmentRectDistanceSquared(localPoint(segment.a, state), localPoint(segment.b, state), hx, hz) <= segment.radius**2+1e-12) return segment;
  }
  for (const disk of world.disks) {
    if (disk.shape === 'rectangle' ? carIntersectsRectangle(state,config,disk) : pointRectDistanceSquared(localPoint([disk.x,disk.z], state), hx, hz) <= disk.radius**2+1e-12) return disk;
  }
  const b = world.bounds;
  if (carCorners(state, config).some(([x,z]) => x < b.minX || x > b.maxX || z < b.minZ || z > b.maxZ)) return { id: 'bounds', kind: 'bounds' };
  return null;
}
function surfaceAt(state, world, config) {
  let surface = { id: 'default', mu: .8 };
  for (const s of world.surfaces) if (Math.abs(state.x-s.x) <= s.w/2 && Math.abs(state.z-s.z) <= s.d/2) surface = s;
  return { id: surface.id, mu: surface.mu*config.frictionMultiplier };
}
function rayCircle(origin, direction, center, radius) {
  const x = origin[0]-center[0], z = origin[1]-center[1], q = x*direction[0]+z*direction[1], c = x*x+z*z-radius*radius;
  if (c <= 0) return 0;
  const discriminant = q*q-c;
  if (discriminant < 0) return Infinity;
  const t = -q-Math.sqrt(discriminant);
  return t >= 0 ? t : Infinity;
}
function rayRectangle(origin, direction, rectangle) {
  let low = 0, high = Infinity;
  for (const [offset, delta, half] of [[origin[0]-rectangle.x,direction[0],rectangle.w/2],[origin[1]-rectangle.z,direction[1],rectangle.d/2]]) {
    if (Math.abs(delta) < 1e-12) { if (Math.abs(offset) > half) return Infinity; }
    else {
      let t0 = (-half-offset)/delta, t1 = (half-offset)/delta;
      if (t0 > t1) [t0,t1] = [t1,t0];
      low = Math.max(low,t0); high = Math.min(high,t1);
      if (low > high) return Infinity;
    }
  }
  return low;
}
function rayCapsule(origin, direction, segment) {
  const { a, b, radius } = segment;
  if (pointSegmentDistanceSquared(origin, a, b) <= radius*radius) return 0;
  let closest = Math.min(rayCircle(origin, direction, a, radius), rayCircle(origin, direction, b, radius));
  const length = dist(a, b);
  if (length > 1e-12) {
    const tx = (b[0]-a[0])/length, tz = (b[1]-a[1])/length;
    const along = (origin[0]-a[0])*tx+(origin[1]-a[1])*tz;
    const across = -(origin[0]-a[0])*tz+(origin[1]-a[1])*tx;
    const alongD = direction[0]*tx+direction[1]*tz, acrossD = -direction[0]*tz+direction[1]*tx;
    if (Math.abs(acrossD) > 1e-12) for (const edge of [-radius,radius]) {
      const t = (edge-across)/acrossD, u = along+t*alongD;
      if (t >= 0 && u >= 0 && u <= length) closest = Math.min(closest, t);
    }
  }
  return closest;
}
function normalNoise(seed, frame, index) {
  let value = (seed ^ Math.imul(frame+1, 0x9e3779b1) ^ Math.imul(index+1, 0x85ebca6b)) >>> 0;
  const random = () => { value = (Math.imul(value, 1664525)+1013904223) >>> 0; return (value+.5)/4294967296; };
  return Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random());
}
function approach(value, target, maximumDelta) {
  return value+clamp(target-value, -maximumDelta, maximumDelta);
}

export function createSimulation(worldInput, configInput = {}) {
  const world = normalizedWorld(worldInput), config = normalizedConfig(configInput);
  const worldHash = hash(world), vehicleRadius = Math.hypot(config.vehicleLength, config.vehicleWidth)/2;
  const targets = world.route.length > 1 ? [...world.route.slice(1),world.route[0]] : [];
  const continuous = config.episodeMode === 'continuous-laps-v1';
  if (continuous && !targets.length) throw new Error('Continuous laps require at least two route gates');
  const routeLength = targets.reduce((length, p, i) => length+dist(i ? targets[i-1] : world.route[0], p), 0);
  const boundSegments = [
    [[world.bounds.minX,world.bounds.minZ],[world.bounds.maxX,world.bounds.minZ]],
    [[world.bounds.maxX,world.bounds.minZ],[world.bounds.maxX,world.bounds.maxZ]],
    [[world.bounds.maxX,world.bounds.maxZ],[world.bounds.minX,world.bounds.maxZ]],
    [[world.bounds.minX,world.bounds.maxZ],[world.bounds.minX,world.bounds.minZ]],
  ].map(([a,b]) => ({a,b,radius:0}));
  let state, metrics, scenario = 'default', actionLog, trace, priorContact = null, initialState = null;
  let lapStartTime = 0, lapStartDistance = 0, lapStartFrame = 0, lapStartCollisions = 0;

  function contactStillNear(id) {
    const hx = config.vehicleLength/2, hz = config.vehicleWidth/2, hysteresis = .025;
    const segment = world.segments.find(s => s.id === id);
    if (segment) return segmentRectDistanceSquared(localPoint(segment.a,state),localPoint(segment.b,state),hx,hz) <= (segment.radius+hysteresis)**2;
    const disk = world.disks.find(s => s.id === id);
    if (disk) return disk.shape === 'rectangle' ? carIntersectsRectangle(state,config,disk,hysteresis) : pointRectDistanceSquared(localPoint([disk.x,disk.z],state),hx,hz) <= (disk.radius+hysteresis)**2;
    if (id === 'bounds') {
      const b = world.bounds;
      return carCorners(state,config).some(([x,z]) => x-b.minX < hysteresis || b.maxX-x < hysteresis || z-b.minZ < hysteresis || b.maxZ-z < hysteresis);
    }
    return false;
  }

  function snapshot() { return copy({ state, metrics, config }); }
  function reset(options = {}) {
    options = options && typeof options === 'object' ? options : {};
    // Validate before changing a running/previous episode; never clamp a
    // requested recovery state into a different, apparently valid condition.
    const restored = Object.hasOwn(options,'initialState') && options.initialState !== undefined
      ? validateInitialState(options.initialState,world,config) : null;
    if (continuous && restored) throw new Error('Continuous laps cannot restore a dynamic recovery state');
    if (finite(options.seed)) config.seed = Math.trunc(options.seed) >>> 0;
    if (typeof options.scenario === 'string') scenario = options.scenario;
    initialState = restored;
    const pose = restored ? {x:restored.x,z:restored.z,yaw:restored.yaw} : world.spawn;
    state = { ...pose, speed: restored?.speed ?? 0, steering: restored?.steering ?? 0, sim_time: 0, frame: 0, paused: false, terminal: false, success: false, reason: null, surface: surfaceAt(pose, world, config).id };
    metrics = { elapsed: 0, distance: 0, collisions: 0, collision_events: [], route_index: restored?.routeIndex ?? 0, route_progress: 0, passed_gates: 0, required_crossing_events: [], finished: false, max_lateral_acceleration: 0, traction_limited_steps: 0 };
    if (continuous) Object.assign(metrics,{completed_laps:0,lap_events:[],current_lap_passed_gates:0,current_lap_required_crossing_events:[]});
    lapStartTime = 0; lapStartDistance = 0; lapStartFrame = 0; lapStartCollisions = 0;
    actionLog = []; trace = []; priorContact = null;
    const hit = collisionAt(state, world, config);
    if (hit) { state.terminal = true; state.reason = 'invalid_spawn'; }
    return snapshot();
  }
  function advanceRequiredCrossings(previous,next,time,interval) {
    const ledger = continuous ? metrics.current_lap_required_crossing_events : metrics.required_crossing_events;
    for (const g of world.requiredCrossings) {
      if (ledger.some(event => event.id === g.id)) continue;
      const a = (previous.z-g.z)*g.direction, b = (next.z-g.z)*g.direction;
      if (!(a < 0 && b >= 0)) continue;
      const fraction = -a/(b-a), x = previous.x+(next.x-previous.x)*fraction;
      const yaw = previous.yaw+wrap(next.yaw-previous.yaw)*fraction;
      const halfX = config.vehicleLength/2*Math.abs(Math.cos(yaw))+config.vehicleWidth/2*Math.abs(Math.sin(yaw));
      // Require the whole planar vehicle width to pass through the sheet's real
      // opening, rather than accepting a nearby wide route gate or a tangential visit.
      if (x-halfX < g.minX-1e-12 || x+halfX > g.maxX+1e-12) continue;
      if (g.surface_id && !world.surfaces.some(s => s.id === g.surface_id && Math.abs(x-s.x) <= s.w/2 && Math.abs(g.z-s.z) <= s.d/2)) continue;
      const event = {id:g.id,surface_id:g.surface_id,time:time-(1-fraction)*interval,frame:state.frame,x,z:g.z,yaw,direction:g.direction};
      if (continuous) { event.lap_index = metrics.completed_laps+1; ledger.push(event); }
      metrics.required_crossing_events.push(event);
    }
  }
  function advanceGates(previous, next, time, interval) {
    const i = metrics.route_index;
    if (!targets.length || i >= targets.length) return;
    const target = targets[i], before = i ? targets[i-1] : world.route[0];
    const length = dist(before,target), nx = (target[0]-before[0])/length, nz = (target[1]-before[1])/length;
    const signed = p => (p.x-target[0])*nx+(p.z-target[1])*nz;
    const a = signed(previous), b = signed(next);
    if (!(a < 0 && b >= 0)) return;
    const fraction = -a/(b-a), x = previous.x+(next.x-previous.x)*fraction, z = previous.z+(next.z-previous.z)*fraction;
    const lateral = Math.abs(-(x-target[0])*nz+(z-target[1])*nx);
    if (lateral > config.gateHalfWidth) return;
    metrics.passed_gates++; metrics.route_index = i+1;
    if (continuous) {
      metrics.current_lap_passed_gates++;
      metrics.route_progress = metrics.current_lap_passed_gates/targets.length;
      if (metrics.current_lap_passed_gates !== targets.length) return;
      const endTime = time-(1-fraction)*interval;
      const endDistance = metrics.distance-(1-fraction)*Math.hypot(next.x-previous.x,next.z-previous.z);
      const crossedAll = world.requiredCrossings.every(g => metrics.current_lap_required_crossing_events.some(event => event.id === g.id));
      const sufficientDistance = endDistance-lapStartDistance >= Math.max(config.vehicleLength,routeLength*.5);
      if (!crossedAll || !sufficientDistance) {
        state.terminal = true; state.reason = crossedAll ? 'lap_distance_insufficient' : 'required_crossing_missing'; state.speed = 0;
        return;
      }
      metrics.completed_laps++;
      metrics.lap_events.push({lap_index:metrics.completed_laps,start_time:lapStartTime,end_time:endTime,elapsed:endTime-lapStartTime,
        start_frame:lapStartFrame,end_frame:state.frame,distance:endDistance-lapStartDistance,passed_gates:targets.length,
        collisions:metrics.collisions-lapStartCollisions,required_crossing_events:copy(metrics.current_lap_required_crossing_events),
        finish:{x,z,yaw:wrap(previous.yaw+wrap(next.yaw-previous.yaw)*fraction),
          speed:previous.speed+(next.speed-previous.speed)*fraction,steering:previous.steering+(next.steering-previous.steering)*fraction}});
      lapStartTime = endTime; lapStartDistance = endDistance; lapStartFrame = state.frame; lapStartCollisions = metrics.collisions;
      // Only scoring cursors are renewed. The vehicle, time, observations,
      // contact hysteresis and complete control/trace history keep evolving.
      metrics.route_index = 0; metrics.route_progress = 0; metrics.current_lap_passed_gates = 0;
      metrics.current_lap_required_crossing_events = [];
      return;
    }
    metrics.route_progress = metrics.passed_gates/targets.length;
    if (initialState && metrics.route_index === targets.length) {
      state.terminal = true; state.reason = 'recovery_segment_complete'; state.speed = 0;
      // Existing gates and required crossings are never credited to a recovery.
      // Even starting with routeIndex=0 does not certify a normal complete lap.
      return;
    }
    if (metrics.passed_gates === targets.length && metrics.distance >= Math.max(config.vehicleLength, routeLength*.5)) {
      const crossedAll = world.requiredCrossings.every(g => metrics.required_crossing_events.some(event => event.id === g.id));
      state.success = crossedAll; state.terminal = true; state.reason = crossedAll ? 'route_complete' : 'required_crossing_missing'; state.speed = 0; metrics.finished = crossedAll;
    }
  }
  function step(action = {}) {
    if (state.terminal || state.paused) return snapshot();
    const steering = number(action?.steering, 0, -config.maxSteer, config.maxSteer);
    const speed = number(action?.speed, 0, -config.maxSpeed, config.maxSpeed);
    actionLog.push({ frame: state.frame, steering, speed });
    let frameContact = null;
    const remainingTime = Math.min(config.dt, config.timeLimit-state.sim_time);
    let elapsed = 0;
    while (elapsed < remainingTime-1e-12 && !state.terminal) {
      // Smaller motion intervals preserve thin barriers even with large commanded speeds.
      const maximumMotionRate = config.maxSpeed+vehicleRadius*config.maxSpeed/config.wheelbase*Math.tan(config.maxSteer);
      const h = Math.min(config.substep, .012/maximumMotionRate, remainingTime-elapsed);
      const previous = { ...state };
      const surface = surfaceAt(state, world, config);
      // Only forward acceleration is estimated from the supplied drive data.
      // Braking/reverse retain a constant provisional response; independent
      // longitudinal/lateral limits are not a coupled tyre friction circle.
      let nextSpeed;
      if (config.longitudinalModel === 'real-drive-response-estimate-v1') {
        const rate = state.speed >= 0 && speed > state.speed
          ? Math.min(config.driveAccelerationCap,surface.mu*9.80665,(speed-state.speed)/config.driveTau)
          : config.brakeDeceleration;
        nextSpeed = approach(state.speed,speed,rate*h);
      } else nextSpeed = approach(state.speed, speed, config.acceleration*h);
      const nextSteering = approach(state.steering, steering, config.steerRate*h);
      const rawRate = nextSpeed/config.wheelbase*Math.tan(nextSteering);
      // 確認 r3 の反事実(摩擦円・縦を優先): 縦の加減速 |a_long| が使った分だけ、横に使える上限を √((μg)² − a_long²) に下げる
      const aLongFc = (nextSpeed-state.speed)/h, latCapFc = Math.sqrt(Math.max(0,(surface.mu*9.80665)**2-aLongFc*aLongFc));
      const limit = latCapFc/Math.max(Math.abs(nextSpeed), .01);
      const yawRate = clamp(rawRate, -limit, limit);
      if (Math.abs(rawRate) > limit) metrics.traction_limited_steps++;
      metrics.max_lateral_acceleration = Math.max(metrics.max_lateral_acceleration, Math.abs(nextSpeed*yawRate));
      const yawMid = state.yaw+yawRate*h/2;
      const trial = { ...state, x: state.x+nextSpeed*Math.cos(yawMid)*h, z: state.z-nextSpeed*Math.sin(yawMid)*h, yaw: wrap(state.yaw+yawRate*h), speed: nextSpeed, steering: nextSteering, surface: surface.id };
      const hit = collisionAt(trial, world, config);
      if (hit) {
        frameContact = hit.id;
        if (priorContact !== hit.id) {
          metrics.collisions++; metrics.collision_events.push({ time: state.sim_time+elapsed+h, frame: state.frame, collider: hit.id, kind: hit.kind, x: trial.x, z: trial.z });
          priorContact = hit.id;
        }
        state.speed = 0; state.steering = nextSteering;
        if (config.stopOnCollision) { state.terminal = true; state.reason = 'collision'; }
        // Preserve the last non-penetrating pose and consume only the current motion interval.
      } else {
        state = trial; metrics.distance += Math.hypot(state.x-previous.x,state.z-previous.z);
        advanceRequiredCrossings(previous,state,state.sim_time+elapsed+h,h);
        advanceGates(previous, state, state.sim_time+elapsed+h, h);
      }
      elapsed += h;
      if (hit && !config.stopOnCollision) { elapsed = remainingTime; break; }
    }
    if (!frameContact && priorContact && !contactStillNear(priorContact)) priorContact = null;
    state.sim_time = Math.min(config.timeLimit, state.sim_time+elapsed);
    state.frame++; metrics.elapsed = state.sim_time;
    if (!state.terminal && state.sim_time >= config.timeLimit-1e-10) {
      state.terminal = true;
      if (continuous) { state.sim_time = config.timeLimit; metrics.elapsed = state.sim_time; state.reason = 'collection_duration_complete'; }
      else { state.reason = 'time_limit'; state.speed = 0; }
    }
    trace.push({ frame: state.frame, time: state.sim_time, x: state.x, z: state.z, yaw: state.yaw, speed: state.speed, steering: state.steering, surface: state.surface });
    return snapshot();
  }
  function observe() {
    const ranges = [];
    const origin = [state.x,state.z];
    for (let i = 0; i < config.rayCount; i++) {
      const angle = -Math.PI/2+i*Math.PI/(config.rayCount-1), absolute = state.yaw+angle;
      const direction = [Math.cos(absolute),-Math.sin(absolute)];
      let range = config.rayRange;
      for (const segment of world.segments) if (segment.height >= config.sensorHeight) range = Math.min(range,rayCapsule(origin,direction,segment));
      for (const segment of boundSegments) range = Math.min(range,rayCapsule(origin,direction,segment));
      for (const disk of world.disks) if (disk.height >= config.sensorHeight) range = Math.min(range,disk.shape === 'rectangle' ? rayRectangle(origin,direction,disk) : rayCircle(origin,direction,[disk.x,disk.z],disk.radius));
      const hit = range < config.rayRange;
      ranges.push({ angle, distance: clamp(range+normalNoise(config.seed,state.frame,i)*config.noiseStd,0,config.rayRange), hit });
    }
    return { speed: state.speed, steering: state.steering, ranges, sim_time: state.sim_time, frame: state.frame };
  }
  function demoAction() {
    if (state.terminal || !targets.length) return { steering: 0, speed: 0 };
    const i = Math.min(metrics.route_index,targets.length-1), gate = targets[i];
    const previous = i ? targets[i-1] : world.route[0];
    const length = dist(previous,gate);
    const projection = clamp(((state.x-previous[0])*(gate[0]-previous[0])+(state.z-previous[1])*(gate[1]-previous[1]))/(length*length),0,1);
    let goal = [previous[0]+(gate[0]-previous[0])*projection,previous[1]+(gate[1]-previous[1])*projection];
    let lookahead = .65;
    for (let j = i; j < targets.length; j++) {
      const next = targets[j], remaining = dist(goal,next);
      if (remaining >= lookahead) { goal = [goal[0]+(next[0]-goal[0])*lookahead/remaining,goal[1]+(next[1]-goal[1])*lookahead/remaining]; break; }
      lookahead -= remaining; goal = next;
    }
    // Ground-truth pose and route are deliberately confined to this demonstration controller.
    const dx = goal[0]-state.x, dz = goal[1]-state.z;
    const angle = wrap(Math.atan2(-dz,dx)-state.yaw), distance = Math.hypot(dx,dz);
    const steering = clamp(Math.atan2(2*config.wheelbase*Math.sin(angle),Math.max(distance,.3)),-config.maxSteer,config.maxSteer);
    const speed = Math.min(config.maxSpeed,.7)*clamp(1-Math.abs(steering)/config.maxSteer*.45,.3,1);
    return { steering, speed };
  }
  function report() {
    return copy({ schema: 'course-simulator-report/v1', world_hash: worldHash, scenario, seed: config.seed, config, ...(continuous ? {episode_kind:'continuous-lap-collection'} : {}), ...(initialState ? {episode_kind:'dynamic-recovery-teacher',initial_state:initialState} : {}), outcome: { terminal: state.terminal, success: state.success, reason: state.reason }, metrics, final_state: state, actions: actionLog, trace, model: { type: 'planar-kinematic-bicycle', traction: 'yaw_rate_limited_by_mu_g_over_abs_speed', surface_sampling: 'vehicle_center_last_matching_region', collision: 'capsule_disk_and_rectangular_foot_against_oriented_rectangle_with_motion_substeps', sensor_noise: 'range_only_metres_frame_seeded_gaussian', step: config.dt, substep_max: config.substep, provisional_vehicle_dimensions: true } });
  }
  function setPaused(paused) { state.paused = !!paused; return snapshot(); }
  reset();
  return { reset, step, observe, snapshot, report, demoAction, setPaused };
}
