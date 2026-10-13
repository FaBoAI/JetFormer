// JetFormer の汎用描画(アップロードしたコースの physics-world.json をそのまま形にする)。
// /sim/c/<コースID>/scene.js として配信する。アップロードされた JS は実行しないので、コースはデータ(_jf/course-data.js)だけで描く。
// - 衝突物(segments・disks)はすべて、その高さのメッシュとして描く(ToF はシーンのメッシュから計算されるため)。
// - TextureLoader で 1 枚以上(床のテクスチャ)を必ず読む(app.js は DefaultLoadingManager.onLoad で「評価を開始」を有効にするため)。
// - 見た目は本来の scene.js(部屋・LED 矢印・テープ・ラベルなど)とは違う。カメラ画像を使う AI の評価は Default と直接は比べられない。
import * as THREE from 'three';
import DATA from './_jf/course-data.js';

const SURFACE_TEXTURES = {
  carpet: ['carpet.jpg', 'carpet-normal.jpg', '#bcc1c3', 0.78],
  turf: ['turf.jpg', 'turf-normal.jpg', '#dee1c4', 0.68],
  'blue-mat': ['blue-pad.jpg', null, '#69d9fa', 0.42],
  shortcut: ['black-mat.jpg', null, '#3b3d3b', 1.0],
};
const SURFACE_COLORS = { ptfe: '#f4f4f0' };

export function buildCourse(scene) {
  const course = new THREE.Group(); course.name = 'jetformer-generic-course'; scene.add(course);
  const visual = DATA.visual || {}, colors = visual.colors || {}, surfaceTextures = visual.surface_textures || {};
  const physics = structuredClone(DATA.physics);
  const mat = (color, opts = {}) => new THREE.MeshStandardMaterial({ color, roughness: .72, ...opts });
  const loader = new THREE.TextureLoader();
  function photo(name, material, sizeX, sizeZ, tile, normal = null) {
    const rx = Math.max(1, sizeX / tile), ry = Math.max(1, sizeZ / tile);
    loader.load('./assets/' + name, t => {
      t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rx, ry); t.anisotropy = 8;
      material.map = t; material.needsUpdate = true;
    });
    if (normal) loader.load('./assets/' + normal, t => {
      t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rx, ry); t.anisotropy = 8;
      material.normalMap = t; material.normalScale.set(.18, .18); material.needsUpdate = true;
    });
  }
  function box(w, h, d, material, x, y, z, yaw = 0) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
    m.position.set(x, y, z); m.rotation.y = yaw; m.castShadow = m.receiveShadow = true; course.add(m); return m;
  }
  function cylinder(r, h, material, x, z) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 16), material);
    m.position.set(x, h / 2, z); m.castShadow = m.receiveShadow = true; course.add(m); return m;
  }
  function plane(w, d, material, x, y, z) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), material);
    m.rotation.x = -Math.PI / 2; m.position.set(x, y, z); m.receiveShadow = true; course.add(m); return m;
  }

  // 床: 範囲(bounds)より 0.5 m 広く。床のテクスチャは必ず 1 枚読む(assetsReady のため)
  const b = physics.bounds, fw = b.maxX - b.minX + 1, fd = b.maxZ - b.minZ + 1;
  const floorMat = mat(colors.floor || '#bcc1c3', { roughness: 1 });
  const floorName = visual.floor_texture || 'carpet.jpg';
  photo(floorName, floorMat, fw, fd, .78, floorName === 'carpet.jpg' ? 'carpet-normal.jpg' : null);
  plane(fw, fd, floorMat, (b.minX + b.maxX) / 2, 0, (b.minZ + b.maxZ) / 2);

  // 路面: id ごとのテクスチャ。重なりがちらつかないよう 0.5 mm ずつ上げる
  physics.surfaces.forEach((s, i) => {
    const preset = SURFACE_TEXTURES[s.id];
    const custom = surfaceTextures[s.id];
    const material = mat(custom ? '#ffffff' : preset ? preset[2] : (SURFACE_COLORS[s.id] || '#9ca3a6'), { roughness: .95 });
    if (custom) photo(custom, material, s.w, s.d, 1.0);
    else if (preset) photo(preset[0], material, s.w, s.d, preset[3], preset[1]);
    plane(s.w, s.d, material, s.x, .001 + .0005 * i, s.z);
  });

  // 衝突物(physics の高さのまま)
  const railA = mat(colors.rail || '#e8e4db', { roughness: .54 }), railB = mat(colors.rail_alt || '#b81627', { roughness: .42 });
  const curtain = mat(colors.curtain || '#11171c', { roughness: 1 });
  const post = mat(colors.post || '#1b2023'), gate = mat(colors.gate || '#bcc2c2', { metalness: .6, roughness: .35 });
  physics.segments.forEach((s, i) => {
    const dx = s.b[0] - s.a[0], dz = s.b[1] - s.a[1], length = Math.max(Math.hypot(dx, dz), .001), yaw = -Math.atan2(dz, dx);
    const cx = (s.a[0] + s.b[0]) / 2, cz = (s.a[1] + s.b[1]) / 2;
    if (s.kind === 'curtain') box(length, s.height, Math.max(2 * s.radius, .03), curtain, cx, s.height / 2, cz, yaw);
    else if (s.kind === 'gate-foot') box(length, s.height, Math.max(2 * s.radius, .02), post, cx, s.height / 2, cz, yaw);
    else box(length, s.height, Math.max(2 * s.radius, .019), i % 2 ? railB : railA, cx, s.height / 2, cz, yaw);
  });
  for (const d of physics.disks) {
    if (d.kind === 'post-foot') box(.16, d.height, .15, post, d.x, d.height / 2, d.z);
    else if (d.kind === 'gate-post') cylinder(d.radius, d.height, gate, d.x, d.z);
    else cylinder(d.radius, d.height, post, d.x, d.z);
  }

  return { course, room: null, curtains: new THREE.Group(), animated: [], landmarks: [], dimensions: [], physics };
}
