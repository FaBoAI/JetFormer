// JetFormer の汎用描画のコース(アップロードしたコースのゲート列・必須横断・開始位置)。
// /sim/c/<コースID>/world.js として配信する。データは同じフォルダーの _jf/course-data.js(サーバーが生成)から読む。
// app.js が使うのは makeWorld と WORLD_VERSION だけ。
import DATA from './_jf/course-data.js';

export const WORLD_VERSION = DATA.world_version;
export const ROUTE = DATA.world.route.map(p => [p[0], p[1]]);
export const REQUIRED_CROSSINGS = DATA.world.requiredCrossings.map(c => ({...c}));
export function makeWorld(physics) {
  return {...physics,
    route: DATA.world.route.map(p => [p[0], p[1]]),
    requiredCrossings: DATA.world.requiredCrossings.map(c => ({...c})),
    spawn: {...DATA.world.spawn}};
}
