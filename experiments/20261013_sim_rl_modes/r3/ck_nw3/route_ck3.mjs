// コースの経路(33 ゲートの輪)を書き出す(線の比べに使う。自前の射影は analyze_ck3.py)
import {BASE_WORLD} from './own_ck3.mjs';
console.log(JSON.stringify({route: BASE_WORLD.route}));
