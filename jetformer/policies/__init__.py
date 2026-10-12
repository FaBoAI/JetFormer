"""方策(モデルの構造)の登録簿。drive.py・web/server.py・training.py も読むので torch を import しない。
構築(build)のときだけ、方策のモジュール(jetformer.policies.<名前>)を遅延 import する。

    from jetformer import policies
    spec = policies.get('act')
    cfg = spec.resolve({'dec_layers': 3})            # 既定値 + 指定(全部のキー)→ model_config.json の policy_cfg
    model = policies.build_policy('act', num_modes=6, cfg=cfg)      # torch(方策のモジュールの build)

約束(どの方策も同じ):
    入力  image [1,3,224,224]・tof [1,2,42,54]・imu [1,10,6]・state [1,3]・mode [1,num_modes](float32・バッチ1・固定の形)
    出力  actions [1,10,2]((µs − 1500)/500、33 ms 刻みの 1〜10 フレーム先の [ステア, スロットル])。追加の出力は名前を変えて足してよい
走行・評価は基本の5つを全部渡す(エンジンが使わない入力は書き出しのときにグラフから消え、読まれないだけ)。

古い学習結果(model_config.json に 'policy' が無い)は jetformer_v0 と v0 の凍結した既定値(dim 256・layers 4・heads 8)で読む。
"""

import copy
import importlib
import importlib.util
import json
import numbers
from pathlib import Path

CONTRACT = 1                       # 入出力の約束の版(model_config.json の 'contract')
DEFAULT = 'jetformer_v0'           # 'policy' が無い学習結果 = 今までのもの
FUTURE_STEPS = 10
BASE_INPUTS = ('image', 'tof', 'imu', 'state', 'mode')
BASE_SHAPES = {'image': [1, 3, 224, 224], 'tof': [1, 2, 42, 54], 'imu': [1, 10, 6], 'state': [1, 3]}   # mode は [1, num_modes]
ROOT = Path(__file__).resolve().parents[2]          # リポジトリの一番上(相対パスの基準)
PWM_CENTER, PWM_SCALE = 1500.0, 500.0               # model.py と同じ(torch を読まずに使うため複製)
V0_FROZEN_CFG = {'dim': 256, 'layers': 4, 'heads': 8}   # 'policy' の無い古い学習結果の構造(変えない)
DEFAULT_GENRE_KEYS = {'embed': 'mode_embed.weight', 'heads': {'heads': FUTURE_STEPS * 2}}
DEFAULT_HINTS = {'train_rate': 200, 'val_rate': 650, 'min_free_mb': 3500, 'amp': True, 'engine_precision': 'fp16',
                 'infer_ms': 2.3, 'params_m': 14.1, 'engine_s': 70}
ENGINE_PRECISIONS = ('fp16', 'bf16', 'fp32', 'fp16_norm_fp32')
# fp16_norm_fp32 = FP16 + 正規化(LayerNorm)・Softmax だけ FP32(tools/build_engine.py。dinov2 の FP16 の精度の確認のため、2026-10-12)
# 方策の担当者が中身を書くまでの仮のモジュールの印(ready() がファイルの文字だけで見分ける。実行はしない)
STUB_MARK = 'POLICY_STUB = True'


class PolicySpec(object):
    """登録簿の1項目。defaults は「凍結する既定値」(学習結果の policy_cfg が無いキーはこの値で読む)。

    options: 管理画面で変えられる設定 [{'key', 'label', 'type': 'int'|'float'|'bool'|'choice', 'min', 'max', 'choices',
             'help'}]。CLI(--policy-cfg)は defaults の全部のキーを変えられる
    hints: 学習の速さ(枚/秒)・空きメモリの目安・AMP・エンジンの精度・推論の目安(ms)・パラメータ数(M)・
           エンジンの作成の目安(秒)。値は bench/ の実測(spec.md §3.0)
    genre_keys: ジャンルの層 {'embed': 'mode_embed.weight', 'heads': {層の名前: 1ジャンルの行の数}}(--init-mode-copy 用。
                None = ジャンルを広げられない)
    requires: 学習に要るファイル(リポジトリからの相対パス。存在だけ確かめる)
    train_only_keys: 構造を変えない学習だけの設定(--init の一致の確認で比べない)"""

    def __init__(self, name, title, module, slug, description, status='experimental', defaults=None, options=(),
                 hints=None, genre_keys=None, outputs=('actions',), postprocess=None, requires=(),
                 requires_pretrained=False, train_only_keys=()):
        self.name, self.title, self.module, self.slug = name, title, module, slug
        self.description, self.status = description, status
        self.defaults = dict(defaults or {})
        self.options = [dict(o) for o in options]
        self.hints = dict(DEFAULT_HINTS, **(hints or {}))
        self.genre_keys = copy.deepcopy(genre_keys)
        self.outputs = tuple(outputs)
        self.postprocess = postprocess
        self.requires = tuple(requires)
        self.requires_pretrained = bool(requires_pretrained)
        self.train_only_keys = tuple(train_only_keys)
        unknown = [o['key'] for o in self.options if o['key'] not in self.defaults]
        unknown += [k for k in self.train_only_keys if k not in self.defaults]
        if unknown:
            raise ValueError('%s: defaults に無い設定 %s' % (name, unknown))
        if self.outputs[:1] != ('actions',):
            raise ValueError('%s: 出力の先頭は actions' % name)
        if self.hints['engine_precision'] not in ENGINE_PRECISIONS:
            raise ValueError('%s: engine_precision は %s のどれか' % (name, ENGINE_PRECISIONS))

    # ---------------------------------------------------------------- 設定
    def option(self, key):
        for o in self.options:
            if o['key'] == key:
                return o
        return None

    def resolve(self, cfg=None, ui=False):
        """既定値 + 指定(dict)→ 全部のキーを持つ dict。

        知らないキー・型に直せない値・options の範囲外 → ValueError。ui=True(管理画面から)は options に無いキーを
        既定値から変えると ValueError(CLI の --policy-cfg はすべてのキーを変えられる)"""
        if cfg is None:
            cfg = {}
        if isinstance(cfg, str):
            cfg = json.loads(cfg) if cfg.strip() else {}
        if not isinstance(cfg, dict):
            raise ValueError('%s の設定は {キー: 値} の形で指定してください' % self.title)
        out = copy.deepcopy(self.defaults)
        for key, value in cfg.items():
            if key not in self.defaults:
                raise ValueError('知らない設定です: %s(使えるもの: %s)' % (key, ', '.join(sorted(self.defaults)) or 'なし'))
            v = _convert(key, value, self.defaults[key])
            opt = self.option(key)
            if opt is None:
                if ui and v != self.defaults[key]:
                    raise ValueError('管理画面からは変えられない設定です: %s(CLI の --policy-cfg で指定してください)' % key)
            else:
                if opt.get('choices') is not None and v not in [_convert(key, c, self.defaults[key]) for c in opt['choices']]:
                    raise ValueError('%s は %s のどれか(指定 %s)' % (key, opt['choices'], value))
                if opt.get('min') is not None and v < opt['min']:
                    raise ValueError('%s は %s 以上(指定 %s)' % (key, opt['min'], value))
                if opt.get('max') is not None and v > opt['max']:
                    raise ValueError('%s は %s 以下(指定 %s)' % (key, opt['max'], value))
            out[key] = v
        return out

    def changed(self, resolved):
        """resolve の結果のうち、既定値から変えたキーだけ(コマンドの --policy-cfg 用)"""
        return {k: v for k, v in resolved.items() if self.defaults.get(k) != v}

    # ---------------------------------------------------------------- 構築(torch)
    def module_obj(self):
        return importlib.import_module(self.module)

    def build(self, num_modes, cfg=None, pretrained=False):
        """方策のモジュールの build(num_modes, pretrained, **resolved)。pretrained=False ではネット・大きな重みを読まない"""
        resolved = self.resolve(cfg)
        return self.module_obj().build(int(num_modes), pretrained=bool(pretrained), **resolved)

    # ---------------------------------------------------------------- 準備ができているか(torch なし・速い)
    def ready(self):
        """→ (True, '') / (False, 理由)。モジュールがあるか(実行しない)・仮のモジュールでないか・requires のファイルがあるか"""
        try:
            found = importlib.util.find_spec(self.module)
        except (ImportError, ValueError):
            found = None
        if found is None:
            return False, '%s のプログラム(%s)がありません' % (self.title, self.module)
        origin = getattr(found, 'origin', None)
        if origin and origin.endswith('.py'):
            try:
                with open(origin, encoding='utf-8') as f:
                    if STUB_MARK in f.read(4096):
                        return False, '%s はまだ実装されていません(%s)' % (self.title, self.module)
            except OSError:
                pass
        for rel in self.requires:
            if not (ROOT / rel).exists():
                return False, '%s に必要なファイル %s がありません(取得の手順は README)' % (self.title, rel)
        return True, ''

    def describe(self):
        """GET /api/policies 用"""
        ok, why = self.ready()
        return {'name': self.name, 'title': self.title, 'slug': self.slug, 'description': self.description,
                'status': self.status, 'defaults': copy.deepcopy(self.defaults), 'options': copy.deepcopy(self.options),
                'hints': dict(self.hints), 'outputs': list(self.outputs), 'postprocess': self.postprocess,
                'requires_pretrained': self.requires_pretrained, 'ready': ok, 'why_not': why}


def _convert(key, value, default):
    """値を既定値の型に直す(直せなければ ValueError)"""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ('true', 'false', '1', '0', 'yes', 'no'):
                return value.strip().lower() in ('true', '1', 'yes')
            raise ValueError
        if isinstance(default, int):
            if isinstance(value, bool):
                raise ValueError
            f = float(value)
            if f != int(f):
                raise ValueError
            return int(f)
        if isinstance(default, float):
            if isinstance(value, bool):
                raise ValueError
            f = float(value)
            if f != f or f in (float('inf'), float('-inf')):
                raise ValueError
            return f
        if isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError
            return value
        if isinstance(default, (list, tuple)):
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, (list, tuple)):
                raise ValueError
            if default:
                return [_convert(key, v, default[0]) for v in value]
            return list(value)
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError('設定 %s の値 %r を %s に直せません' % (key, value, type(default).__name__))


REGISTRY = {}


def register(spec):
    if spec.name in REGISTRY:
        raise ValueError('同じ名前の方策があります: %s' % spec.name)
    REGISTRY[spec.name] = spec
    return spec


def get(name):
    if name not in REGISTRY:
        raise KeyError('知らないモデルの種類です: %s(登録されているもの: %s)' % (name, ', '.join(REGISTRY)))
    return REGISTRY[name]


def names():
    return list(REGISTRY)


def list_specs():
    """[describe()] 登録の順"""
    return [s.describe() for s in REGISTRY.values()]


def title_of(name):
    """登録簿に無ければ名前のまま(走行・一覧は登録簿に無い方策も表示できる)"""
    s = REGISTRY.get(name or DEFAULT)
    return s.title if s else str(name)


def policy_of(cfg):
    """model_config(dict)の方策の名前。無ければ jetformer_v0"""
    return (cfg or {}).get('policy') or DEFAULT


def from_config(cfg):
    """model_config(dict)→ (spec, resolved_cfg)。'policy' 無し → jetformer_v0 と v0 の凍結した既定値"""
    cfg = cfg or {}
    name = policy_of(cfg)
    spec = get(name)
    pcfg = cfg.get('policy_cfg')
    if 'policy' not in cfg and pcfg is None:
        pcfg = dict(V0_FROZEN_CFG)
    return spec, spec.resolve(pcfg)


def inputs_of(cfg):
    """書き出した ONNX から読み直した入力の名前(古い学習結果は基本の5つ)"""
    return list((cfg or {}).get('inputs') or BASE_INPUTS)


def outputs_of(cfg):
    return list((cfg or {}).get('outputs') or ['actions'])


def _norm_shape(shape):
    """形 → int の list。固定でない(-1・None・小数・空)なら None"""
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    out = []
    for d in shape:
        if isinstance(d, bool) or not isinstance(d, numbers.Integral) or int(d) <= 0:
            return None
        out.append(int(d))
    return out


def check_io(inputs, outputs, num_modes):
    """エンジン(または ONNX)の入出力が走行で使えるか。inputs/outputs = {名前: 形(list)}。→ (True, '') / (False, 理由)。

    入力の名前がすべて BASE_INPUTS にあり、形が BASE_SHAPES(mode は [1,num_modes])と同じ。出力に actions [1,10,2] がある。
    すべての形が正の整数(-1・None なし)。num_modes が None なら mode の列の数は確かめない"""
    norm = {}
    for kind, io in (('入力', inputs), ('出力', outputs)):
        norm[kind] = {}
        for name, shape in io.items():
            s = _norm_shape(list(shape) if isinstance(shape, tuple) else shape)
            if s is None:
                return False, '%s %s の形 %s が固定ではありません(バッチ1・固定の形で書き出してください)' % (kind, name, shape)
            norm[kind][name] = s
    for name, shape in norm['入力'].items():
        if name not in BASE_INPUTS:
            return False, '入力 %s は走行で作れません(使える入力: %s)' % (name, ', '.join(BASE_INPUTS))
        if name == 'mode':
            if len(shape) != 2 or shape[0] != 1 or (num_modes is not None and shape[1] != int(num_modes)):
                return False, '入力 mode の形 %s が [1, %s] ではありません' % (shape, num_modes if num_modes is not None else 'ジャンル数')
        elif shape != BASE_SHAPES[name]:
            return False, '入力 %s の形 %s が走行で渡す形 %s と違います' % (name, shape, BASE_SHAPES[name])
    if 'actions' not in outputs:
        return False, '出力 actions がありません(出力: %s)' % (', '.join(outputs) or 'なし')
    if norm['出力']['actions'] != [1, FUTURE_STEPS, 2]:
        return False, '出力 actions の形 %s が [1, %d, 2] ではありません' % (norm['出力']['actions'], FUTURE_STEPS)
    return True, ''


def drivable(cfg):
    """model_config だけから、走行で使えるか → (bool, 理由)。登録簿は見ない(登録簿に無い方策も走れる)"""
    bad = [n for n in inputs_of(cfg) if n not in BASE_INPUTS]
    if bad:
        return False, '走行で作れない入力があります: %s' % ', '.join(bad)
    if 'actions' not in outputs_of(cfg):
        return False, '出力 actions がありません'
    try:
        newer = int((cfg or {}).get('contract', CONTRACT)) > CONTRACT
    except (TypeError, ValueError):
        newer = True
    if newer:
        return False, '新しい版の約束(contract %s)の学習結果です。プログラムを更新してください' % cfg.get('contract')
    return True, ''


def build_policy(name, num_modes, cfg=None, pretrained=False):
    return get(name).build(num_modes, cfg, pretrained)


# ==================================================================== 登録(値は spec.md §2.1.2・§3.0)

_V0_GENRES = {'embed': 'mode_embed.weight', 'heads': {'heads': FUTURE_STEPS * 2}}

register(PolicySpec(
    'jetformer_v0', 'JetFormer v0(今の標準)', 'jetformer.policies.jetformer_v0', 'v0',
    'ResNet-18 の画像 49・ToF 42・IMU・操作値・ジャンルのトークンを Transformer の Encoder 4層で融合し、'
    '出力トークンからジャンル別のヘッドで 10 フレーム先までの操作を出す(今までのモデル)。',
    status='stable', defaults=V0_FROZEN_CFG, options=(),
    hints={'train_rate': 200, 'val_rate': 650, 'min_free_mb': 3500, 'infer_ms': 2.3, 'params_m': 14.1, 'engine_s': 70},
    genre_keys=_V0_GENRES))

register(PolicySpec(
    'act', 'ACT 型(行動クエリの Decoder)', 'jetformer.policies.act', 'act',
    'v0 と同じトークンを Encoder 4層で融合し、10 個の行動クエリ(1 クエリ = 1 フレーム先)の Decoder で操作列を出す'
    '(Action Chunking Transformer 型)。CVAE は既定オフ。',
    defaults={'dim': 256, 'enc_layers': 4, 'dec_layers': 2, 'heads': 8, 'ffn': 512, 'dropout': 0.1, 'cvae': False,
              'latent_dim': 32, 'kl_weight': 10.0, 'cvae_layers': 2},
    options=({'key': 'dec_layers', 'label': 'Decoder の層の数', 'type': 'int', 'min': 1, 'max': 4},
             {'key': 'cvae', 'label': 'CVAE(学習のときだけの潜在変数。推論は z = 0)', 'type': 'bool'},
             {'key': 'kl_weight', 'label': 'CVAE の KL の重み', 'type': 'choice', 'choices': [1.0, 10.0]}),
    hints={'train_rate': 190, 'val_rate': 620, 'min_free_mb': 3500, 'infer_ms': 2.5, 'params_m': 15.6, 'engine_s': 96},
    genre_keys={'embed': 'mode_embed.weight', 'heads': {'heads': 2}}, train_only_keys=('kl_weight',)))

register(PolicySpec(
    'tcp', 'TCP / TransFuser++ 型(通過点 + 目標速度)', 'jetformer.policies.tcp', 'tcp',
    'v0 と同じトークンの Encoder の後、Decoder のクエリで通過点(0.1 秒刻みで 1 秒先まで)・目標速度・操作の 3 つを出す。'
    '走行の既定は操作の枝そのまま。走行タブで目標速度 → 制御器(速度の上限つき)に切り替えられる。',
    defaults={'dim': 256, 'enc_layers': 4, 'dec_layers': 2, 'heads': 8, 'ffn': 512, 'dropout': 0.1, 'wp_steps': 10,
              'wp_dt': 0.1, 'speed_bins': 10, 'speed_max': 4.5, 'w_wp': 0.5, 'w_speed': 0.1, 'cumsum': 'op',
              'labels': 'tcp_labels_v1'},
    options=(),
    hints={'train_rate': 185, 'val_rate': 600, 'min_free_mb': 3500, 'infer_ms': 2.7, 'params_m': 15.6, 'engine_s': 95},
    genre_keys={'embed': 'mode_embed.weight', 'heads': {'heads': 2, 'heads_wp': 2, 'heads_v': 10}},
    outputs=('actions', 'waypoints', 'speed_logits'), postprocess='tcp_controller', train_only_keys=('w_wp', 'w_speed')))

register(PolicySpec(
    'msfusion', '多段融合(TransFuser 型)', 'jetformer.policies.msfusion', 'msf',
    'ResNet-18 の 4 段と ToF の 4 段の CNN を、各段の小さな Transformer で融合してから(本家 TransFuser に近い形)、'
    '最終段のトークンを v0 型で融合する。',
    defaults={'dim': 256, 'scales': [1, 2, 3, 4], 'ex_layers': [2, 2, 2, 1], 'ex_heads': 4, 'ex_ffn': [4, 4, 4, 2],
              'late_layers': 2, 'late_heads': 8, 'head': 'tokens', 'dropout': 0.1},
    # [1, 2, 3] = 4段目の交換なし(design_research §5.3: エンジンが 4 ms を超えたときの代わり)
    options=({'key': 'scales', 'label': '融合する段', 'type': 'choice',
              'choices': [[1, 2, 3, 4], [2, 3, 4], [3, 4], [1, 2, 3]]},
             {'key': 'head', 'label': '出力の作り方', 'type': 'choice', 'choices': ['tokens', 'pooled']}),
    hints={'train_rate': 135, 'val_rate': 430, 'min_free_mb': 3500, 'infer_ms': 3.1, 'params_m': 18.9, 'engine_s': 210},
    genre_keys=_V0_GENRES))

register(PolicySpec(
    'dinov2', 'DINOv2 の画像特徴(ViT-S/14 凍結)', 'jetformer.policies.dinov2', 'dino',
    '自己教師あり学習の DINOv2 ViT-S/14(凍結)の画像トークン 64 + CLS と、v0 の ToF・IMU・操作値・ジャンルのトークンを'
    ' v0 型の Encoder 4層で融合する。重み data/pretrained/dinov2_vits14_pretrain.pth が要る(自動では取らない)。',
    defaults={'variant': 'vits14', 'weights': 'data/pretrained/dinov2_vits14_pretrain.pth', 'pool': 2, 'use_cls': True,
              'finetune_last': 0, 'lr_mult': 0.1, 'dim': 256, 'layers': 4, 'heads': 8, 'vit_depth': 12},
    options=({'key': 'pool', 'label': '画像トークンの平均(2 = 64 トークン、1 = 256 トークン)', 'type': 'choice', 'choices': [1, 2]},
             {'key': 'finetune_last', 'label': 'ViT の最後の何ブロックを学習するか(0 = 全部凍結)', 'type': 'choice',
              'choices': [0, 2]}),
    hints={'train_rate': 230, 'val_rate': 300, 'min_free_mb': 3500, 'infer_ms': 5.1, 'params_m': 24.5,
           'params_trainable_m': 2.9, 'engine_s': 105,
           # FP16 だけでは学習した重みで精度の確認(2.5 µs)に落ちることがあった(種 1 2.85 µs)。正規化・Softmax だけ FP32 で
           # 1.72・2.26 µs、速さは同じ(policy zoo の学習の段 2026-10-12)
           'engine_precision': 'fp16_norm_fp32'},
    genre_keys=_V0_GENRES, requires=('data/pretrained/dinov2_vits14_pretrain.pth',), requires_pretrained=True,
    train_only_keys=('finetune_last', 'lr_mult')))
