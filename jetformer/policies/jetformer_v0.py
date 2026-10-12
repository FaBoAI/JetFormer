"""jetformer_v0: 今までの JetFormer v0(jetformer/model.py の JetFormer)

包まずに JetFormer をそのまま返すので、今までの best.pth・last.pth・jetformer.onnx・エンジンはそのまま使える
(state_dict のキーも、書き出す ONNX のグラフも今までと同じ)。
"""

from ..model import JetFormer


def build(num_modes, pretrained=False, dim=256, layers=4, heads=8):
    return JetFormer(num_modes=num_modes, dim=dim, layers=layers, heads=heads, pretrained=pretrained)
