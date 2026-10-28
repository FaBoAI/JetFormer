import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json, time
from wd import Browser
b = Browser(900, 1000)
try:
    b.go('http://127.0.0.1:18771/phone?w=420')
    time.sleep(3)
    b.frame(0)
    b.js("document.documentElement.style.scrollbarWidth='none'; showTab('drive')")
    time.sleep(3)
    print(json.dumps(b.js("return {iw: innerWidth, cw: document.documentElement.clientWidth, sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight, ver: document.querySelector('#dr-client-ver').textContent}"), ensure_ascii=False))
finally:
    b.close()
