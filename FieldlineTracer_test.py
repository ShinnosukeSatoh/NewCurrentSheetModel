import JupiterMag as jm
from CSField import CSField
from FieldlineTracer import FieldlineTracer

# 初期設定（1回だけ）
jm.Internal.Config(Model='jrm33', CartesianIn=True, CartesianOut=True)

cs = CSField()
cs.config()

tracer = FieldlineTracer(cs_field=cs)

# トレース実行
result = tracer.trace(
    x0=10.0, y0=0.0, z0=0.0,
    MLT=0.0,
    direction='both',
    h_stop_km=500.0,   # 電離圏高度で止める場合
)

print(result['stop_fwd'], result['stop_bwd'])  # 'surface', 'r_max', 'max_arc'
