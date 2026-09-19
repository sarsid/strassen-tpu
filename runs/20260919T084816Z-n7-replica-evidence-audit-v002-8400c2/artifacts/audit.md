# N5–N9 evidence audit

Audit: PASS

## 20260919T084449Z-N7-replicate-v5e-v004-955768

Archive integrity: PASS

Status counts: `{"ok": 128}`

- native_xla vs cubic_selected: {"counts": {"win": 9, "inconclusive": 1, "loss": 6}, "median_ratio": 1.0283919803388155, "min_ratio": 0.9265580955034037, "max_ratio": 1.7792324522485605}
- native_xla vs selector: {"counts": {"inconclusive": 12, "loss": 1}, "median_ratio": 1.0000409781380275, "min_ratio": 0.9844594306323975, "max_ratio": 1.0395620245073762}
- cubic_selected vs native_xla: {"counts": {"loss": 9, "inconclusive": 1, "win": 6}, "median_ratio": 0.9724107671151391, "min_ratio": 0.5620401082142015, "max_ratio": 1.0792631404906077}
- strassen_selected vs native_xla: {"counts": {"loss": 9, "win": 6, "inconclusive": 1}, "median_ratio": 0.9685352394062756, "min_ratio": 0.5461833532439646, "max_ratio": 1.1014596980974045}
- strassen_selected vs cubic_selected: {"counts": {"inconclusive": 4, "loss": 4, "win": 8}, "median_ratio": 1.0144056837479378, "min_ratio": 0.8142194006934562, "max_ratio": 1.0756423323258506}
- selector vs native_xla: {"counts": {"inconclusive": 12, "win": 4}, "median_ratio": 1.0027645739864566, "min_ratio": 0.9619435651026944, "max_ratio": 1.1152331161898028}
- selector vs cubic_selected: {"counts": {"win": 9, "inconclusive": 5, "loss": 2}, "median_ratio": 1.035515701842901, "min_ratio": 0.9493676896971956, "max_ratio": 1.7887490993511124}
- cubic_selected vs selector: {"counts": {"inconclusive": 1}, "median_ratio": 0.9886557472931295, "min_ratio": 0.9886557472931295, "max_ratio": 0.9886557472931295}

## 20260919T084349Z-smoke-v5e-v004-bf46a0

Archive integrity: PASS

Status counts: `{"ok": 24}`

- native_xla vs cubic_basic: {"counts": {"loss": 1, "win": 2}, "median_ratio": 1.048817370548187, "min_ratio": 0.9594716128364662, "max_ratio": 1.1238938053097347}
- native_xla vs cubic_quadrant: {"counts": {"inconclusive": 2, "win": 1}, "median_ratio": 1.0182073398974694, "min_ratio": 0.9903967487285832, "max_ratio": 1.0929432464075246}
- cubic_basic vs native_xla: {"counts": {"win": 1, "loss": 2}, "median_ratio": 0.9534548416921514, "min_ratio": 0.889763779527559, "max_ratio": 1.0422403191728837}
- cubic_basic vs cubic_quadrant: {"counts": {"inconclusive": 3}, "median_ratio": 0.9724613137326793, "min_ratio": 0.9708147180717284, "max_ratio": 1.0322314235026648}
- cubic_quadrant vs native_xla: {"counts": {"inconclusive": 2, "loss": 1}, "median_ratio": 0.9821182393958161, "min_ratio": 0.9149605922237714, "max_ratio": 1.0096963679290596}
- cubic_quadrant vs cubic_basic: {"counts": {"inconclusive": 3}, "median_ratio": 1.0283185417028229, "min_ratio": 0.9687750026120169, "max_ratio": 1.0300626694105346}
- strassen_basic vs native_xla: {"counts": {"inconclusive": 2, "loss": 1}, "median_ratio": 0.9566898484533941, "min_ratio": 0.9377219506510552, "max_ratio": 0.9935341199199614}
- strassen_basic vs cubic_basic: {"counts": {"loss": 1, "inconclusive": 1, "win": 1}, "median_ratio": 1.003392931285032, "min_ratio": 0.9532677844476644, "max_ratio": 1.0538998914396818}
- strassen_basic vs cubic_quadrant: {"counts": {"inconclusive": 3}, "median_ratio": 0.983992962119644, "min_ratio": 0.9741086257006436, "max_ratio": 1.024876872972161}

