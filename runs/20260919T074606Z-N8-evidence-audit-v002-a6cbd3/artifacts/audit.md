# N5–N9 evidence audit

Audit: PASS

## 20260919T073511Z-N8-v5e-v004-0a55e8

Archive integrity: PASS

Status counts: `{"ok": 312}`

- cubic_full_packed_unfused vs cubic_full_unpacked_unfused: {"counts": {"loss": 8}, "median_ratio": 0.3126660135530524, "min_ratio": 0.21884917806043877, "max_ratio": 0.5814544383629111}
- cubic_full_fused vs cubic_full_packed_unfused: {"counts": {"win": 8}, "median_ratio": 1.273695369164233, "min_ratio": 1.1749513667338283, "max_ratio": 1.5709401378314736}
- cubic_quadrant_packed_unfused vs cubic_quadrant_unpacked_unfused: {"counts": {"loss": 8}, "median_ratio": 0.32821529842046426, "min_ratio": 0.23177810274002927, "max_ratio": 0.5831862738653497}
- cubic_quadrant_fused vs cubic_quadrant_packed_unfused: {"counts": {"win": 8}, "median_ratio": 1.2931510627988572, "min_ratio": 1.1973296172278731, "max_ratio": 1.591187541700521}
- strassen_unpacked_unfused vs native_joint_graph: {"counts": {"inconclusive": 2, "loss": 9, "win": 5}, "median_ratio": 0.9466905021250112, "min_ratio": 0.49065694446132885, "max_ratio": 1.192944115683644}
- strassen_unpacked_unfused vs cubic_full_unpacked_unfused: {"counts": {"inconclusive": 4, "win": 12}, "median_ratio": 1.0454799369563013, "min_ratio": 0.9994999453572898, "max_ratio": 1.0620742313421647}
- strassen_unpacked_unfused vs cubic_quadrant_unpacked_unfused: {"counts": {"win": 14, "inconclusive": 2}, "median_ratio": 1.1211877689353693, "min_ratio": 1.0200667273468644, "max_ratio": 1.1634231616291995}
- strassen_packed_unfused vs strassen_unpacked_unfused: {"counts": {"loss": 8}, "median_ratio": 0.304327604373267, "min_ratio": 0.21171869666489398, "max_ratio": 0.5712764594994721}
- strassen_packed_unfused vs native_joint_graph: {"counts": {"loss": 8}, "median_ratio": 0.2625764270911738, "min_ratio": 0.10388124879093182, "max_ratio": 0.571898196079089}
- strassen_packed_unfused vs cubic_full_packed_unfused: {"counts": {"inconclusive": 1, "win": 7}, "median_ratio": 1.0112181499416937, "min_ratio": 1.0074813509190763, "max_ratio": 1.0179786515189126}
- strassen_packed_unfused vs cubic_quadrant_packed_unfused: {"counts": {"inconclusive": 1, "win": 7}, "median_ratio": 1.0287279294726546, "min_ratio": 1.022663421601235, "max_ratio": 1.0412587492950405}
- strassen_fused vs strassen_packed_unfused: {"counts": {"win": 8}, "median_ratio": 1.272151417490965, "min_ratio": 1.1905999717226743, "max_ratio": 1.6004681627083193}
- strassen_fused vs native_joint_graph: {"counts": {"loss": 10, "win": 6}, "median_ratio": 0.7804209838397362, "min_ratio": 0.1236810118729995, "max_ratio": 1.2422730955885992}
- strassen_fused vs cubic_full_fused: {"counts": {"inconclusive": 1, "win": 15}, "median_ratio": 1.038097378158889, "min_ratio": 0.9880077380303577, "max_ratio": 1.1135344962862914}
- strassen_fused vs cubic_quadrant_fused: {"counts": {"inconclusive": 2, "win": 14}, "median_ratio": 1.0355755067301864, "min_ratio": 0.9916242941240944, "max_ratio": 1.1146969443733614}
- strassen_early vs strassen_fused: {"counts": {"win": 1, "loss": 12, "inconclusive": 3}, "median_ratio": 0.9933898889454125, "min_ratio": 0.9452014100534597, "max_ratio": 1.0424039580677757}
- strassen_early vs native_joint_graph: {"counts": {"loss": 10, "inconclusive": 1, "win": 5}, "median_ratio": 0.7998201352790617, "min_ratio": 0.1232745993438515, "max_ratio": 1.195979725417027}
- strassen_early vs cubic_full_fused: {"counts": {"win": 14, "inconclusive": 2}, "median_ratio": 1.0323359138702206, "min_ratio": 0.9723274346408408, "max_ratio": 1.0607303714305358}
- strassen_early vs cubic_quadrant_fused: {"counts": {"win": 15, "loss": 1}, "median_ratio": 1.0318769325819654, "min_ratio": 0.9696426481736682, "max_ratio": 1.0576072831016512}
- cubic_full_fused vs cubic_full_unpacked_unfused: {"counts": {"inconclusive": 2, "loss": 2, "win": 4}, "median_ratio": 1.0089844235270995, "min_ratio": 0.9895355315512073, "max_ratio": 1.0306934071201928}
- cubic_quadrant_fused vs cubic_quadrant_unpacked_unfused: {"counts": {"inconclusive": 2, "win": 6}, "median_ratio": 1.0903735378881114, "min_ratio": 1.0175926998329454, "max_ratio": 1.124861164317318}
- strassen_fused vs strassen_unpacked_unfused: {"counts": {"inconclusive": 1, "win": 7}, "median_ratio": 1.0386240419410946, "min_ratio": 1.0238243627219208, "max_ratio": 1.0708033156367935}
- strassen_n5_selected_unfused vs native_joint_graph: {"counts": {"inconclusive": 1, "loss": 1}, "median_ratio": 0.9169118811463486, "min_ratio": 0.8311232237845521, "max_ratio": 1.0027005385081451}
- strassen_n5_selected_unfused vs cubic_n5_selected_unfused: {"counts": {"inconclusive": 1, "loss": 1}, "median_ratio": 0.9904656194356385, "min_ratio": 0.9741553776560478, "max_ratio": 1.0067758612152293}

