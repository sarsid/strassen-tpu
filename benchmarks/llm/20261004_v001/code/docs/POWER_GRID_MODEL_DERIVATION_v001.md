# A shape-only candidate-cost formula for the fixed TPU study

This is a proposed, computable model structure, not a fitted predictor or a claim of globally optimal tiles. It uses the existing exploratory kernel/candidate design; no reserved holdout was read or measured for this derivation. Hardware, JAX/compiler versions, BF16 inputs and Strassen pre-adds, FP32 accumulation/output, single-chip execution, resident inputs, and the chosen timing scope are fixed constants of the model.

The runtime input can be **only (M,N,K)**. Calibration coefficients and the finite implementation catalog are stored constants, just as a compiled kernel contains hardware-specific constants. Returning a custom tile is possible; native XLA's internal tile is not exposed by these implementations. A truthful native result must instead be `algorithm=native`, `tile=compiler_managed`, plus its selected compiler preset. Assigning a guessed native (BM,BN,BK) would not describe the implementation that was measured.

## 1. The complete decision formula

Let c identify an implementation: family a, its fixed variant, and either a custom tile b=(b_m,b_n,b_k) or one of the four native compiler presets. Freeze the tested catalog C: 24 cubic choices, 24 Strassen choices, and four native choices.

\[
c^*(M,N,K)=\operatorname*{argmin}_{c\in\mathcal C_{\mathrm{admissible}}(M,N,K)}
\widehat T_c(M,N,K).
\]

This is an explicit finite formula: at most 52 inexpensive scalar evaluations and one minimum, without runtime benchmarking. Its result is the **predicted lowest latency among the registered candidates**, not an assertion about all possible tiles, compiler transformations or matrix multiplication algorithms. Break exact ties deterministically; for nearly tied predictions, a separately registered policy can prefer lower estimated memory or native. Any such policy changes the selector and must be included in validation.

The most defensible initial target is synchronized **complete-call arithmetic mean latency**, because the experiment selected its winners on that scope. Train a separate model if prepared-kernel latency is desired. Do not force complete-call latency to equal prepared latency plus a nonnegative padding charge: the recorded scopes use different compiled executions and include ranking reversals even for aligned inputs.

## 2. Quantities determined exactly by shape and tile

For custom tile b define

\[
I=\left\lceil\frac M{b_m}\right\rceil,\quad
J=\left\lceil\frac N{b_n}\right\rceil,\quad
L=\left\lceil\frac K{b_k}\right\rceil,
\]
\[
M_p=Ib_m,\quad N_p=Jb_n,\quad K_p=Lb_k,\quad
V_p=M_pN_pK_p,\quad O=IJ,\quad q=IJL.
\]

O is the number of **logical independent output tiles**, L is the sequential K-panel count per output tile, and q is the number of tile-panel products. O is not measured physical concurrency, a core count, or a hardware-occupancy estimate. One full matrix multiplication remains one timed host call; q must not be interpreted as q host launches.

Source-estimated dot work, counting a multiply and an add as two FLOPs, is

\[
F_C=2V_p,\qquad F_S=\frac74V_p.
\]

The current source-level vector element-operation counts are

\[
V_C=q b_m b_n=\frac{V_p}{b_k},
\]
\[
V_S=q\left[\frac54b_k(b_m+b_n)+3b_mb_n\right]
=\frac54V_p\left(\frac1{b_m}+\frac1{b_n}\right)+3\frac{V_p}{b_k}.
\]

These count the explicit pre-additions and accumulator updates, not every compiler-generated instruction. Initialization, pipeline startup, stores and compiler effects need separate proxies. At equal tile and padding,

\[
\frac{V_S-V_C}{F_C-F_S}
=\frac5{b_m}+\frac5{b_n}+\frac8{b_k}.
\]

This explains why increasing tile dimensions can reduce the extra vector work per saved dot FLOP. It does not by itself identify the fastest tile: on-chip storage, transfers, partial tiles, dot efficiency and scheduling also change.

## 3. Data reuse, output size, and pipeline proxies

Separate one padded copy of each operand from repeated logical tile reads:

\[
U_A=2M_pK_p,\qquad U_B=2K_pN_p,\qquad C_{\rm out}=4M_pN_p,
\]
\[
R_A=2M_pK_p(J-1)=2V_p/b_n-U_A,
\]
\[
R_B=2K_pN_p(I-1)=2V_p/b_m-U_B.
\]

All are byte-valued **proxies** for BF16 input blocks and FP32 output blocks. U_A+R_A and U_B+R_B are what a simple no-cross-output-tile-reuse schedule requests logically. Actual warm-call HBM traffic was not measured. In particular, these counts must not be divided by latency and called achieved HBM bandwidth; earlier evidence already showed why original-operand bytes are not a validated per-warm-call traffic lower bound.

A physical interpretation is

\[
Q_{\rm proxy}=U_A+U_B+C_{\rm out}+\rho_A R_A+\rho_B R_B,
\qquad 0\le\rho_A,\rho_B\le1,
\]

where the rho terms describe an effective retained fraction of redundant logical reads. They are phenomenological parameters, not measured cache hit rates. M and N need not be interchangeable: operand layouts and traversal can give different effective A and B costs.

O represents opportunities for per-output-tile initialization, drain and restart cost; q represents repeated panel/product scheduling cost. Including both distinguishes a small output grid with a long reduction from a large output grid with a short reduction. Avoid imposing a generic “more output tiles always means more physical parallelism” penalty: that is not established for this single-TPU Pallas implementation.

The existing exploratory contrast is informative here: the 8192 cube and the three shapes formed by placing 131072 on one axis and 2048 on the other two have equal volume and use the same Strassen tile (2048,2048,512). Their F_S and V_S estimates are identical, but O×L is respectively 16×16, 64×4, 64×4 and 1×256. Their output footprints and logical reuse opportunities differ, and their measured latencies differ. A formula using only MNK and the elementary arithmetic counts cannot reproduce this distinction.

## 4. Padding and cropping must be explicit

Let indicator A be one when M_p differs from M or K_p differs from K; define indicator B analogously for K,N, and indicator C for M,N. A transparent padding/crop proxy is

\[
D_{\rm pad}=2(MK+M_pK_p)\,1_A
+2(KN+K_pN_p)\,1_B
+4(MN+M_pN_p)\,1_C.
\]

This represents possible original reads plus padded writes and output crop handling. It is not asserted to be actual traffic: the compiler may elide, combine or reschedule operations. Include an indicator `P = 1{(M_p,N_p,K_p) != (M,N,K)}` for a fixed cost of entering a different preparation path. Ceilings make both P and D_pad discontinuous at relevant tile boundaries, which a smooth MNK-only power law would miss.

Record directional padding ratios and their product rho=V_p/(MNK). For independently selected cubic and Strassen tiles, retaining a lower Strassen dot-work estimate requires

\[
\frac{\rho_S}{\rho_C}<\frac87.
\]

This is only a condition for fewer estimated dot FLOPs, not for lower time. For example, the observed 1025-cube choices have rho_S/rho_C=16/9, producing F_S/F_C=14/9 rather than a saving.

## 5. A compact model that the current data can estimate

For each custom family, start with the following nonnegative effective-cost model:

\[
\boxed{
\begin{aligned}
\widehat T_a={}&\ell_a+d_aF_a
+u^A_aU_A+u^B_aU_B
+r^A_aR_A+r^B_aR_B\\
&+v_a\frac{V_p}{b_k}
+w_aC_{\rm out}
+g_aO+h_aq
+p_aD_{\rm pad}+j_aP.
\end{aligned}}
\]

This has twelve coefficients per family including the fixed-cost intercept. The fitted coefficients are effective predictive rates/costs; none should be advertised as a separately recovered TPU peak throughput, vector rate, bandwidth or overlap fraction. A smaller first version can tie A/B coefficients, or omit whichever of U/R/O/q has no stable validation benefit. Scale features using training-only constants before fitting and regularization, then freeze that scaling with the coefficients.

There is a specific identifiability trap: adding V_S as another independent regressor alongside all input/reuse terms and V_p/b_k introduces exact linear dependence,

\[
V_S=\frac58(U_A+R_A+U_B+R_B)+3V_p/b_k.
\]

For cubic V_C=V_p/b_k already. Therefore aggregate latency alone cannot separate the contribution of vector pre-adds from movement when their predictors have these relationships. The boxed model intentionally combines them into identifiable feature directions. It may still have empirical collinearity; inspect condition numbers and coefficient stability rather than assigning physical meaning to unstable fitted values.

The factor 7/4 should remain visible in F_S, but fitting separate d_C and d_S permits different effective dot efficiency for full versus half-tile products. Enforcing d_C=d_S can be tested as a stronger restricted model, not assumed to be physically exact. Similarly, F_a, q and inverse-tile features can absorb finite dot-shape efficiency without adding several unidentifiable “MXU utilization” factors. Add a correction only if grouped validation shows a stable improvement.

A simple native model for each preset p is

\[
\widehat T_{X,p}=\ell_{X,p}+a_p(2MNK)+b_p(2MK)+c_p(2KN)+d_p(4MN).
\]

It uses no invented native tile. Its limitations are real: native tiling and compiler decisions can introduce shape discontinuities. If the five-term form underfits, a small fixed set of alignment-residual or aspect-ratio basis functions can be added and selected inside grouped validation. Such alignment features are predictive bases, not claims to have recovered XLA's internal padding. Avoid fitting an arbitrary high-order three-dimensional surface to only sixty distinct geometries.

## 6. An analytical tile-balance relation

Away from padding boundaries, F_a, U_A, U_B and C_out are constant in tile choice. The leading tile-dependent terms of the boxed model have the form

\[
\frac{\widehat T_a-\text{tile-independent terms}}{MNK}
\simeq\frac A{b_m}+\frac B{b_n}+\frac C{b_k}
+\frac{G}{K b_m b_n}+\frac H{b_m b_n b_k}.
\]

The first two terms summarize transfer/reuse and input-combination cost, the third summarizes panel accumulation cost, and the last two represent output-tile and panel scheduling. This makes K's directional role explicit, even at equal MNK. It also explains why maximizing only BK or only BM×BN need not be optimal.

For a continuous heuristic, temporarily omit the final two terms and use a simplified on-chip storage budget

\[
c_o b_m b_n+c_i b_k(b_m+b_n)\le S.
\]

Here c_o and c_i describe effective bytes per output/input tile element including selected buffering assumptions; S is an effective tile storage allowance. The source suggests FP32 output and BF16 inputs, but compiler buffers, Strassen intermediates, alignment and pipeline liveness prevent equating this expression to exact compiled VMEM usage. All tested candidates passed this campaign, so these data do not identify a failure frontier or prove the entire configured 48 MiB is available to this elementary formula.

The interior stationary conditions for minimizing A/b_m+B/b_n+C/b_k are

\[
\frac A{b_m^2}=\lambda(c_o b_n+c_i b_k),\quad
\frac B{b_n^2}=\lambda(c_o b_m+c_i b_k),\quad
\frac C{b_k^2}=\lambda c_i(b_m+b_n).
\]

These are a useful explicit balance law rather than an arbitrary rule that every tile dimension should be equal. When A=B=a, symmetry gives b_m=b_n=s. Set

\[
r=\frac{c_iC}{ac_o},\qquad
u=\frac{2}{2+r+\sqrt{r^2+8r}}.
\]

Then the relaxed optimum is

\[
s=\sqrt{\frac{uS}{c_o}},\qquad
b_k=\frac{S-c_os^2}{2c_is}.
\]

This follows from `(1-u)^2 = r u(1+u)` and applies for positive coefficients and an active budget. It is an explanatory candidate generator, not the final dispatcher. Real tiles are discrete, asymmetric costs can matter, inputs can be smaller than the relaxed tile, and ceiling penalties can dominate. Round neighboring candidates to the actual alignment constraints—cubic (8,128,128), Strassen (16,256,256)—then evaluate the exact ceiling-based cost. For the first fitted predictor, stay within the already tested 24-per-family catalog instead of claiming those newly generated tiles are qualified.

The raw operation-count ratio `5/BM+5/BN+8/BK` must not be minimized as if all terms had equal time cost. Input/reuse movement and vector accumulation have different effective costs. In particular, that dimensionless source-work ratio alone does not derive the observed preference for BM=BN=2048, BK=512; the fitted cost weights and feasibility constraints are essential.

## 7. Calibration and validation supported by the existing evidence

The screen contains 3,120 candidate observations per timing scope: 60 shapes × (24 cubic +24 Strassen +4 native). They provide tile variation, but there are still only sixty distinct geometries. Seven timing rounds from one observation are not seven independent shapes or device allocations.

Fit on the **full screening candidate table**, not only its selected winners. Use numerical/compilation eligibility as a separate mask; do not give failures an artificial large measured latency. Fit with shape-balanced relative or log error so the 16k cube does not determine every coefficient. A practical positive-time objective is a robust loss on `log(T_hat/T_observed)` with training-normalized coefficient regularization; a relative-error weighted nonnegative least-squares version is another transparent starting point.

All candidates and scopes belonging to a geometry must stay in the same fold. Otherwise test-shape timing leaks through another tile or scope. Grouped folds over the sixty exploratory geometries can compare small model variants and regularization. A stricter family/size-held-out sensitivity analysis is useful because near-neighbor shapes are not independent coverage of new regimes. Do not consume the twelve reserved shapes during this development.

Validate the **decision**, not just latency regression: predicted winner's observed regret relative to the lowest observed eligible candidate on held-out exploratory shapes, algorithm agreement with uncertainty, tolerance-set coverage, errors by size/aspect/padding stratum, and comparisons to always-native and a simple fixed-tile baseline. The lowest observed screen mean is a noisy finite-catalog reference, not an oracle. Bootstrap or rerun variability must not be called independent machine replication.

The confirmation set is selected by the original screen shortlist. A newly fitted model can choose a catalog candidate absent from that shortlist. Its independent confirmation latency is then **missing**: do not silently evaluate a different candidate, replace it with the screen winner, or report its screening time as fresh validation. Report candidate coverage and keep the screen-based grouped evaluation separate. Freeze the chosen formula, coefficients and policy before any later reserved-shape measurement.

## 8. What a shape-only return value can and cannot mean

The returned mapping can contain algorithm, fixed variant, custom tile or native preset, predicted latency, predicted runner-up gap, and a modeled near set. All can be deterministic functions of M,N,K and stored constants. A modeled near set remains a set of explicit candidate tuples; a Cartesian range is not implied.

An admission check is necessary for arbitrary inputs: even BF16 A/B plus one FP32 output require `2MK+2KN+4MN` bytes before padding and temporary storage. Shapes beyond the registered resident-memory budget must return an out-of-domain/unsupported result rather than pretending native makes them fit. Compiler feasibility on unseen geometries and exact peak liveness are not proven by this cost formula. Within the tested design, a catalog qualification mask is safer than learning a fictitious OOM boundary from zero failures.

Finally, M,N,K alone cannot determine numerical safety for arbitrary matrix entries. The formula applies to the fixed validated synthetic input contract; earlier cancellation failures remain relevant. Hardware/software changes, other precision, more Strassen levels, and different timing scopes require new calibration. “Optimal” should therefore mean **predicted best eligible registered choice under this fixed contract**, until independent evidence supports a stronger statement.

Primary implementation basis: [kernel v001](../src/strassen_mm/kernels_v001.py), [kernel v002](../src/strassen_mm/kernels_v002.py), [registered protocol](../protocols/POWER_MIDPOINT_GRID_v001.md), and [exploratory findings review](POWER_GRID_FINDINGS_REVIEW_v001.md). This document performs no fitting, new benchmark, holdout access, source/configuration edit or commit.
