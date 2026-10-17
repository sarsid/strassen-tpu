# Gemma4 approved validation split — October 16

The user approved this change after reviewing the completed numerical diagnosis.
This supersedes the pending proposal v001 and the cross-backend BF16 stopping
rule for new attempts only. Historical failed runs remain failed and preserved.

Model fidelity: CPU FP32, real checkpoint and official layer inputs, finite
outputs and relative L2 <=1e-5 at layers 0, 5 and 53. The completed control is
reused only when checkpoint identity, official Transformers source and all
qualified model/control sources match the recorded hashes. The new attempt
records that it reuses evidence; it does not claim a new FP32 execution.
Any model source change requires a new real-weight control. BF16 CPU/Torch/TPU
drift and the exact-logit-tie finding remain reported diagnostics.

Algorithm quality: Native BF16 on the same v6e and identical token inputs is
the reference. Keep projection/block checks, four quality segments, repetitive
abs NLL delta <=0.01, KL <=0.005, agreement >=0.99; natural abs NLL delta <=0.01,
KL <=0.02, agreement >=0.97. Report errors and perplexity; agreement is not
accuracy against true labels. A failure is preserved, never relabeled a pass.

Natural text: each independent S1024 sequence is BOS plus 1023 contiguous
corpus tokens. Corpus positions are consumed once across segment/batch order.
Hash exact input tensors and explicit tokenization policy. Repetitive inputs
stay unchanged, with BOS asserted. Native/Tuned Native/Tuned Cubic/parent S1,
BF16 output, DEFAULT precision, FP32 accumulation, batches 8/16/32/64/4/1,
compiler pins, tuning and timing scopes remain unchanged.

Local regression checks precede execution. The prior diagnostic TPU disappeared
from Colab at 14:33:47 UTC without a controller release; independently confirmed
absent at 14:37. Use a new owned allocation and immutable attempt. Retain the TPU
on infrastructure or scientific failure; release only after every requested
batch completes and passes, or an explicit user instruction. Keep the dashboard
at port 8790 current. Publish code and numerical metrics, not activation tensors.
