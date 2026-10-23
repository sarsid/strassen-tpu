# Execute the operational bootstrap from a real remote file

The replacement v6e was allocated on October 20 at 18:25 UTC. The first
operational launch restored the seed and then failed before recording launch
or starting evaluation. The notebook transport sets `__file__` to the submitted
laptop filename; the bootstrap attempted to hash that nonexistent remote path.

Preserve the partially prepared remote and local v009 run. Reuse its allocated
session and endpoint, unchanged scientific archive and seed, in a new v010 run.
Upload the exact existing operational bootstrap as a real remote file and call
it through the hash-verifying v001 wrapper. `runpy.run_path` supplies the actual
remote `__file__`. Do not change the bootstrap, frozen scientific worker,
checkpoint contract, environment pins, admission, datasets or scoring.

Validation reproduces notebook transport with a nonexistent host filename and
checks that the uploaded file can hash itself, receives unchanged arguments,
and cannot execute with the wrong hash or outside the experiment directory.
Only execute after confirming the previous foreground client exited and its
remote launch receipt remains absent. The allocated TPU and keepalive remain;
no new allocation or release is requested. Observe through the contents API
while the genuine evaluation notebook cell is busy.
