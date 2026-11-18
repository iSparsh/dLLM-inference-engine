# Euler probe

`probe_euler.sh` is a one-shot, polite survey of what Euler gives us: partitions, GPU types,
limits, modules (CUDA/gcc/python), storage/quota, outbound internet from a compute node, and a
tiny CUDA compile-and-run test. It follows the course rules (`slurm_usage.md`): everything runs via
`sbatch` on the `instruction` partition, 1 GPU, 1 core, 8 GB, 10 min. Nothing but `sbatch` runs on
the login node.

## 1. Copy to Euler (from your laptop)

```bash
scp scripts/euler/probe_euler.sh <netid>@euler.engr.wisc.edu:~/
```

(The username is your **CAE** username; password is your CAE password.)

## 2. Run (on the Euler login node)

```bash
ssh <netid>@euler.engr.wisc.edu
mkdir -p ~/probe && cd ~/probe && mv ~/probe_euler.sh .
sbatch probe_euler.sh
```

Optional: also learn the compute capability of each GPU type in the partition (submits one extra
tiny job per distinct type, only once, from inside the main job):

```bash
sbatch probe_euler.sh --all-gpu-types
```

`bash probe_euler.sh` also works, but it only forwards to `sbatch` (the course says never to run
scripts directly on the login node, so prefer `sbatch`).

## 3. Where the output lands

In the directory you submitted from:

- `probe_<jobid>.out`: main log (stdout + stderr merged)
- `probe_<gputype>_<jobid>.out`: one per extra GPU type (only with `--all-gpu-types`)

```bash
squeue -u $USER                 # wait until the job is gone
cat probe_<jobid>.out
```

## 4. Optional: login-node internet check (run by hand, cheap)

The job tests internet from the compute node. To compare with the login node:

```bash
for u in https://huggingface.co https://pypi.org https://download.pytorch.org; do
  curl -sS -m 10 -o /dev/null -w "$u %{http_code}\n" $u; done
```

## 5. What to paste back

The whole contents of `probe_<jobid>.out` (and any `probe_<type>_*.out`), plus the output of the
login-node `curl` loop above. If the file is huge, the sections `SINFO`, `GPU TYPES PER PARTITION`,
`ACCOUNT / QOS / LIMITS`, `CUDA MODULE VERSIONS + NVCC`, `OUTBOUND INTERNET`, `LOCAL SCRATCH`,
`STORAGE`, and `CUDA COMPILE + RUN SANITY TEST` are the important ones.
