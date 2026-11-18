#!/usr/bin/env bash
# Euler capability probe for ECE 759 (UW-Madison).
#
# Conventions followed (from the course's slurm_usage.md / Euler lecture slides):
#   * everything that runs, runs through sbatch on a compute node, partition "instruction"
#   * 1 GPU (--gres=gpu:1), 1 CPU core, short time limit (course max on instruction = 60 min)
#   * nothing but `sbatch` is executed on the login node
#
# Usage (preferred):   sbatch probe_euler.sh [--all-gpu-types]
# Also accepted:       bash probe_euler.sh [--all-gpu-types]   (only forwards to sbatch)
#
# Flags:
#   --all-gpu-types   after the main probe, submit ONE extra tiny GPU-only job per distinct
#                     GPU type found in the instruction partition (explicit gres syntax
#                     --gres=gpu:<type>:1), so we learn each type's compute capability.
#   --gpu-only        (internal) run only the per-node GPU/toolchain section.
#
#SBATCH -p instruction
#SBATCH -J probe-euler
#SBATCH -t 0-00:10:00
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=8G
#SBATCH -o probe_%j.out
# (stderr goes to the same file as stdout when -e is not given)

PARTITION=instruction
ALL_TYPES=0
GPU_ONLY=0
for a in "$@"; do
  case "$a" in
    --all-gpu-types) ALL_TYPES=1 ;;
    --gpu-only)      GPU_ONLY=1 ;;
    *) echo "unknown arg: $a (valid: --all-gpu-types)" ; exit 2 ;;
  esac
done

###############################################################################
# Launcher mode: not inside a Slurm job -> just hand ourselves to sbatch.
###############################################################################
if [ -z "${SLURM_JOB_ID:-}" ]; then
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch not found. Run this on the Euler login node (euler.engr.wisc.edu)."
    exit 1
  fi
  self=$(readlink -f "$0")
  out=$(PROBE_SCRIPT="$self" sbatch --parsable "$self" "$@") || { echo "sbatch failed: $out"; exit 1; }
  jobid=${out%%;*}
  echo "Submitted probe job $jobid (partition $PARTITION, 1 GPU, 10 min max)."
  echo "Check queue:      squeue -u \$USER"
  echo "View the result:  cat $PWD/probe_${jobid}.out      (or: tail -f probe_${jobid}.out)"
  [ "$ALL_TYPES" = 1 ] && echo "Per-GPU-type jobs will be submitted by the main job; results land in probe_<type>_<jobid>.out"
  exit 0
fi

###############################################################################
# Helpers (inside the Slurm job from here on)
###############################################################################
cd "${SLURM_SUBMIT_DIR:-$PWD}" || true
SELF="${PROBE_SCRIPT:-}"
if [ -z "$SELF" ] || [ ! -f "$SELF" ]; then
  SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi

section() { echo; echo "===== $* ====="; }
# run: print the command, run it, never abort
run() { echo "\$ $*"; "$@" 2>&1 || echo "FAILED (rc=$?): $*"; }
# runsh: same, but for a shell pipeline given as one string
runsh() { echo "\$ $1"; bash -c "$1" 2>&1 || echo "FAILED (rc=$?): $1"; }

# Make the `module` command available in this non-interactive bash.
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
           /opt/apps/lmod/lmod/init/bash /usr/share/lmod/lmod/init/bash \
           /usr/share/Modules/init/bash /etc/profile.d/z00_lmod.sh; do
    [ -f "$f" ] && . "$f" 2>/dev/null && type module >/dev/null 2>&1 && break
  done
fi
HAVE_MODULE=0; type module >/dev/null 2>&1 && HAVE_MODULE=1

# Terse module listing for a pattern (module avail writes to stderr).
modlist() { module -t avail "$@" 2>&1 | grep -v '^$' | grep -v ':$' | sed 's/(.*)//'; }

###############################################################################
# Per-node GPU / toolchain section (runs on the allocated GPU node)
###############################################################################
gpu_section() {
  section "JOB INFO"
  echo "date: $(date)"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-?}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  runsh 'echo "hostname: $(hostname)"; echo "user: $USER"; uname -srm; cat /etc/os-release 2>/dev/null | head -3'
  runsh 'scontrol show job $SLURM_JOB_ID | grep -E "JobState|TimeLimit|TRES|Gres|Partition|NodeList"'

  section "NVIDIA-SMI"
  run nvidia-smi
  run nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.memory,clocks.max.sm,power.limit,driver_version --format=csv
  run nvidia-smi -L
  runsh 'nvidia-smi topo -m | head -5'

  section "CPU / MEMORY"
  runsh 'lscpu | head -20'
  run free -g
  runsh 'echo "cgroup mem limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"; echo "nproc: $(nproc)"'

  section "MODULES: AVAILABLE (cuda/nvhpc/gcc/cmake/python/conda)"
  if [ "$HAVE_MODULE" = 1 ]; then
    echo "\$ module -t avail (filtered)"
    modlist 2>&1 | grep -iE 'cuda|nvhpc|nvidia|gcc|cmake|python|conda|mamba|miniforge|anaconda|ninja|cudnn|nccl' || echo "(no matches)"
    echo "--- full module list (terse), for reference ---"
    modlist 2>&1 | head -300
  else
    echo "FAILED: 'module' command not available in this shell"
  fi

  section "CUDA MODULE VERSIONS + NVCC"
  CUDA_MODS=""
  if [ "$HAVE_MODULE" = 1 ]; then
    CUDA_MODS=$(modlist nvidia/cuda cuda 2>&1 | grep -iE 'cuda' | sort -uV)
    echo "CUDA-related modules found:"; echo "${CUDA_MODS:-(none)}"
    echo "--- default 'module load nvidia/cuda' (as in the course HW01 example) ---"
    ( module load nvidia/cuda 2>&1; module list 2>&1; which nvcc; nvcc --version 2>&1 | tail -3 ) || echo "FAILED: default nvidia/cuda"
    for m in $CUDA_MODS; do
      echo "--- $m ---"
      ( module load "$m" 2>&1; which nvcc; nvcc --version 2>&1 | tail -3 ) || echo "FAILED: $m"
    done
  fi
  # newest versioned CUDA module (nvidia/cuda/X.Y[.Z])
  NEWEST_CUDA=$(echo "$CUDA_MODS" | grep -E '/[0-9]+(\.[0-9]+)*$' | sort -V | tail -1)
  echo "newest CUDA module: ${NEWEST_CUDA:-(none; will use default nvidia/cuda or system nvcc)}"

  section "GCC"
  runsh 'gcc --version | head -1'
  if [ "$HAVE_MODULE" = 1 ]; then
    GCC_MODS=$(modlist gcc 2>&1 | grep -E '^gcc/' | sort -uV)
    echo "gcc modules: $(echo $GCC_MODS)"
    if echo "$GCC_MODS" | grep -qx 'gcc/13.2.0'; then
      module load gcc/13.2.0 2>&1 && echo "loaded gcc/13.2.0 (the version named in the course example)"
    fi
    runsh 'gcc --version | head -1; g++ --version | head -1'
  fi

  section "PYTHON / TOOLS"
  runsh 'for t in python3 python pip3 pip uv conda mamba cmake ninja make git curl wget; do printf "%-8s " $t; (command -v $t || echo "-"); done'
  runsh 'python3 --version; pip3 --version; uv --version; conda --version; cmake --version | head -1; ninja --version; git --version'
  runsh 'python3 -c "import sys;print(sys.version)"; python3 -c "import torch;print(\"torch\",torch.__version__,torch.version.cuda,torch.cuda.is_available())"'
  runsh 'python3 -m venv --help >/dev/null && d=${TMPDIR:-/tmp}/probe_venv_$$ && python3 -m venv $d && echo "venv creation OK ($d)" ; rm -rf ${TMPDIR:-/tmp}/probe_venv_$$'

  section "OUTBOUND INTERNET FROM COMPUTE NODE"
  for u in https://huggingface.co https://pypi.org https://files.pythonhosted.org https://download.pytorch.org https://github.com \
           https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json; do
    printf '%-80s ' "$u"
    curl -sS -m 10 -L -o /dev/null -w "%{http_code}\n" "$u" 2>&1 || echo "FAILED: $u"
  done
  runsh 'env | grep -iE "^(http|https|no|all)_proxy=" || echo "(no proxy env vars)"'

  section "LOCAL SCRATCH / TMPDIR"
  runsh 'echo "TMPDIR=$TMPDIR  SLURM_TMPDIR=${SLURM_TMPDIR:-}  SCRATCH=${SCRATCH:-}"'
  runsh 'df -h /tmp ${TMPDIR:-/tmp} $HOME /dev/shm 2>&1'
  runsh 'ls -ld /scratch /scratch/* /srv /srv/* /mnt /mnt/* /local /local/* /data 2>/dev/null | head -30'
  runsh 'mount | grep -E " (/home|/scratch|/srv|/tmp) " | head'

  section "CUDA COMPILE + RUN SANITY TEST"
  T=$(mktemp -d "${TMPDIR:-/tmp}/probe_cuda.XXXXXX") || T=$PWD
  cat > "$T/t.cu" <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int *o) { o[threadIdx.x] = threadIdx.x * 2; }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
int main() {
  int drv = 0, rt = 0, n = 0;
  CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt)); CK(cudaGetDeviceCount(&n));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int memclk = 0, buswidth = 0, smclk = 0;
  CK(cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0));
  CK(cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, 0));
  CK(cudaDeviceGetAttribute(&smclk, cudaDevAttrClockRate, 0));
  printf("devices: %d  driver CUDA: %d  runtime CUDA: %d\n", n, drv, rt);
  printf("name: %s\ncompute capability: %d.%d\nSMs: %d\nglobal mem: %.1f GiB\n", p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
  printf("mem bus width: %d bit\nmem clock: %.0f MHz\nSM clock: %.0f MHz\n", buswidth, memclk / 1000.0, smclk / 1000.0);
  printf("peak mem BW (2*clk*bus/8): %.0f GB/s\nL2: %d KiB  smem/block optin: %zu KiB  regs/SM: %d\n", 2.0 * memclk * 1e3 * (buswidth / 8) / 1e9, p.l2CacheSize / 1024, p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor);
  int *d, h[32]; CK(cudaMalloc(&d, sizeof h)); k<<<1, 32>>>(d); CK(cudaGetLastError());
  CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; i++) if (h[i] != 2 * i) { printf("KERNEL RESULT MISMATCH\n"); return 1; }
  printf("kernel launch OK\n"); return 0;
}
EOF
  # Try newest CUDA first, then fall back to older ones (driver may not support the newest).
  CANDS=$(echo "$CUDA_MODS" | grep -E '/[0-9]+(\.[0-9]+)*$' | sort -rV | head -4)
  [ -z "$CANDS" ] && CANDS="nvidia/cuda"
  OK=0
  for m in $CANDS; do
    [ "$HAVE_MODULE" = 1 ] && { echo "--- loading $m ---"; module load "$m" 2>&1; }
    command -v nvcc >/dev/null 2>&1 || { echo "FAILED: nvcc not on PATH after loading $m"; continue; }
    nvcc --version | tail -2
    for arch in native sm_80 sm_86 sm_75 sm_70 sm_61; do
      flag="-arch=$arch"
      echo "\$ nvcc $flag $T/t.cu -o $T/t"
      if nvcc $flag "$T/t.cu" -o "$T/t" 2>&1; then
        echo "compile OK with $flag ($m)"
        echo "\$ $T/t"
        if "$T/t" 2>&1; then echo "RUN OK"; OK=1; else echo "FAILED: run (rc=$?) with $m $flag"; fi
        break
      else
        echo "FAILED: compile with $flag ($m)"
      fi
    done
    [ "$OK" = 1 ] && break
  done
  [ "$OK" = 1 ] && echo "CUDA SANITY TEST: SUCCESS" || echo "CUDA SANITY TEST: FAILED (all attempts)"
  [ "$T" != "$PWD" ] && rm -rf "$T"
}

###############################################################################
# Cluster-wide survey (read-only Slurm/storage queries; run once, in the main job)
###############################################################################
cluster_survey() {
  section "SINFO (all partitions)"
  run sinfo -o "%P %a %l %D %G %f %N"
  section "SINFO (instruction partition, per node)"
  run sinfo -p "$PARTITION" -N -o "%N %T %c %m %G %f"

  section "SCONTROL SHOW PARTITION $PARTITION"
  run scontrol show partition "$PARTITION"
  section "SCONTROL SHOW PARTITION (all, key fields)"
  runsh 'scontrol show partition | grep -E "PartitionName|AllowGroups|AllowAccounts|AllowQos|MaxTime|DefaultTime|MaxNodes|MaxCPUsPerNode|DefMemPerCPU|MaxMemPer|TRES|OverSubscribe|State"'

  section "GPU TYPES PER PARTITION (from sinfo GRES)"
  runsh 'sinfo -h -o "%P|%G|%D" | sort | awk -F"|" "{printf \"%-20s %-45s nodes=%s\n\", \$1, \$2, \$3}"'
  echo "--- distinct GPU types per partition ---"
  runsh 'sinfo -h -o "%P|%G" | awk -F"|" "{n=split(\$2,a,\",\"); for(i=1;i<=n;i++) if(a[i] ~ /^gpu/){split(a[i],b,\":\"); t=(b[2] ~ /^[0-9]/)?\"(untyped)\":b[2]; print \$1, t}}" | sort | uniq -c'
  echo "--- instruction-partition nodes: Gres / CPUs / RealMemory / features (scontrol) ---"
  runsh 'for n in $(sinfo -h -p '"$PARTITION"' -N -o "%N" | sort -u | head -60); do scontrol show node $n -o | grep -oE "NodeName=[^ ]+|Gres=[^ ]+|CPUTot=[^ ]+|RealMemory=[^ ]+|ActiveFeatures=[^ ]+" | tr "\n" " "; echo; done'

  section "ACCOUNT / QOS / LIMITS"
  runsh 'id; groups'
  runsh 'sacctmgr -n show assoc user=$USER format=Cluster,Account,User,Partition,QOS,MaxJobs,MaxSubmit,MaxWall,GrpTRES,MaxTRES,DefaultQOS%20'
  runsh 'sacctmgr -n show qos format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxTRES,GrpTRES'
  runsh 'sshare -U 2>&1 | head -20'
  runsh 'scontrol show config | grep -iE "MaxJobCount|MaxArraySize|DefMemPerCPU|DefMemPerNode|GresTypes|SelectTypeParameters|MaxSubmit|PriorityType|SlurmVersion|TaskPlugin|ProctrackType"'
  runsh 'squeue -p '"$PARTITION"' -h | wc -l | sed "s/^/jobs currently in '"$PARTITION"' queue: /"; squeue -u $USER'

  section "MODULES (login-style survey from compute node)"
  if [ "$HAVE_MODULE" = 1 ]; then
    for pat in cuda nvhpc gcc cmake python conda mamba miniforge anaconda; do
      echo "--- module avail $pat ---"
      modlist "$pat" 2>&1 | head -40
    done
    runsh 'module list 2>&1'
    runsh 'echo "MODULEPATH=$MODULEPATH"'
  else
    echo "FAILED: 'module' command not available"
  fi

  section "STORAGE: HOME / QUOTA / SCRATCH"
  runsh 'echo "HOME=$HOME"; df -h $HOME'
  runsh 'quota -s 2>&1 | head -20'
  runsh 'ls -d /scratch* /srv* /mnt/* /data* /work* /local* 2>/dev/null; echo "SCRATCH=${SCRATCH:-} WORK=${WORK:-}"'
  runsh 'df -h 2>/dev/null | head -40'
  runsh 'du -sh $HOME 2>/dev/null | head -1'
  runsh 'echo "HF_HOME=${HF_HOME:-} XDG_CACHE_HOME=${XDG_CACHE_HOME:-}"; ls -la $HOME | head -30'
  runsh 'ulimit -a | head -20'

  section "USER-LEVEL TOOLS"
  runsh 'for t in python3 uv conda mamba pip3 git cmake ninja; do printf "%-8s " $t; (command -v $t || echo "-"); done'
  runsh 'python3 --version; uv --version; conda --version'
}

###############################################################################
# Main
###############################################################################
echo "ECE 759 Euler probe -- job $SLURM_JOB_ID on $(hostname) -- $(date)"
echo "script: ${SELF:-unknown}   args: $*"

if [ "$GPU_ONLY" = 1 ]; then
  gpu_section
else
  cluster_survey
  gpu_section
fi

# --- optional: one tiny GPU-only job per distinct GPU type -------------------
if [ "$ALL_TYPES" = 1 ] && [ "$GPU_ONLY" = 0 ]; then
  section "PER-GPU-TYPE JOBS (--all-gpu-types)"
  # Distinct typed GRES names for the instruction partition, e.g. gpu:gtx1080:2(S:0-1) -> gtx1080
  TYPES=$(sinfo -h -p "$PARTITION" -o "%G" 2>/dev/null | tr ',' '\n' | awk -F: '/^gpu:/ { if ($3 != "" && $2 !~ /^[0-9]+$/) print $2 }' | sort -u)
  MYTYPE=$(sinfo -h -p "$PARTITION" -n "$(hostname -s)" -o "%G" 2>/dev/null | tr ',' '\n' | awk -F: '/^gpu:/ { if ($3 != "" && $2 !~ /^[0-9]+$/) print $2 }' | head -1)
  echo "GPU types found: ${TYPES:-(none -- GRES is untyped; nothing extra to submit)}"
  echo "type already covered by this job (node $(hostname -s)): ${MYTYPE:-unknown}"
  if [ -z "$SELF" ] || [ ! -f "$SELF" ]; then
    echo "FAILED: cannot locate original script path; not submitting per-type jobs"
  else
    for t in $TYPES; do
      [ "$t" = "$MYTYPE" ] && { echo "skip $t (covered by this job)"; continue; }
      echo "\$ sbatch --gres=gpu:$t:1 ... $SELF --gpu-only"
      sbatch --parsable -p "$PARTITION" -J "probe-$t" -t 0-00:10:00 -c 1 --mem=8G --gres="gpu:$t:1" \
             -o "$SLURM_SUBMIT_DIR/probe_${t}_%j.out" "$SELF" --gpu-only 2>&1 || echo "FAILED: submit for $t"
    done
    echo "Per-type results: ls $SLURM_SUBMIT_DIR/probe_*_*.out  (queued separately; may take a while)"
  fi
fi

section "DONE"
echo "Finished: $(date)"
echo "View this log with:  cat ${SLURM_SUBMIT_DIR:-$PWD}/probe_${SLURM_JOB_ID}.out"
