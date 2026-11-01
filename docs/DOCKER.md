# Docker's role in this project

## Did the repair work require Docker?

**No.** The September 10 repair pass did not build a Docker image or start a real
container. The 72-test suite ran locally. REPL execution tests used local Python
subprocesses; the Docker transport regression mocked Docker commands and executed
the supplied Python locally. That checks script transport and output handling,
but does not validate container isolation, Docker memory limits, or the actual
in-container timeout command.

The training-library import check used a temporary Python virtual environment.
GPU dependency resolution also ran without Docker. Neither check exercised CUDA
training or established GPU memory requirements.

## Why does the project support Docker?

The model generates executable Python actions such as `search(...)` and `read(...)`.
The REPL runs those actions and returns their output to the model. Docker provides
a separate execution environment for that generated code.

In the current implementation:

- The historical `DockerREPL` creates one container per episode, with networking
  disabled and a default 512 MB memory limit. Its exact source is pinned by
  archived experiment manifests and was not changed after those runs.
- New arXiv MCP readers use `HardenedDockerREPL`, which adds one CPU, a 64-process
  limit, dropped capabilities, no privilege escalation, and a read-only root
  filesystem. A bounded 16 MB `/tmp` tmpfs holds the current step script.
- The root `Dockerfile` uses a non-root `sandbox` user and copies the document
  corpus into the image. This REPL configuration does not mount the host workspace
  or forward the host's full environment into the container.
- Each action runs through an in-container timeout; the container is removed when
  the episode closes.

These controls reduce exposure of the host to generated code. They are not proof
that the current configuration is sufficient for a public arbitrary-code service.

## Is Docker required for GPU training?

**The repository does not require Docker to run its training scripts.** Model
generation and gradient updates run in the trainer's Python process on the GPU
machine. The REPL container runs document-exploration code and is not configured
with GPU access. A provider may separately use a container to supply the GPU
machine's software environment; that is distinct from this per-episode sandbox.

`PersistentREPL` selects its backend as follows:

| Setting | Behavior |
| --- | --- |
| `use_docker=None` (automatic) | Uses Docker if `docker image inspect rlm-sandbox` succeeds; otherwise uses local Python. |
| `use_docker=False` | Always uses local Python. |
| `use_docker=True` | Requires Docker; startup failures surface as errors. |

The local fallback inherits the parent process's environment and working directory.
Generated actions therefore have that process's access to files, credentials, and
networking. For controlled development, use a disposable environment containing
only the data and access the run needs. A checkpoint being your own does not make
its generated code inherently safe. A public replay demo can display saved
trajectories without executing new model-generated code.

Installing the Python `docker` package from the requirements does **not** install
or start the Docker engine, provide the `docker` executable, or build `rlm-sandbox`.
The current REPL invokes the Docker CLI through subprocesses.

## What remains to validate?

On October 31, 2026, a local real-container smoke of the hardened backend
confirmed startup, two-step state, root-filesystem write rejection, `/tmp`
scratch writes, and cleanup. It did not validate every resource limit under
adversarial load. The archived case-study runs used the historical backend.
Rebuild the root image when its baked-in data changes; benchmark runs instead
use a mounted frozen corpus.

Docker reconstructs state by replaying previous successful actions, so earlier
actions run again at every step. This grows work with episode length and cannot
undo side effects. The local worker keeps state in one process instead.

Implementation: [historical REPL backends](../src/env/repl.py),
[hardened backend](../src/env/hardened_repl.py), [sandbox image](../Dockerfile),
[REPL tests](../tests/test_repl.py). Training gates and verification results are in
the [GPU readiness report](GPU_TRAINING_READINESS.md).
