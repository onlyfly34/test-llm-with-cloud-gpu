# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Not an application — an operational recipe for standing up GLM-5.2 NVFP4 on
GPUhub/AutoDL RTX PRO 6000 Blackwell (sm_120) boxes. Three shell scripts plus a
pinned `requirements-frozen.txt`. There is nothing to build, lint, or unit test.

Claude Code normally runs **on the rented GPU instance itself**
(`root@gpuhub-container-*`), executing these scripts against live hardware. Edits
to the scripts and the machine they configure are the same act.

## The three scripts

```bash
bash rebuild-env.sh 2>&1 | tee rebuild.log   # ~20-30 min, mostly downloads
bash verify-weights.sh                        # single-GPU, run before upgrading to 8
bash sanitize-for-image.sh --dry-run          # before publishing an image
```

`rebuild-env.sh` is idempotent — every mutation is guarded, so re-running after a
failure resumes rather than corrupts. It ends with its own verification block
(torch/vllm versions, `has_flashinfer`, NCCL load-by-name, sm_120).

`verify-weights.sh` exists to keep expensive failures off the 8-GPU box
($7.30/hr vs $0.91/hr). Steps 1-3 check weight integrity and need no vLLM;
steps 4-5 need the environment. It is safe to run before `rebuild-env.sh`
succeeds — the weight half still reports.

`sanitize-for-image.sh` **deletes the credentials of any Claude Code session that
runs it**. Never invoke it from inside Claude Code; tell the user to run it from a
plain shell.

## Constraints that come from the machine, not preference

**Billed per second.** Do not idle-poll. Steps needing the web console (shutdown,
save image, change configuration, disk expansion) are hard stops — say so
explicitly and let the user act.

**Two disks, different fates.** `/` (30GB) is captured into a saved image;
`/root/autodl-tmp` (expandable) is bound to the physical host and is not. Weights,
HF cache, and pip cache go on the data disk. Anything meant to survive a VM change
goes on the system disk — which also means anything secret placed there gets baked
into the image.

**Long commands belong in `screen`** (`dl` / `serve` / `bench`). SSH drops here are
routine; a detached screen survives them, and so does the work. Claude Code's own
watchers do not — re-arm them after a reconnect and check the screens directly.

**`pkill -f "vllm serve"` matches the shell running it** and kills its own caller.
Record the PID instead. This has bitten twice.

## Environment landmines

The README documents these in full; the ones that change how you should act:

- **Do not use `uv`** — its bundled DNS resolver ignores `/etc/gai.conf`, so it
  keeps hitting the broken IPv6 on these hosts. Measured 0.2 MB/s vs pip's 11 MB/s.
- **Installing torch must not add `--extra-index-url pypi.org`** — pip would then
  pull 8GB of `nvidia-*` CUDA libs from whichever index has the higher version,
  which can be the slow one. Per-source bandwidth varies enormously between
  machines in the same region.
- **`~/.bashrc` line 6 returns early for non-interactive shells**, so exports at the
  end of the file are invisible to anything launched from a script or screen. Set
  `HF_HOME` explicitly in download commands; `rebuild-env.sh` writes the durable
  copies to `/etc/profile.d/` and `/etc/environment`.
- **flashinfer JIT-compiles sm_120 kernels on first serve** — roughly ten minutes,
  and it looks exactly like a hang. Its cache under `/root/.cache/flashinfer` is on
  the system disk, so a saved image skips the wait. Do not clean it.
- **Never download weights while installing packages.** Contention starves pip into
  read timeouts, which leave truncated wheels in the cache that resurface later as
  hash mismatches. Recovery is `pip cache purge`.

## Verifying a download

Xet buffers whole ~9GB shards in memory before writing, so a short `du` sample
reads as 0 MB/s while the network is saturated — check the network counter or the
`.incomplete` files, not directory size. Interrupted runs leave orphan
`.incomplete` files behind forever; distinguish them by mtime, since counting them
makes a finished download look unfinished.

## Commit conventions

Commits carry `Co-Authored-By` but **must not carry a `Claude-Session:` trailer** —
history was rewritten once to remove them. Commit directly to `main`; this is a
solo repo and its whole history is linear.
