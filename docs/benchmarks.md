# Benchmarks

Everything here was run on one machine on 5 November 2026, so these numbers say how the three systems behave relative to each other on one box, not how they'd do on a real network. The raw results are in `benchmarks/results`, and `benchmarks/run_all.py` and `benchmarks/fedtally/crash.py` reproduce them.

**Versions.** Flower 1.43.0, installed from PyPI, and its code matches commit `91e6c680b`. NVFlare built from commit `3083649b3`, reporting 2.10.0dev0+170. Both run from a Python 3.11 virtualenv, on stock settings unless noted.

## Workload

`benchmarks/common.py` is shared by all three harnesses. A model is 8 float32 tensors of seeded random values. "Training" multiplies each tensor by `1 + 0.001 * (site + 1)`, which costs one model's worth of arithmetic and memory, so these runs measure the framework and not a trainer. Each harness checks the final model against the known answer and fails otherwise.

The first version of this workload used constant tensors. Flower's content-addressed chunks then moved 1/8 of the bytes, which made the comparison meaningless. With random values its traffic matches the nominal size to within 0.01%.

## Method

- Each system runs as separate processes: aggregator and helper, SuperLink and SuperNodes, or a POC server and clients.
- Per-round time is `(T3 - T1) / 2`, from runs of 3 rounds and 1 round, which cancels start-up and shutdown. Medians over 3 repeats for the 64 and 256 MiB runs at 4 sites, and a single run for the others, so treat those more loosely.
- Memory is the peak of summed resident memory per process tree, sampled at up to 20 Hz. Shared pages count more than once, which affects everyone.
- Server means fedtally's aggregator plus helper, Flower's SuperLink, SuperExec and ServerApp, and NVFlare's server parent and job process.
- Disk is fedtally's aggregator directory at its peak. Disk use wasn't measured for the others.

**This isn't like for like on security.** `fedtally-masked` has TLS, masking, a helper, fingerprints and audits. `fedtally-plain` has TLS and fingerprints, and the aggregator reads updates. NVFlare has mTLS and the server reads updates. Flower has `--insecure`, no TLS, and the server reads updates.

## By model size, 4 sites

| Model MiB | System | s per round | Server peak MiB | Peak per site MiB | Server disk MiB |
|---|---|---|---|---|---|
| 64 | fedtally-masked | 2.3 | 159 | 215 | 1,088 |
| 64 | fedtally-plain | 1.5 | 123 | 177 | 960 |
| 64 | Flower | 5.1 | 1,990 | 1,041 | not measured |
| 64 | NVFlare | 2.7 | 1,008 | 558 | not measured |
| 256 | fedtally-masked | 6.5 | 178 | 527 | 4,352 |
| 256 | fedtally-plain | 3.3 | 141 | 512 | 3,840 |
| 256 | Flower | 6.4 | 4,111 | 2,356 | not measured |
| 256 | NVFlare | 4.0 | 4,477 | 2,050 | not measured |
| 1024 | fedtally-masked | 25.6 | 182 | 1,690 | 17,409 |
| 1024 | fedtally-plain | 14.8 | 169 | 2,085 | 15,361 |
| 1024 | Flower | 16.5 | 12,953 | 6,466 | not measured |
| 1024 | NVFlare | 7.7 | 13,516 | 5,882 | not measured |

## By site count, 256 MiB

| Sites | System | s per round | Server peak MiB | Peak per site MiB | Server disk MiB |
|---|---|---|---|---|---|
| 2 | fedtally-masked | 5.7 | 137 | 371 | 3,328 |
| 2 | fedtally-plain | 2.9 | 98 | 575 | 2,816 |
| 2 | Flower | 5.8 | 3,295 | 2,300 | not measured |
| 2 | NVFlare | 3.0 | 2,780 | 2,038 | not measured |
| 8 | fedtally-masked | 11.0 | 264 | 601 | 6,400 |
| 8 | fedtally-plain | 6.6 | 253 | 454 | 5,888 |
| 8 | Flower | 9.2 | 6,070 | 2,365 | not measured |
| 8 | NVFlare | 4.9 | 5,527 | 2,030 | not measured |
| 16 | fedtally-masked | 22.2 | 594 | 441 | 10,497 |
| 16 | fedtally-plain | 13.3 | 478 | 539 | 9,985 |
| 16 | Flower | 15.5 | 10,542 | 2,386 | not measured |
| 16 | NVFlare | 6.4 | 7,386 | 2,175 | not measured |

## What this shows

**fedtally uses far less server memory.** At 1 GiB it peaks around 180 MiB, where the others use about 13 GiB, and the figure stays roughly flat as the model grows. It does grow with sites, at 25 to 35 MiB per connection. Sites use about a third of the memory.

**That memory is paid for in disk.** The aggregator spools every vector, so it needs about `2 * model * (sites + 4)` free during a round: 17 GiB for this 1 GiB, 4-site run.

**NVFlare is the fastest at scale.** At 1 GiB it's 2 to 3 times faster per round, and at 16 sites 2 to 3.5 times. fedtally's cost comes from what it does: 8 bytes per coordinate instead of 4, disk writes with syncs, fingerprints, and an extra download per audit. Masking and auditing add 50-100% over plain.

**Flower, running without TLS here, is about level with plain fedtally** at 1 GiB and at 16 sites. It's slower at small sizes, where its polling intervals dominate. At 64 MiB fedtally is also ahead of NVFlare.

These runs don't include real training, which would dominate per-round time in most real work, or a real network, which might change the order.

## When the server is killed

SIGKILL during round 2 of a 4-round run, with 4 sites and a 32 MiB model, then a restart. The Flower and NVFlare scenarios are in `benchmarks/flower/crash.py` and `benchmarks/nvflare/crash.py`.

| System and setup | Killed | Outage | Run finished? | Afterwards |
|---|---|---|---|---|
| fedtally | Aggregator | 2 s (*3), 30 s, 90 s, 300 s | **Yes**, 6 of 6 | Log complete, right model, no site trained twice |
| fedtally | Aggregator and helper | 2 s, 90 s | **Yes**, 2 of 2 | Same |
| Flower, default flags | SuperLink | 2 s | No | Run gone, SuperNodes exit, resubmit from round 1 |
| Flower, `--database` | SuperLink | 0-10 s | No | `finished:failed`, resubmit from round 1 |
| Flower, `--isolation process --database` | SuperLink | 2 s (*3), 30 s | **Yes**, 4 of 4 | Round 2 continued, right model |
| Flower, `--isolation process --database` | SuperLink | 90 s (*2) | No | `No heartbeat received from the task` |
| Flower, `--isolation process --database` | SuperLink, SuperExec, ServerApp | 2 s | No | `No heartbeat received from the task` |
| NVFlare, POC defaults | Server parent, and with its job process | About 6 s | No, 4 of 4 trials | `RUNNING` for at least 3m45s with no job process. Resubmitting from the start or the last saved model works |

Flower's process-isolation setup does recover, as long as the ServerApp survives the crash and the SuperLink returns within a heartbeat window, which the code suggests is 60 s. That's a real capability, and fedtally's advantage over it is narrower than over the default. fedtally's limit on an outage is a setting on the sites, `give_up_secs`, 10 minutes by default and tested to 300 s, and it has no separate process whose loss ends the run.

**Not tested:** real power loss, since SIGKILL leaves the page cache intact. fedtally syncs its log and uploads, but that hasn't been exercised on a machine losing power. Also not tested: Docker or multi-host deployments, or a network partition.

## Accuracy under privacy

`benchmarks/privacy/prs.py` trains a polygenic risk score on synthetic genotypes: 200 variants, 40 of them causal, 1,500 people per site, and logistic regression by FedAvg for 10 rounds through a real federation. The score is AUC on 6,000 held out people. Privacy is at the level of a site, with `delta = 1 / sites^1.5` and a clip of 0.2. Means over 3 seeds, from `benchmarks/results/privacy.json`.

| Sites | delta | Pooled | Federated | epsilon = 8 | epsilon = 3 | epsilon = 1 |
|---|---|---|---|---|---|---|
| 8 | 0.044 | 0.803 | 0.803 | 0.743 | 0.641 | 0.563 |
| 16 | 0.016 | 0.805 | 0.804 | 0.779 | 0.723 | 0.604 |
| 32 | 0.0055 | 0.805 | 0.805 | 0.797 | 0.776 | 0.667 |
| 64 | 0.0020 | 0.813 | 0.813 | 0.808 | 0.799 | 0.760 |

**Without privacy, federating costs nothing here.** The federated model matches one trained on the pooled data to three decimals.

**Site-level privacy is expensive for small federations.** The noise has to hide a whole site, so its share of the sum falls only as sites are added. At 8 sites and epsilon = 1 the model is little better than chance. At 64 sites epsilon = 3 costs 0.014 of AUC.

**The budget spent is below the budget asked for.** Noise is a whole number of grid units, so it rounds up. The runs asking for epsilon of 1, 3 and 8 spent about 0.83, 2.5 and 6.5.

This is one synthetic task with a linear model of 201 parameters. It says nothing about deep models, where clipping bites harder, and three seeds give a rough mean and no interval.

## On the ledger

Compute units per instruction are in [ledger/README.md](../ledger/README.md), measured in LiteSVM. A round with a fixed cohort of 16 takes 20 transactions: three from the aggregator at 3,035, 3,004 and 957 units, an acknowledgement of 750 from each site, and 911 to settle. That is about 20,000 units and, at 5,000 lamports a signature, 100,000 lamports in base fees. Each round also creates one account of 772 bytes, which locks about 0.0046 SOL of rent that the program never returns.

Proving fraud at one leaf of a million-coordinate model with 16 sites takes 18 proof parts of 7,400 to 9,600 units each. A part is at most 1,824 bytes of leaf and path, which goes into the buffer over a few transactions before the one that checks it.

None of this was run on a live cluster.
