# Benchmarks

Everything here was run on one machine, so these numbers say how the three systems behave relative to each other on one box, not how they'd do on a real network. The raw results are in `benchmarks/results`, and `benchmarks/run_all.py` and `benchmarks/fedtally/crash.py` reproduce them.

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
| 64 | fedtally-masked | 1.8 | 166 | 154 | 1,088 |
| 64 | fedtally-plain | 1.3 | 133 | 165 | 960 |
| 64 | Flower | 5.1 | 1,990 | 1,041 | not measured |
| 64 | NVFlare | 2.7 | 1,008 | 558 | not measured |
| 256 | fedtally-masked | 4.6 | 187 | 561 | 4,352 |
| 256 | fedtally-plain | 3.2 | 146 | 552 | 3,840 |
| 256 | Flower | 6.4 | 4,111 | 2,356 | not measured |
| 256 | NVFlare | 4.0 | 4,477 | 2,050 | not measured |
| 1024 | fedtally-masked | 19.0 | 227 | 1,913 | 17,409 |
| 1024 | fedtally-plain | 15.7 | 166 | 2,144 | 15,361 |
| 1024 | Flower | 16.5 | 12,953 | 6,466 | not measured |
| 1024 | NVFlare | 7.7 | 13,516 | 5,882 | not measured |

## By site count, 256 MiB

| Sites | System | s per round | Server peak MiB | Peak per site MiB | Server disk MiB |
|---|---|---|---|---|---|
| 2 | fedtally-masked | 3.7 | 112 | 549 | 3,328 |
| 2 | fedtally-plain | 2.5 | 102 | 413 | 2,816 |
| 2 | Flower | 5.8 | 3,295 | 2,300 | not measured |
| 2 | NVFlare | 3.0 | 2,780 | 2,038 | not measured |
| 8 | fedtally-masked | 10.7 | 292 | 594 | 6,400 |
| 8 | fedtally-plain | 8.7 | 266 | 554 | 5,888 |
| 8 | Flower | 9.2 | 6,070 | 2,365 | not measured |
| 8 | NVFlare | 4.9 | 5,527 | 2,030 | not measured |
| 16 | fedtally-masked | 24.6 | 580 | 589 | 10,497 |
| 16 | fedtally-plain | 24.9 | 646 | 591 | 9,985 |
| 16 | Flower | 15.5 | 10,542 | 2,386 | not measured |
| 16 | NVFlare | 6.4 | 7,386 | 2,175 | not measured |

## What this shows

**fedtally uses far less server memory.** At 1 GiB it peaks around 230 MiB, where the others use about 13 GiB, and the figure stays roughly flat as the model grows. It does grow with sites, at 25 to 35 MiB per connection. Sites use about a third of the memory.

**That memory is paid for in disk.** The aggregator spools every vector, so it needs about `2 * model * (sites + 4)` free during a round: 17 GiB for this 1 GiB, 4-site run.

**NVFlare is the fastest at scale.** At 1 GiB it's 2 to 2.5 times faster per round, and at 16 sites about 4 times, though see the note above on those rows. fedtally's cost comes from what it does: 8 bytes per coordinate instead of 4, disk writes with syncs, fingerprints, and an extra download per audit. Masking and auditing add 20 to 50% over plain up to 8 sites. At 16 sites, on 14 cores, the difference is lost in the spread.

**Flower, running without TLS here, is about level with plain fedtally** at 1 GiB. It's slower at small sizes, where its polling intervals dominate. At 64 and 256 MiB masked fedtally is also ahead of Flower, and at 64 MiB ahead of NVFlare.

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

## FLamby

`benchmarks/flamby/run.py` runs the two tabular FLamby tasks through a real federation, with FLamby's centres and train and test splits. Data is fetched at run time from pinned sources and checked by hash. Five noise repeats per cell, 100 gradient steps in all.

| Task | Sites | Pooled | Federated | epsilon = 8 | epsilon = 3 | epsilon = 1 |
|---|---|---|---|---|---|---|
| Fed-TCGA-BRCA, C-index, delta = 0.068 | 6 | 0.849 | 0.848 | 0.804 +/- 0.049 | 0.774 +/- 0.036 | 0.731 +/- 0.083 |
| Fed-TCGA-BRCA, C-index, delta = 1e-5 | 6 | 0.849 | 0.848 | 0.776 +/- 0.056 | 0.712 +/- 0.078 | 0.628 +/- 0.105 |
| Fed-Heart-Disease, AUC, delta = 0.125 | 4 | 0.802 | 0.805 | 0.790 +/- 0.012 | 0.762 +/- 0.037 | 0.702 +/- 0.044 |
| Fed-Heart-Disease, AUC, delta = 1e-5 | 4 | 0.802 | 0.805 | 0.744 +/- 0.022 | 0.687 +/- 0.105 | 0.562 +/- 0.090 |

- **The private columns are flattered.** Clip and rounds were picked per cell on the test metric, among six settings. The winning setting changed between two full runs, and means moved by up to 0.03.
- **delta = 1 / sites^1.5 means little here.** It is 0.068 and 0.125 at 6 and 4 sites, so the delta = 1e-5 rows are the ones to read.
- **Heart-Disease is normalised per centre**, which a federation can do. Pooled training with pooled statistics reaches 0.881, because per-centre normalisation erases each centre's base rate.
- Fed-Camelyon16 was not run. It needs a data use registration and hundreds of gigabytes.

## Attacks

Scripts are in `benchmarks/attacks`, on the polygenic risk task. Each result file says which numbers are measured on real runs, which are simulated in numpy from a real plan's terms, and which are arithmetic.

**Sybil identities.** 12 honest sites, 5 rounds, fakes that flip labels. With no admission policy, 2, 4 and 8 fakes take the AUC from 0.804 to 0.800, 0.793 and 0.750. With credentials required, none was admitted in any run and the AUC stayed at 0.804. A fake with no credentials, one with self issued credentials and one with credentials copied from an honest site were all refused, the last one whether it joined before or after the site it copied.

**Poisoning.** 16 sites, 5 rounds, epsilon = 8, 3 seeds. A label flipper trains with cases and controls swapped. A backdoor site adds a trigger and scales its update to replace the average. A dishonest client clips 4,096 times too far out. All columns are real runs.

| Attack | Dishonest sites | No privacy | Private, honest client | Private, dishonest client | Dishonest client, norm proofs |
|---|---|---|---|---|---|
| none | 0 | 0.804 | 0.783 | 0.773 | 0.775 |
| label flip, AUC | 1 | 0.803 | 0.774 | 0.678 | 0.773 |
| label flip, AUC | 2 | 0.801 | 0.761 | 0.440 | 0.773 |
| label flip, AUC | 4 | 0.795 | 0.737 | 0.237 | 0.769 |
| backdoor, AUC | 4 | 0.789 | 0.777 | 0.771 | 0.764 |
| backdoor, trigger rate | 1 | 0.99 | 0.59 | 1.00 | 0.49 |
| backdoor, trigger rate | 2 | 0.99 | 0.66 | 1.00 | 0.41 |
| backdoor, trigger rate | 4 | 1.00 | 0.89 | 1.00 | 0.38 |

The trigger rate is the share of test people called a case once the trigger is set, about 0.5 with no attack. Without norm proofs one dishonest client is enough to replace the model, privacy terms or not. With them, the dishonest sites fail the proof and are left out of every round. A dishonest trainer behind an honest client is held to the clip like everyone else, which contains a label flipper but lets four backdoor sites of 16 push the trigger rate to 0.89: the proof bounds how long an update is, not where it points. A numpy model of the private round agrees with the real runs to within 0.03.

**Free riding.** A site that returns the model unchanged, or noise, is paid the same equal share as everyone else: 0.125 at 8 sites. Nothing detects it. What it costs is accuracy: with 1, 2 and 4 of 8 sites returning the model unchanged, a private run at epsilon = 8 goes from 0.738 to 0.728, 0.710 and 0.680.

**Gradient inversion.** Against single record updates of the logistic model, in closed form.

| What the adversary holds | Relative error | Cosine with the record |
|---|---|---|
| The aggregator's files in a masked run (measured) | 1.7 | 0.003 |
| The aggregator's files with no helper (measured) | 1e-7 | 1.000 |
| One update with every mask removed, noise for epsilon = 1 (simulated) | 2.8 | 0.05 |
| The same, epsilon = 3 (simulated) | 2.8 | 0.11 |
| The same, epsilon = 8 (simulated) | 2.4 | 0.22 |

The masked bytes pass a uniformity test (chi-square 283 on 255 degrees of freedom). The last three rows are the fallback when every mask is lost. Over five rounds at 8 sites that fallback is worth epsilon of 4.5, 13.9 and 40.8, so it blunts reconstruction and is no strong guarantee on its own.

**Privacy audit.** Canary sites in the manner of Steinke, Nasr and Jagielski (2023): each joins, gets an update or none by a coin flip, and is guessed from the released model. Lower bounds on epsilon at 95% confidence:

| epsilon the log charged | 64 canaries, real run | 2,000 canaries, simulated | 2,000 canaries, no noise |
|---|---|---|---|
| 1 | 0.00 | 0.30 | 6.0 |
| 3 | 0.16 | 1.16 | 5.3 |
| 8 | 1.92 | 2.79 | 5.3 |

No lower bound is above what the log charged. With 2,000 canaries the method tops out near 6, so it could have caught a wrong claim of 1 or 3 and not one of 8.

## Norm proofs

`cargo run --release -p fedtally-core --example norm_cost`, exact bound, one helper:

| Coordinates | Proof | One party's message | Prove | Aggregator's check | Helper's check |
|---|---|---|---|---|---|
| 65,536 | 50 kB | 4 kB | 0.012 s | 0.008 s | 0.007 s |
| 1,048,576 | 69 kB | 16 kB | 0.072 s | 0.055 s | 0.057 s |
| 16,777,216 | 143 kB | 66 kB | 0.157 s | 0.123 s | 0.128 s |
| 67,108,864 | 242 kB | 132 kB | 0.449 s | 0.346 s | 0.360 s |

The proof is under 1% of the update at a million coordinates. Above the exact range only the interval tests remain, which cost the same pass over the vector, and the proof is tens of kilobytes whatever the model size. 

What the protocol's options add to a round, 4 sites, 64 MiB, medians of 3:

| Plan | s per round |
|---|---|
| Masked, no privacy | 1.8 |
| Private, epsilon = 8 | 3.2 |
| Private, with norm proofs | 4.8 |
| Private, with scored groups of 2 | 4.0 |

Drawing noise is most of the step from masked to private. This model is past the exact range, so the proofs here are the interval tests alone. Scoring also adds a third to the aggregator's disk, 1,600 MiB against 1,088 MiB.

## Hot paths

`cargo run --release -p fedtally-core --example hotpath -- 256` times the kernels a round spends its processor time in. On the M4 Pro, one 131,072 coordinate chunk on one thread unless noted, before and after the kernels were rewritten:

| Kernel | Before MiB/s | After MiB/s |
|---|---|---|
| Mask expansion and add | 1,320-1,350 | 2,230-2,470 |
| Fingerprint, one chunk | 1,200-1,310 | 12,300-13,900 |
| Merkle root of a transferred chunk | 2,540 | 8,100-10,000 |
| Fingerprint of a file, all cores | 8,700-10,100 | 48,000-63,000 |
| BLAKE3 of a model file | 1,880 | 7,400-7,900 |

Output is byte for byte what it was. A round is bound by synced disk writes and single connection transfers, so most of this doesn't reach the clock. Running the old and new builds in turn under the same conditions, a masked round at 4 sites goes from 6.8 to 5.9 s at 256 MiB (medians of 3) and from 28.8 to 27.5 s at 1 GiB, and at 16 sites it's unchanged, 27.5 and 27.3 s.

## On the ledger

Compute units per instruction are in [ledger/README.md](../ledger/README.md), measured in LiteSVM. A round with a fixed cohort of 16 takes 20 transactions: three from the aggregator at 3,035, 3,004 and 957 units, an acknowledgement of 750 from each site, and 911 to settle. That is about 20,000 units and, at 5,000 lamports a signature, 100,000 lamports in base fees. Each round also creates one account of 780 bytes, which locks about 0.0046 SOL of rent that the program never returns.

Proving fraud at one leaf of a million-coordinate model with 16 sites takes 18 proof parts of 7,400 to 9,600 units each. A part is at most 1,824 bytes of leaf and path, which goes into the buffer over a few transactions before the one that checks it.

On a local validator (`cargo test -p fedtally-chain --test validator -- --ignored --nocapture`, 3 sites, 2 rounds, a challenge window of 25 slots), a round settles 28 to 31 slots after it opens. A round costs 55,000 lamports in base fees for eleven transactions, claims and closing the round included, and a priority fee of 1,000 microlamports a unit adds 140. Compute units on the validator match the figures above, apart from instructions that create an account and search for its address. When the aggregator posts a root for an aggregate that is not the sum, the watcher proves it in 19 transactions and 16 to 28 slots for 95,000 lamports, and takes the aggregator's stake.

A watcher that can't reach the aggregator gets what it needs through the ledger, with a window of 80 slots and 3 sites over 2 rounds. When the aggregator answers from its spool, 30 leaves come through, the run settles in about 110 s and the watcher pays 300,000 lamports in fees, plus 50,000 for each answer. When the aggregator has lost its vectors, it loses its stake about 70 s after the run is anchored, and the watcher ends with that stake less 150,000 lamports in fees.

Demanding a leaf takes 3,013 units, answering 3,328 and closing 1,196. Closing a paid out round takes 751 and returns its 0.0046 SOL of rent.

None of this was run on a public cluster.
