# Design

## Roles

| Role | Holds | Must be trusted for |
|---|---|---|
| **Site** | Its data, its key, its trainer | Its own contribution |
| **Aggregator** | Masked vectors, the log | Staying up. Not secrecy, not correctness |
| **Helper** | Sealed seeds it can open | Secrecy, jointly: an update stays secret if at least one helper doesn't collude with the aggregator or with the site assigned to audit it |
| **Sponsor** | A signed plan, delegated authority | Nothing during the run |

Aggregator and helpers are Rust binaries. They run no Python and accept no code from anyone. Each site runs the trainer it was started with.

## The log is the only control state

A run is a sequence of signed records, one JSON object per line, chained by hash. The aggregator orders them and broadcasts them. Every party keeps a replica and folds it through the same pure function, `Replay::apply` in `crates/fedtally-core/src/replay.rs`, which either applies a record whole or refuses it with the state unchanged. Each record has exactly one valid encoding.

There are no other control messages. Each role looks at the state and does what's next:

```
Phase       aggregator          helper             site
Joining     open when ready                        join
Collecting  close when ready                       train, contribute, upload
Masking     outcome when ready  sum masks, upload
Attesting   seal when ready     attest             check, attest
Acking      next round                             check, acknowledge
```

Consequences:

- **A restart is a replay.** All state is the log and the spool. Kill any process and the rest reconnect, and it works out what to do next from the log. Derived files are rebuilt deterministically and checked against the digests already in the log.
- **An auditor is a site with no data.** `fedtally verify` runs the same rules as the protocol.
- **A round finishes with every included site's signature on one head, or the run stops.** A later round can't open until that's true, so a site shown a forked log never sees it continue.

## One round

Let `x_i` be site `i`'s fixed-point update, weighted, with two random pad coordinates on the end.

1. **Contribute.** The site draws one seed per helper and computes `y_i = x_i + sum_h PRG(s_ih)` in the Goldilocks field. It signs a contribution naming the digest of `y_i`, its weight, and the digest of each sealed seed, then uploads `y_i` and the seeds, each sealed to its helper.
2. **Close.** The aggregator picks the complete contributions, sums them to `Y`, and signs the cohort and the digest of `Y`. The cohort must meet `min_cohort`.
3. **Mask sum.** Each helper checks the log, opens the seeds for exactly that cohort, computes `M_h = sum_i PRG(s_ih)`, uploads it and signs its digest. A helper does this once per round: there's one valid Close, and its replica proves it.
4. **Outcome.** The aggregator computes `A = Y - sum_h M_h`, which is the plain weighted sum. It builds the next model and signs the digests of both. **The log head after this record is the challenge.** Everything has now been committed.
5. **Attest.** From the challenge, everyone derives a point `rho` in F_p^2. Define `F(v) = sum_k v_k rho^k`.
   - Each helper signs `F(M_h)` and `F(m_ih)` for each site.
   - Each site then downloads `A`, rebuilds the model and compares digests. It computes `F(m_ih)` from its own seeds and compares with what the helper signed. It re-checks one other site's vector (below). If all of that holds, it signs `F(y_i)`.
6. **Seal.** The aggregator signs `F(y_i)` for every site, `F(Y)` and `F(A)`. The replay rejects any seal unless:
   - every included site has attested, and its value is the one sealed
   - `sum_i F(y_i) = F(Y)`
   - `sum_i F(m_ih) = F(M_h)` for each helper
   - `F(A) = F(Y) - sum_h F(M_h)`
7. **Acknowledge.** Each site compares `F(A)` from the seal with what it computed over the bytes it received, its own `F(y_i)`, and the audited site's. If they match, it signs the head.

### Why this proves the sum

For an honest site, `F(y_i)` and `F(m_ih)` are right because it computed them itself, over vectors committed before `rho` existed. So in the sealed equation, everything is the true fingerprint of a committed vector, and `F(A_received) = F(sum_i x_i)`. `F` is a polynomial of degree at most `d` in `rho`, so two different vectors collide with probability at most `d / p^2`, below `2^-90` for a billion coordinates (Schwartz-Zippel). The aggregator influences `rho` only by changing what it commits to, and grinding `2^40` candidates still leaves a negligible chance.

This costs a few field multiplications per coordinate, in parallel over chunks. It needs no zero-knowledge proof, trusted hardware or extra cryptographic assumption.

### Why every site must attest

An earlier version let the aggregator speak for a silent site. That was broken: with one fingerprint unbacked, an aggregator can solve for its value and make the equations hold for any aggregate. So now no site can be spoken for.

### Why sites audit each other

A site colluding with the aggregator could attest to the fingerprint of something other than what it committed to, choosing that value after the true sum is known. So each site also downloads another site's masked vector, checks it against the signed digest, and compares its fingerprint with the seal. The assignment comes from the challenge, and nobody audits themselves, so an aggregator with one site on its side always meets an honest auditor. A masked vector is noise to the site reading it, and the aggregator serves it only to the assigned auditor. `audits` in the plan sets how many each site checks, at one extra model download each. A default grant refuses a masked plan with no audits.

## Secrecy

The aggregator holds `y_i` and `Y`, and learns `A`. Without a seed `s_ih` for at least one honest helper, `y_i` is a one-time pad under ChaCha12. Each helper holds seeds and the `M_h` it computes itself, but no vectors. A plan carries each helper's signature over its sealing key, so a sponsor can't name an honest helper and pair it with someone else's key.

**Fingerprints reveal nothing.** `F(y_i) - sum_h F(m_ih) = F(x_i)` is public, so `F(x_i)` could leak. The pad prevents it: the last two coordinates of `x_i` are uniform and fresh each round, and `b1*rho^d + b2*rho^(d+1) = rho^d (b1 + b2*rho)` is a bijection of F_p^2 whenever `rho not in F_p`, which `challenge()` guarantees. So `F(x_i)` is uniformly distributed whatever the update is. One pad only hides one evaluation, so sites never publish `F(m_ih)` themselves, and each helper signs one set of shares per round. An aggregator that forks the log can't collect a second evaluation. The pad coordinates of `A` are discarded.

**Who is in the round matters.** Every party learns `A`, so a sum over a small cohort says a lot about each member. The aggregator chooses the cohort, so with `k` sites on its side it can isolate one honest site's update if the cohort can be as small as `k + 1`. The default `min_cohort` is three quarters of the sites. A site can demand more in its grant, and can list who it will be averaged with, so a sponsor can't meet the minimum with parties of its own.

## What each adversary can do

| Adversary | Cannot | Can |
|---|---|---|
| Aggregator alone | Read an update. Get an acknowledgement for a model that isn't the committed sum | Stop or stall the run. Leave sites out, down to `min_cohort`. If it refuses to log a site's record, that site knows, and the report lists the site as absent, but nothing proves why |
| Helper alone | Read an update. Change the sum: its share claims are checked by each site, and a changed total breaks `F(A)` | Stop the run |
| Site alone | Read another site's update. Spoil a seal by lying: the aggregator refuses the attestation | Send any update it likes, since a masked vector can't be range-checked. Stop the run, by not attesting or by sealing a seed that won't open |
| Aggregator + 1 site | Change the sum unnoticed, with `audits >= 1`: its auditor is honest | Read a victim's update if `min_cohort` is 2 |
| Aggregator + `k` sites | Read an honest site's update if `min_cohort > k + 1`. Change the sum unnoticed if `k <= audits` | With `k > audits`, have its sites audit each other by grinding the challenge. Raise `audits` towards the site count to prevent this |
| Site + every helper | Read a vector it isn't assigned to audit | Read the update of the site it audits, without the aggregator |
| Aggregator + site + helper | | Replace the model with anything, unnoticed. The helper lies about that site's mask share, and only those two hold the seed. A share lie is no more than the site could do alone unless the aggregator is in on it too, and then the three of them can pick the result after seeing the true sum |
| Aggregator + every helper | | Read everything. Masking rests on this not happening |

Without helpers the aggregator reads every update, and no audit runs because sites must not read each other's plain updates. A site's default grant refuses such a plan.

## Dropouts

A site that drops before contributing is simply not in the Close. There's no secret-sharing round and no recovery phase, because masks only involve sites and helpers, not pairs of sites. This is where the design is simpler than SecAgg, which uses O(n^2) key agreement and Shamir shares.

**A site that drops after Close stops the run**, once the deadline passes. It can restart and resume from its disk within the deadline. If it's gone for good, the log names it and the consortium starts a new plan from the last sealed model. Redoing the round in place wouldn't be safe: the aggregator would hold two sums over nearly the same updates, whose difference is the missing site's update.

## Admission

A site's key is the root of authority. `fedtally grant` issues a UCAN delegation for `/fedtally/contribute` with conditions: purpose, trusted helpers, allowed sites, minimum cohort, maximum rounds, expiry. The sponsor exercises it with an invocation carrying the plan's digest and terms, and can re-delegate in between. `crates/fedtally-core/src/admission.rs` validates the chain with `pq-ucan`.

So if the sponsor picks a helper the site doesn't trust, the chain fails on the site's own machine, without anyone needing to read a contract. The CLI makes a site name its helpers and its minimum cohort, since a grant with neither would let a sponsor run its own helper and pick a cohort of two. Each `Join` record holds the tokens, so an auditor can re-check why each site took part. A site re-checks its grant before every round, so revoking stops its next contribution.

There's no CA to run. The aggregator's TLS certificate is pinned by digest in the signed plan, and each peer signs over a TLS exporter value to prove its key.

## Post-quantum parts

These matter where data is long-lived, such as genomes.

- Signatures: Ed25519 or ML-DSA-65 per party, through `pq-ucan`. A run can mix them.
- Sealed seeds: X25519 and ML-KEM-768 together, so breaking one isn't enough.
- TLS: connections settle on X25519MLKEM768, and a test confirms the negotiated group.
- Masks: ChaCha12 with 256 bit seeds.

## Data plane

A vector is a file of little-endian field elements, cut into fixed chunks. Each chunk has its own mask stream, so everything runs chunk by chunk in parallel: encoding, masking, summing, hashing and fingerprinting. Memory depends on chunk size, threads and connections, not on the model or the number of sites, and the disk takes the rest.

A chunk's digest is checked on arrival and written with `pwrite`, and an upload is synced before it can be counted in a round. A restart rehashes the spool to find what's missing, and peers resend only that.

## Limits this design accepts

- **Masked updates can't be inspected.** A site can send garbage of any size and nobody can tell. There's no robust aggregation and no range proof. Without helpers, the aggregator range-checks coordinates.
- **One party can stop a run.** Any site or helper can. This trades liveness for the guarantee that nobody takes a model some party couldn't check.
- **A helper's disk must survive.** "Once per round" rests on the helper's replica. If a helper loses it, a malicious aggregator can replay contributions and ask for a second mask total over a different cohort, whose difference isolates one site's mask, and for a second set of fingerprint shares. Don't run helpers on ephemeral storage.
- **All helpers are needed**, and the aggregator is one process. A restart resumes it, but there's no failover.
- **Fixed point costs bandwidth.** Coordinates are 8 bytes on the wire, twice float32, and precision is `2^-frac_bits`.
- **Timestamps are self-asserted.** The time in a `Join` is the site's word, and an auditor validates grants at that time.
- **Stalling isn't detected by sites.** A site waits as long as the aggregator stays connected. Sites can't tell a slow aggregator from a stalling one.
