# Design

fedtally implements the EkayanaFL protocol: federated learning for biomedical data where admission rests on credentials, updates are private, and anyone can check the aggregate.

## Roles

| Role | Holds | Must be trusted for |
|---|---|---|
| **Site** | Its data, its key, its credentials, its trainer | Its own contribution |
| **Aggregator** | Masked vectors, the log | Staying up. Not secrecy, not correctness |
| **Helper** | Sealed seeds it can open | Secrecy, jointly: an update stays secret if at least one helper doesn't collude with the aggregator or with the site assigned to audit it |
| **Sponsor** | A signed plan, delegated authority | Nothing during the run |
| **Issuer** | A key that signs credentials and revocation lists | The claims it makes about sites, datasets and operators |

The aggregator and the helpers together are the paper's aggregation committee. They are Rust binaries that run no Python and accept no code from anyone. Each site runs the trainer it was started with.

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

- **A restart is a replay.** All state is the log and the spool. Kill any process and the rest reconnect, and it works out what to do next from the log.
- **An auditor is a site with no data.** `fedtally verify` runs the same rules as the protocol, offline.
- **A round finishes with every included site's signature on one head, or the run stops.** A later round can't open until that's true, so a site shown a forked log never sees it continue.
- **Every rule is integer arithmetic over the log.** The privacy budget, the cohort draw and credential checks all replay to the same answer on every machine. Anything that needs the outside world, such as a clock or a registry, is asked by each party for itself and never decides what a replay accepts.

## One round

Let `x_i` be site `i`'s update on the integer grid, with two random pad coordinates on the end.

1. **Open.** The aggregator starts the round from the current model. If the plan samples, the round's invited sites are fixed here by the beacon (below). In a private run, the round only opens if the budget covers it.
2. **Contribute.** The site draws one seed per helper and computes `y_i = x_i + sum_h PRG(s_ih)` in the Goldilocks field. It signs a contribution naming the Merkle root of `y_i`, its weight, and the digest of each sealed seed, then uploads `y_i` and the seeds, each sealed to its helper.
3. **Close.** The aggregator picks the complete contributions, sums them to `Y`, and signs the cohort and the root of `Y`. The cohort must meet `min_cohort`. A private run pays for the round here.
4. **Mask sum.** Each helper checks the log, opens the seeds for exactly that cohort, computes `M_h = sum_i PRG(s_ih)`, uploads it and signs its root. A helper does this once per round.
5. **Outcome.** The aggregator computes `A = Y - sum_h M_h`, builds the next model and signs the roots of both. **The log head after this record is the challenge.** Everything has now been committed.
6. **Attest.** From the challenge, everyone derives a point `rho` in F_p^2. Define `F(v) = sum_k v_k rho^k`.
   - Each helper signs `F(M_h)` and `F(m_ih)` for each site.
   - Each site downloads `A`, rebuilds the model and compares digests. It computes `F(m_ih)` from its own seeds and compares with what the helper signed. It re-checks another site's vector (below). If all of that holds, it signs `F(y_i)`.
7. **Seal.** The aggregator signs `F(y_i)` for every site, `F(Y)` and `F(A)`. The replay rejects any seal unless every included site has attested the value sealed for it, `sum_i F(y_i) = F(Y)`, `sum_i F(m_ih) = F(M_h)` for each helper, and `F(A) = F(Y) - sum_h F(M_h)`.
8. **Acknowledge.** Each site compares `F(A)` from the seal with what it computed over the bytes it received, its own `F(y_i)`, and the audited site's. If they match, it signs the head.

### Why this proves the sum

For an honest site, `F(y_i)` and `F(m_ih)` are right because it computed them itself, over vectors committed before `rho` existed. So in the sealed equation everything is the true fingerprint of a committed vector, and `F(A_received) = F(sum_i x_i)`. `F` is a polynomial of degree at most `d` in `rho`, so two different vectors collide with probability at most `d / p^2`, below `2^-90` for a billion coordinates (Schwartz-Zippel).

This costs a few field multiplications per coordinate. It needs no zero-knowledge proof, trusted setup or trusted hardware.

### Why every site must attest, and why sites audit each other

If one fingerprint could be left to the aggregator, it could solve for that value and make the equations hold for any aggregate. So no site can be spoken for.

A site colluding with the aggregator could still attest to the fingerprint of something other than what it committed to. So each site also downloads another site's masked vector, checks it against the signed root, and compares its fingerprint with the seal. The assignment comes from the challenge, and nobody audits themselves. `audits` in the plan sets how many each site checks. A default grant refuses a masked plan with no audits.

## Commitments

A vector's root is a Merkle tree as RFC 6962 defines it, with SHA-256, over the vector's bytes in 1 KiB leaves. Chunks are whole subtrees, so hashing runs in parallel and a transfer is checked chunk by chunk.

A standard tree means one leaf can be shown to anyone who holds the root, with about twenty hashes for a million-coordinate model. `vector::first_wrong_leaf` finds the first leaf where an aggregate is not the sum of the committed vectors, and `vector::prove_leaf` proves each vector's leaf. That is a fraud proof a program on a ledger can check with nothing but SHA-256, which is what [the ledger](#the-ledger) does.

## Privacy

A plan may carry privacy terms. Then each site, before masking:

1. clips its update to an L2 norm of `clip`,
2. rounds each coordinate onto the grid at random, so rounding adds no bias,
3. adds noise from the discrete Gaussian with variance `noise` to every coordinate.

The noise in a sum is the noise of every site in it, so nobody has to trust the aggregator or a helper to add it. This is the distributed discrete Gaussian mechanism of Kairouz, Liu and Steinke.

**The guarantee is for a whole site's data.** Who took part in a round is public, so the guarantee is about what a site contributed, not whether it was there. Replacing all of one site's data with nothing changes a sum by at most `clip + sqrt(d) + 1` grid units, so a round over `k` sites, `c` of which are assumed to give their noise away, is `rho`-zCDP with `rho = (clip + sqrt(d) + 1)^2 / (2 * noise * (k - c))`.

**The budget is one integer.** `rho` adds up across rounds, so the log tracks a single number in units of `2^-40`, computed in integers, and every replica agrees on it. A round is paid for when it closes, since the sum is as good as released once the cohort is fixed, whether or not the round is ever sealed. A round opens only if the budget covers it at the smallest cohort the plan allows, which is the dearest case. `epsilon` at the plan's `delta` is worked out from `rho` for people, with the conversion of Canonne, Kamath and Steinke, and plays no part in the rules.

**What the code guarantees itself:**

- The sampler is exact. It uses integers only, because floating point Gaussian samplers leak through their rounding.
- The clip is enforced in integers. After rounding, the site checks the vector's norm against the bound the accounting uses, and rescales if floating point ever put it over.
- Every site counts the same. Weighting by example count would make the sensitivity depend on private data, so a private plan fixes `max_weight` at 1.
- A private plan needs a helper. Without one the aggregator would read each noisy update on its own, where only the much weaker per-site figure below holds.

**If every mask is removed later**, each site's update still carries its own noise: `rho = (clip + sqrt(d) + 1)^2 / (2 * noise)` per round. The report prints this figure. It is `k` times weaker than the masked one.

A site can make privacy a condition of joining: its grant can cap the run's whole budget and set the fewest colluding sites the accounting must assume.

## Admission

Two things must both hold before a site is in a run.

**The run admits the site.** A plan may carry a policy: a rule over credentials, built from `all`, `any` and `holds`. A `holds` names a credential type, the issuers accepted for it, whether it is about the site or its dataset, claims that must match, and optionally a claim that must name the site's dataset and a claim holding data use terms. The paper's example is `all` of an affiliation, an ethics approval, an access grant for the site and its dataset, and provenance for the dataset. A second rule can ask the same of the aggregator and each helper, whose credentials the plan lists.

A rule only asks for credentials, never for their absence. So adding one can't turn an admitted site away, and revoking one can't let a refused site in.

- **Credentials** are W3C Verifiable Credentials (Data Model 2.0) with an `eddsa-jcs-2022` proof. The code reproduces the W3C test vector for that suite byte for byte. A GA4GH visa is carried as its `ga4gh_visa_v1` object in the credential's subject.
- **Data use.** A dataset's terms are GA4GH Data Use Ontology codes. The plan states what the run will do with data, and `duo::permits` decides whether the terms allow it: a purpose fits the same permission or a broader one, never a narrower one. Anything the code can't decide is refused, since a person should look at it.
- **Revocation** is a W3C Bitstring Status List: one signed list of bits per issuer. A list is a record in the log, so a replay needs no network. A newer list may only add bits. Any party may bring one, and each site brings any newer list it holds before every round. Once a list reaches the log, a site whose credentials no longer meet the policy is out of every later cohort. If an operator's no longer do, the run stops.
- **Every replica makes the same decision**, at the time the site's Join states, against the lists in the log. A credential that can be revoked only counts once its issuer's list is in the log. Lists are kept per issuer, so one issuer can't take another's.
- **A dataset belongs to one site.** A second site naming the same dataset is refused, so credentials about a dataset can't be borrowed from another site's Join.
- **A site that holds a newer list than the log** offers it, and gives nothing to an aggregator that won't log it.
- **Time is a site's word, within limits.** A Join can't be dated before a list already in the log, and the aggregator refuses one more than ten minutes from its own clock.

**The site admits the run.** A site's key is the root of its own authority. `fedtally grant` issues a UCAN delegation for `/fedtally/contribute` with conditions: purpose, trusted helpers, allowed sites, minimum cohort, maximum rounds, audits, privacy budget, expiry. The sponsor exercises it with an invocation carrying the plan's digest and terms. If the sponsor picks a helper the site doesn't trust, the chain fails on the site's own machine. A site re-checks its grant before every round, so revoking stops its next contribution.

There's no CA to run. The aggregator's TLS certificate is pinned by digest in the signed plan, and each peer signs over a TLS exporter value to prove its key.

## Who is in a round

Every party learns `A`, so a sum over a small cohort says a lot about each member. The default `min_cohort` is three quarters of the sites, and a site can demand more in its grant.

A plan may **sample**: each round invites a fixed number of sites. The round's beacon is `SHA-256(plan, round, model the round starts from)`, each site's ticket is `SHA-256(beacon, site)`, and the sites with the smallest tickets are invited. Anyone can recompute the draw, only invited sites may contribute, and the report doesn't count the others as missing.

After the first round, the model a beacon is drawn from carries updates or noise from honest sites, which the aggregator cannot know before the round that produced it has closed, as long as one helper is honest. The first round's beacon is fixed by the plan alone, so whoever writes the plan can steer that one draw.

## Identities

A party is a `did:key`, or a `did:bio` holding the same Ed25519 key. A plan can't list one key twice under two names. A `did:bio` identifier is also a Solana address, so a party's name in the plan and its account on a ledger are the same key.

A registered `did:bio` document can drop its key or be deactivated. `registry::standing` reads a registry account and says whether the identifier's own key may still act. A site started with an endpoint it trusts asks this about every `did:bio` operator, site and issuer before each round, and sits the round out if one no longer stands, or if it cannot ask. The log never depends on the answer.

## Secrecy

The aggregator holds `y_i` and `Y`, and learns `A`. Without a seed `s_ih` for at least one honest helper, `y_i` is a one-time pad under ChaCha12. Each helper holds seeds and the `M_h` it computes itself, but no vectors. A plan carries each helper's signature over its sealing key.

**Fingerprints reveal nothing.** `F(y_i) - sum_h F(m_ih) = F(x_i)` is public. The last two coordinates of `x_i` are uniform and fresh each round, and `b1*rho^d + b2*rho^(d+1)` is a bijection of F_p^2 whenever `rho not in F_p`, which `challenge()` guarantees. So `F(x_i)` is uniformly distributed whatever the update is. One pad hides one evaluation, so sites never publish `F(m_ih)` themselves, and each helper signs one set of shares per round.

## What each adversary can do

| Adversary | Cannot | Can |
|---|---|---|
| Aggregator alone | Read an update. Get an acknowledgement for a model that isn't the committed sum. Spend past the privacy budget, or draw its own cohort after round 1 | Stop or stall the run. Leave invited sites out, down to `min_cohort`. Refuse to log a record: the site knows, and the report lists it as absent |
| Helper alone | Read an update. Change the sum | Stop the run |
| Site alone | Read another site's update. Spoil a seal by lying. Stay in after its credential is revoked and the list is logged | Send any update it likes, since a masked vector can't be range checked. Stop the run |
| Aggregator + 1 site | Change the sum unnoticed, with `audits >= 1` | Read a victim's update if `min_cohort` is 2 and the run has no privacy terms |
| Aggregator + `k` sites | Read an honest site's update if `min_cohort > k + 1`. Change the sum unnoticed if `k <= audits` | With `k > audits`, have its sites audit each other |
| Aggregator + site + helper | | Choose that site's input after seeing the true sum. With no bound on inputs this means any model |
| Aggregator + every helper | Learn more than the per-site privacy figure allows, in a private run | Read each noisy update. Without privacy terms, read everything |
| Issuer | Admit a site alone, where the policy asks for several issuers | Vouch falsely for what it issues. Revoke at any time |
| Sponsor | Change the plan after signing. Meet a site's cohort condition with parties the site didn't list | Steer round 1's draw |

## Dropouts

A site that drops before contributing is simply not in the Close. There's no secret-sharing round and no recovery phase, because masks only involve sites and helpers.

**A site that drops after Close stops the run**, once the deadline passes. Redoing the round in place wouldn't be safe: the aggregator would hold two sums over nearly the same updates.

## The ledger

`ledger/` holds a Solana program that anchors a run. The log stays the source of truth for what happened. The ledger is the neutral place for what needs one: stake, reward escrow, a public privacy budget, and a ruling when someone says the aggregate is wrong.

Per round the aggregator posts constant-size data in three transactions: the cohort and the root over its members' vector roots, then the mask roots, the aggregate's root and the next model's digest. Each site in the cohort acknowledges on the ledger after it has checked the round off the ledger and seen that the roots posted there are the ones in its own log. A round settles when every member has acknowledged and the challenge window has passed.

During the window anyone holding the vectors can open a dispute at one leaf. Each challenger has a dispute of their own, so the aggregator can't keep one out by disputing first, and a round settles only when none is open. They prove each cohort member's leaf, each helper's and the aggregate's against the posted roots. The program adds them up in the field. If the aggregate's leaf isn't the sum, the aggregator's stake goes to the challenger and the task ends.

The program enforces the same privacy arithmetic and the same cohort draw as the replay, so neither can be misstated to it.

See [ledger/README.md](../ledger/README.md) for accounts, instructions and measured compute costs. The nodes don't submit transactions themselves yet.

## Post-quantum parts

- Signatures on records and grants: Ed25519 or ML-DSA-65 per party. A run can mix them.
- Sealed seeds: X25519 and ML-KEM-768 together.
- TLS: connections settle on X25519MLKEM768, and a test confirms the negotiated group.
- Masks: ChaCha12 with 256 bit seeds.
- Not yet: credentials and revocation lists are Ed25519 only, since no post-quantum suite for them is final. `did:bio` and ledger accounts are Ed25519 by nature.

## Data plane

A vector is a file of little-endian field elements, cut into fixed chunks. Each chunk has its own mask stream, so everything runs chunk by chunk in parallel. Memory depends on chunk size, threads and connections, not on the model or the number of sites, and the disk takes the rest.

A chunk's root is checked on arrival, and an upload is synced before it can be counted. A restart rehashes the spool to find what's missing, and peers resend only that.

## Limits this design accepts

- **Masked updates can't be inspected.** A site can send an update of any size. Privacy terms bound what an honest site sends, not a dishonest one. Proving a norm bound on a masked vector is the main open item; the split between aggregator and helper is the setting PINE-style proofs are made for.
- **Privacy with few sites is expensive.** Noise in the average falls with the cohort size, so tens of sites pay heavily in accuracy at small epsilon. [benchmarks.md](benchmarks.md) has measurements.
- **Rewards can't follow contribution.** A per-site score is a statistic about one site, which is exactly what site-level privacy hides. The ledger pays an equal share per round and slashes only what can be proven.
- **One party can stop a run.** Any site or helper can. This trades liveness for the guarantee that nobody takes a model some party couldn't check.
- **A helper's disk must survive.** "Once per round" rests on the helper's replica.
- **All helpers are needed**, and the aggregator is one process. A restart resumes it, but there's no failover.
- **Fixed point costs bandwidth.** Coordinates are 8 bytes on the wire, twice float32.
- **Time is each party's own.** The time in a Join is the site's word, held near the aggregator's clock. Credential dates are checked against it once, at joining, and each site checks its own before every round. A site and an aggregator together can backdate a Join.
- **Sampling is public.** Who was drawn is in the log, so the privacy accounting takes no credit for it.
- **The sampler isn't constant-time.** How long noise takes to draw depends a little on its size. It's drawn locally, alongside training.
