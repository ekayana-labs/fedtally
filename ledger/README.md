# fedtally-ledger

The Solana program that anchors a fedtally run. The run itself is driven by a
signed log off chain. This program is the neutral record and arbiter: it holds
stake and reward escrow, enforces the privacy budget, checks cohort sampling,
records each round's commitments, takes each site's acknowledgement, settles
rounds, pays rewards, and slashes the aggregator when anyone proves that the
posted aggregate is not the sum of the committed vectors.

It never sees a model, an update or patient data. Everything it stores is a
32 byte digest, a bitmap or a number.

| Information | Account Address |
| --- | --- |
| fedtally-ledger program | `BZ3EJcg4FUF2Yzp48FnADkw5FhobY1NAvgneoSV2v6gD` |

The program ID is a placeholder. Its keypair was generated for the address
and never kept, so the program cannot be deployed there. Replace the ID in
`program/src/lib.rs` before deploying. `cargo build-sbf` writes a keypair of
its own under `target/deploy`, which has nothing to do with this ID and is
not kept in the repository either.

The program is built with [Pinocchio](https://github.com/anza-xyz/pinocchio).
It is `no_std` and allocation free. Every account has a fixed layout that
handlers edit in place, and all hashing goes through the SHA-256 syscall.

## Accounts

All accounts are program-derived. Data starts with
`sha256("account:<Name>")[..8]`, followed by packed little-endian fields. The
field tables are in [`program/src/state.rs`](program/src/state.rs). A per-site
bitmap is 128 bytes, and bit `i` is bit `i % 8` of byte `i / 8`.

| Account | Seeds | Bytes | Holds |
| --- | --- | --- | --- |
| Task | `"task", publisher, plan_id` | 561 | The terms, the budget spent, the latest round and model, the enrolled and revoked bitmaps. Above its rent: the reward escrow and the aggregator's stake. |
| Enrollment | `"site", task, index u16` | 261 | A site's wallet, index, status, admission digest and up to 4 status references. Above its rent: the site's stake. |
| Round | `"round", task, round u32` | 780 | State, the number of open disputes, beacon, cohort, the committed and posted roots, the acked and claimed bitmaps, the share. |
| Dispute | `"dispute", task, round u32, challenger` | 2275 | Challenger, disputed leaf, deadline, the running sum and the aggregate leaf. Above its rent: the bond. |
| Demand | `"demand", task, challenger, round u32 + kind + position u16 + leaf u64` | 101 + proof | Who asks for which leaf, the deadline, and the proof once it is answered. Above its rent: the bond. |
| StatusList | `"status", issuer, list u32` | 57 + bits / 8 | Issuer, size, sequence and the bitmap, up to 80,000 bits. |
| Buffer | `"buffer", owner` | 45 + size | A proof on its way to `dispute_leaf`, up to 10,000 bytes. |

## Instructions

The first 8 bytes of the data are `sha256("global:<name>")[..8]`. Arguments
follow in the order given, integers little-endian. A list is a u32 count and
then its items. Every instruction checks the owner, the derivation, and the
signer and writable flags of its accounts, and refuses trailing bytes.
Errors are custom codes from 6000, listed in
[`program/src/error.rs`](program/src/error.rs).

| Instruction | Accounts | Arguments | Effect |
| --- | --- | --- | --- |
| `publish` | publisher (signer), task, system program | the terms, in the order of `client::TaskTerms` | Creates the task and deposits `rounds * reward_per_round`. |
| `back` | aggregator (signer), task, system program | | Deposits the aggregator's stake, once. |
| `enroll` | wallet (signer), task, enrollment, system program | index u16, path list, admission, status reference list | Checks the wallet's path in the sites root, creates the enrollment and deposits the site's stake. |
| `open_round` | aggregator (signer), task, round, system program | | Opens the next round and fixes its beacon. Needs budget for a round at the minimum cohort. |
| `commit_root` | aggregator (signer), task, round | cohort bitmap, inputs root, close head, threshold | Checks the cohort and, in a sampling task, the tickets. Charges the budget for the cohort's size. |
| `post_aggregate` | aggregator (signer), task, round | mask root list, aggregate root, next model, seal head | Posts the round. The challenge window starts. |
| `ack` | wallet (signer), enrollment, round | round u32, seal head | Records a cohort member's acknowledgement. |
| `settle` | task, round | | After every ack and the challenge window, without an open dispute: finalizes the round and sets the shares. |
| `claim` | wallet (signer), enrollment, task, round | | Pays a cohort member its share of a final round. |
| `dispute_open` | challenger (signer), task, round, dispute, system program | round u32, leaf index u64 | Opens the challenger's dispute over one leaf and takes the bond. |
| `dispute_leaf` | challenger (signer), task, round, dispute, buffer | kind u8, position u16 | Checks one part of the proof from the buffer and folds it into the dispute. |
| `dispute_close` | task, round, dispute, challenger, aggregator | | Decides the dispute and closes it. |
| `demand_open` | challenger (signer), task, round, demand, system program | round u32, kind u8, position u16, leaf index u64 | Demands one leaf of one vector and takes the bond. |
| `demand_answer` | responder (signer), task, round, demand, buffer | | Checks the proof in the responder's buffer, keeps it in the demand, and pays the responder its fee. |
| `demand_close` | task, round, demand, challenger | | Closes a demand. An unanswered one past its deadline slashes the aggregator. |
| `round_close` | aggregator, task, round | | Returns the rent of a final round that is paid out and isn't the latest. |
| `status_init` | issuer (signer), status list, system program | list u32, bits u32 | Creates a status list. |
| `revoke` | issuer (signer), status list | bit u32 | Sets a bit, once. |
| `revoke_check` | task, enrollment, then the status lists it names | | Revokes the enrollment if a named bit is set. |
| `abort` | caller (signer), task, then the latest round if any | | Aborts an open task. |
| `withdraw` | wallet (signer), task, then the enrollment or the latest round | role u8 | Returns a stake or the publisher's remainder. |
| `buffer_init` | owner (signer), buffer, system program | size u32 | Creates the owner's buffer. |
| `buffer_write` | owner (signer), buffer | offset u32, byte list | Writes a chunk. |
| `buffer_close` | owner (signer), buffer | | Closes the buffer and returns its rent. |

Some behaviour worth knowing before writing a client:

- A round's cohort is listed in ascending site index, and `dispute_leaf`
  names an input by its position in that list. Kinds are 0 for an input, 1
  for a helper's mask and 2 for the aggregate.
- A proof in the buffer is the leaf's coordinates, then for an input the
  member's vector root and its path in the inputs root, then the leaf's path
  in its vector root. Paths have exactly the hashes their position needs, so
  a proof has one valid length. A buffer write at offset 0 starts the
  contents over, and any other write continues where the last one ended.
- A dispute is decided when every input, every mask and the aggregate leaf
  are in. If the sum differs from the aggregate in any coordinate, the round
  becomes Fraud, the task is aborted, and the aggregator's stake and the bond
  go to the challenger. If the sum agrees, the bond goes to the aggregator.
  If the deadline passes with parts missing, the bond stays in the task and
  ends up with the publisher. It never goes to the aggregator, who could
  otherwise open empty disputes for free.
- Each challenger has a dispute of their own, so a round can have any number
  open and nobody can keep a challenger out by disputing first. A round
  settles only once none is open. The first dispute to prove fraud ends the
  task, and the others on that round then close with their bonds returned.
- A fraud proof needs the vectors, so anyone refused a leaf during the challenge window can demand it. Anyone holding the leaf answers, with the proof a dispute takes. The answer pays a fee of 50,000 lamports, about the cost of its transactions, so withholding earns nothing, and the rest of the bond returns at the close. If nobody answers by the deadline, the round becomes Fraud and the aggregator's stake goes to the challenger. Deadlines on a round queue at least 64 slots apart, so a burst of demands can't outrun an honest aggregator. A demand holds the round open until it closes, and an answered one lets its challenger open a dispute over that leaf past the window.
- An aggregator therefore has to keep every round's vectors until the round settles. Run it with `--keep-spool`. Its anchor deletes a round's folder once it has settled.
- Getting a whole withheld chunk means one demand per leaf. Each holds a bond and rent until it closes, so a plan that may be anchored should use small chunks, and a long enough window for it.
- `publish` refuses a task whose `stall_slots` is not longer than its
  `challenge_slots`.
- `abort` is for the aggregator or the publisher, or for anyone once nothing
  has changed for longer than `stall_slots`. A round with open disputes
  blocks it.
- `withdraw` takes a role: 0 for a site, 1 for the aggregator, 2 for the
  publisher. If a task is aborted while a round is posted, the aggregator's
  stake stays until that round's challenge window has passed and no dispute
  is open, and the round can still be disputed. Aborting is therefore no way
  around a fraud proof.
- `revoke_check` succeeds without a change when no named bit is set. A named
  address that holds no status list counts as having no bit set.
- A task that does not sample ignores the threshold of `commit_root`.

## Shared definitions

The modules `merkle`, `field`, `privacy` and `sampling` implement the
definitions this program shares with the off-chain code: RFC 6962 trees over
SHA-256 with RFC 9162 inclusion checks, the field with p = 2^64 - 2^32 + 1,
the privacy cost of a round, and the beacon and tickets of cohort sampling.
They compile for the host too, and `client` builds every instruction and
derives every address off chain. Accounts decode through the views in
`state`, for example `Task::parse(data)`.

## Building and testing

You need the Rust version pinned in [rust-toolchain.toml](rust-toolchain.toml)
and the Solana CLI v3.1 or later, which provides `cargo build-sbf`.

```console
cargo build-sbf --manifest-path program/Cargo.toml
cargo test
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
```

## Compute units

Measured by `cargo test --test compute_units -- --nocapture` with fixed
wallets, so the figures are reproducible. An instruction that creates an
account searches for its bump, which costs more or less depending on the
address. The binary is about 109 KB.

| Instruction | CU |
| --- | --- |
| `publish` | 2398 |
| `back` | 1722 |
| `enroll` (16 sites, 2 status references) | 3371 |
| `open_round` | 3035 |
| `commit_root` (no sampling, cohort of 16) | 3004 |
| `commit_root` (sampling 8 of 16) | 5473 |
| `commit_root` (sampling 32 of 128) | 22345 |
| `commit_root` (sampling 64 of 1024) | 159077 |
| `post_aggregate` (1 helper) | 957 |
| `ack` | 750 |
| `settle` | 911 |
| `claim` | 1074 |
| `dispute_open` | 2851 |
| `dispute_leaf` (input of 16, full leaf, 2-level path) | 6164 |
| `dispute_leaf` (mask, full leaf, 2-level path) | 4910 |
| `dispute_leaf` (aggregate, full leaf, 2-level path) | 4006 |
| `dispute_leaf` (aggregate, full leaf, 20-level path) | 7435 |
| `dispute_leaf` (input of 16, full leaf, 20-level path) | 9608 |
| `dispute_close` (aggregate upheld) | 1308 |
| `dispute_close` (fraud) | 1312 |
| `demand_open` | 3013 |
| `demand_answer` (input of 3, full leaf) | 3328 |
| `demand_close` (answered) | 1196 |
| `round_close` | 751 |
| `status_init` (80000 bits) | 2738 |
| `status_init` (64 bits) | 2032 |
| `revoke` | 414 |
| `revoke_check` (2 status lists) | 1335 |
| `abort` | 545 |
| `withdraw` (site) | 749 |
| `withdraw` (aggregator) | 470 |
| `withdraw` (publisher) | 651 |
| `buffer_init` (4000 bytes) | 2322 |
| `buffer_write` (900 bytes) | 427 |
| `buffer_close` | 380 |

Sampling costs one hash per site of the task, about 155 CU each. At 1024
sites `commit_root` fits the default limit of 200,000 CU with little to
spare, so a client should ask for more.

## Anchoring a run

`fedtally anchor` sends these transactions for one party, from the crate
[`fedtally-chain`](../crates/fedtally-chain). It runs beside the party's node
and follows the log that node keeps. Each step it reads the task, the rounds
and its own accounts, works out the one transaction that is due, sends it and
waits for it to be confirmed. It keeps nothing about the ledger between
steps, so it can be stopped and started at any point.

| Role | Sends |
| --- | --- |
| `publisher` | `publish` with the plan's terms and the stakes given, and `withdraw` once the task is over. |
| `aggregator` | `back`, then per round `open_round`, `commit_root` and `post_aggregate` as the log reaches them, and `settle` after every ack and the window. `demand_answer` for any demand, from its spool. `round_close` once a round is paid out. `abort` if the log was aborted, and `withdraw` at the end. |
| `site` | `enroll` with the digest of its Join record. `ack` for a round it acknowledged in its own log, and only while the ledger holds that round's roots. `claim` for each final round, and `withdraw` at the end. |
| `watcher` | Nothing while rounds add up. It is not in the plan, so the aggregator has to name it with `--watcher`. It fetches each round's vectors, checks them against the ledger's roots and adds them up. For a wrong leaf it sends `buffer_init`, `dispute_open`, then `buffer_write` and `dispute_leaf` for every part, and `dispute_close`. Halfway through the window it sends `demand_open` for any leaf it still can't fetch, and `demand_close` when that's settled. It also sends `revoke_check` for a site the log has revoked, again until the bit lands. |
| `issuer` | `status_init` and `revoke`, for its own lists in the log, up to 80,000 bits. |

A site that finds other roots on the ledger than in its log does not
acknowledge, and says so. A watcher says so too, and still disputes if the
aggregator serves vectors that match what it posted.

The aggregator serves a round's vectors while the round is under way, and
afterwards only if it keeps its spool, which it needs anyway to answer demands.
A status list is found on the ledger under its issuer's wallet and
the first four bytes of the digest of the list's id.

```console
fedtally anchor --role publisher --rpc $RPC --program $ID --key sponsor.key --plan plan.json \
    --stakes 1000000000 5000000000 300000000 1000000000 150 1500
fedtally anchor --role aggregator --rpc $RPC --program $ID --key aggregator.key --plan plan.json --dir aggregator
fedtally anchor --role site --rpc $RPC --program $ID --key site.key --plan plan.json --dir site
fedtally anchor --role issuer --rpc $RPC --program $ID --key issuer.key --plan plan.json --dir site
fedtally anchor --role watcher --rpc $RPC --program $ID --key watcher.key --plan plan.json --dir watcher \
    --connect aggregator.example:7443
```

[`program/tests/anchored.rs`](program/tests/anchored.rs) mirrors a real run
into LiteSVM by hand. The same run against a local validator, with every
role and a dishonest aggregator, is an ignored test of `fedtally-chain`:

```console
cargo test -p fedtally-chain --test validator -- --ignored --nocapture
```

## What it does not do

- Rewards and stakes are lamports. There are no token rewards.
- A dispute is over one leaf, and a challenger has one dispute per round at
  a time. Different challengers dispute a round side by side.
- Task and enrollment accounts are never closed, nor is the latest round, so
  their rent stays locked. Other rounds return theirs with `round_close` once
  every share is claimed. A share nobody claims keeps its round open.
- `revoke_check` only sees the status references a site chose to give when
  it enrolled. A site that gives none cannot be revoked here.
- Nothing here checks a site's credentials. An enrollment records the
  admission digest and the status references the site gives, and the off
  chain side decides whether they are the right ones.
- Sites are not slashed. A site's stake is only held and returned.
- The program ID is a placeholder, as said above.

## License

[Apache-2.0](../LICENSE)
