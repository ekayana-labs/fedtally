//! A real fedtally run, anchored. Three sites, a helper and an aggregator
//! train over TLS. Their log is then mirrored here with the sites' own keys,
//! and a dispute is settled on the vectors the run left on disk.

mod common;

use std::path::{Path, PathBuf};

use common::{key, World};
use fedtally_core::anchor::{self, Terms};
use fedtally_core::merkle;
use fedtally_core::model::{write_model, Dtype, Layout, TensorSpec, TensorView};
use fedtally_core::replay::Replay;
use fedtally_core::vector::{self, Digest, Shape, LEAF_COORDS};
use fedtally_ledger::client::{self, ProofPart, TaskTerms};
use fedtally_ledger::state::{ROUND_FINAL, ROUND_FRAUD, ROUND_POSTED, TASK_ABORTED, TASK_FINISHED};
use fedtally_node::aggregator::run_aggregator;
use fedtally_node::helper::run_helper;
use fedtally_node::local::{LocalFederation, Options};
use fedtally_node::site::{run_site, Submit, TrainContext, Trainer};
use solana_keypair::Keypair;
use solana_signer::Signer;

const SITES: usize = 3;
const ROUNDS: u32 = 2;
const STAKE: u64 = 1_000_000_000;
const REWARD: u64 = 300_000_000;
const WINDOW: u64 = 50;

fn layout() -> Layout {
    Layout {
        tensors: vec![TensorSpec {
            name: "w".into(),
            dtype: Dtype::F32,
            shape: vec![300],
        }],
    }
}

struct Towards(f32);

impl Trainer for Towards {
    fn train(&mut self, _: &TrainContext, submit: &mut Submit<'_>) -> Result<(), String> {
        submit(&[TensorView::F32(&[self.0; 300])], 1)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

/// What a finished run leaves behind.
struct Trained {
    _dir: tempfile::TempDir,
    fed: LocalFederation,
    replay: Replay,
    terms: Terms,
}

impl Trained {
    fn round_file(&self, round: u32, name: &str) -> PathBuf {
        self.fed
            .aggregator_dir()
            .join(format!("round-{round}"))
            .join(name)
    }

    fn shape(&self) -> Shape {
        self.fed.plan.shape()
    }
}

fn train() -> Trained {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let init = dir.path().join("init.safetensors");
        write_model(&init, &layout(), &[TensorView::F32(&[0.5; 300])]).unwrap();
        let mut options = Options::new(SITES, 1, ROUNDS);
        options.chunk_coords = LEAF_COORDS;
        options.did_bio = Some("localnet".into());
        let fed = LocalFederation::create(dir.path(), layout(), &init, &options).unwrap();
        fed.save().unwrap();

        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut config = fed.aggregator_config("127.0.0.1:0");
        config.keep_spool = true;
        config.on_ready = Some(Box::new(move |addr| {
            let _ = tx.send(addr);
        }));
        let aggregator = tokio::spawn(run_aggregator(config));
        let addr = rx.await.unwrap();
        tokio::spawn(run_helper(fed.helper_config(0, &addr)));
        let sites: Vec<_> = (0..SITES)
            .map(|i| {
                let trainer = Box::new(Towards(i as f32));
                tokio::spawn(run_site(fed.site_config(i, &addr), trainer))
            })
            .collect();
        for site in sites {
            site.await.unwrap().unwrap();
        }
        let replay = aggregator.await.unwrap().unwrap();
        let terms = fed.plan.ledger_terms(fed.plan_id).unwrap();
        Trained {
            _dir: dir,
            fed,
            replay,
            terms,
        }
    })
}

/// A party's Solana keypair: the Ed25519 key its `did:bio` names.
fn keypair(root: &Path, name: &str) -> Keypair {
    let file: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join(name)).unwrap()).unwrap();
    let mut seed = [0u8; 32];
    hex::decode_to_slice(file["seed"].as_str().unwrap(), &mut seed).unwrap();
    Keypair::new_from_array(seed)
}

struct Anchored {
    world: World,
    task: [u8; 32],
    aggregator: Keypair,
    sites: Vec<Keypair>,
}

fn publish(trained: &Trained) -> Anchored {
    let mut world = World::new();
    let root = &trained.fed.root;
    let publisher = keypair(root, "sponsor.key");
    let aggregator = keypair(root, "aggregator.key");
    let sites: Vec<Keypair> = (0..SITES)
        .map(|i| keypair(root, &format!("site-{i}.key")))
        .collect();
    for wallet in sites.iter().chain([&publisher, &aggregator]) {
        world.svm.airdrop(&wallet.pubkey(), common::WALLET).unwrap();
    }
    // The key in each party's name is the key that signs for it here.
    for (site, did) in sites.iter().zip(&trained.fed.plan.sites) {
        assert_eq!(key(site), anchor::wallet(did).unwrap());
    }
    assert_eq!(
        key(&aggregator),
        anchor::wallet(&trained.fed.plan.aggregator.did).unwrap()
    );

    let terms = &trained.terms;
    let task_terms = TaskTerms {
        aggregator: key(&aggregator),
        plan_id: terms.plan.0,
        sites_root: terms.sites_root,
        model: terms.model.0,
        sites: terms.sites,
        helpers: terms.helpers,
        coords: terms.coords,
        rounds: terms.rounds,
        min_cohort: terms.min_cohort,
        sample: terms.sample,
        sensitivity: terms.sensitivity,
        noise: terms.noise,
        colluding: terms.colluding,
        budget: terms.budget,
        site_stake: STAKE,
        aggregator_stake: 5 * STAKE,
        reward_per_round: REWARD,
        dispute_bond: STAKE,
        challenge_slots: WINDOW,
        stall_slots: 10 * WINDOW,
    };
    let task = client::task(&key(&publisher), &terms.plan.0);
    world.run(
        client::publish(&key(&publisher), &task_terms),
        &[&publisher],
    );
    world.run(client::back(&key(&aggregator), &task), &[&aggregator]);

    let wallets: Vec<merkle::Hash> = sites.iter().map(|s| merkle::leaf(&key(s))).collect();
    for (i, site) in sites.iter().enumerate() {
        let path = merkle::path(wallets.clone(), i);
        let admission = [i as u8; 32];
        world.run(
            client::enroll(&key(site), &task, i as u16, &path, &admission, &[]),
            &[site],
        );
    }
    Anchored {
        world,
        task,
        aggregator,
        sites,
    }
}

impl Anchored {
    /// Posts a round as the log has it, except for the aggregate root.
    fn post(&mut self, trained: &Trained, round: u32, aggregate_root: Digest) {
        let committed = trained.replay.committed(round).unwrap();
        let posted = trained.replay.posted(round).unwrap();
        let aggregator = key(&self.aggregator);
        self.world.run(
            client::open_round(&aggregator, &self.task, round),
            &[&self.aggregator],
        );
        // The program draws the same beacon from the model it holds.
        assert_eq!(
            self.world.round(&self.task, round).beacon(),
            &committed.beacon
        );
        let cohort = client::site_bitmap(committed.cohort.iter().map(|s| *s as u16));
        self.world.run(
            client::commit_root(
                &aggregator,
                &self.task,
                round,
                &cohort,
                &committed.inputs_root,
                &committed.close_head.0,
                &committed.threshold,
            ),
            &[&self.aggregator],
        );
        let masks: Vec<[u8; 32]> = posted.mask_roots.iter().map(|m| m.0).collect();
        self.world.run(
            client::post_aggregate(
                &aggregator,
                &self.task,
                round,
                &masks,
                &aggregate_root.0,
                &posted.next_model.0,
                &posted.seal_head.0,
            ),
            &[&self.aggregator],
        );
    }

    fn acknowledge_and_settle(&mut self, trained: &Trained, round: u32) {
        let head = trained.replay.posted(round).unwrap().seal_head.0;
        for (i, site) in self.sites.iter().enumerate() {
            self.world.run(
                client::ack(&key(site), &self.task, i as u16, round, &head),
                &[site],
            );
        }
        self.world.warp(WINDOW);
        self.world.run(client::settle(&self.task, round), &[]);
    }

    /// Proves leaf `leaf` of every vector in a round, with the aggregate
    /// taken from `aggregate`. Returns the challenger.
    fn dispute(&mut self, trained: &Trained, round: u32, leaf: u64, aggregate: &Path) -> Keypair {
        let challenger = self.world.wallet();
        let who = key(&challenger);
        self.world
            .run(client::buffer_init(&who, 4_000), &[&challenger]);
        self.world.run(
            client::dispute_open(&who, &self.task, round, leaf),
            &[&challenger],
        );

        let shape = trained.shape();
        let state = &trained.replay.rounds[round as usize - 1];
        let prove = |path: &Path| {
            let manifest = vector::manifest_of(path, shape).unwrap();
            let proof = vector::prove_leaf(path, shape, &manifest, leaf).unwrap();
            assert!(proof.fits(&manifest.root, shape));
            (manifest.root, proof)
        };
        let bytes =
            |values: &[u64]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };

        let roots: Vec<Digest> = state
            .included
            .iter()
            .map(|site| state.contributions[site].masked)
            .collect();
        let mut parts = Vec::new();
        for (position, site) in state.included.iter().enumerate() {
            let (root, proof) = prove(&trained.round_file(round, &format!("site-{site}.bin")));
            assert_eq!(
                root, roots[position],
                "the spool holds what the log commits to"
            );
            let within = anchor::inputs_path(&roots, position);
            parts.push((
                ProofPart::Input(position as u16),
                client::input_proof(&bytes(&proof.values), &root.0, &within, &proof.path),
            ));
        }
        let (_, mask) = prove(&trained.round_file(round, "mask-0.bin"));
        parts.push((
            ProofPart::Mask(0),
            client::leaf_proof(&bytes(&mask.values), &mask.path),
        ));
        let (_, total) = prove(aggregate);
        parts.push((
            ProofPart::Aggregate,
            client::leaf_proof(&bytes(&total.values), &total.path),
        ));

        for (part, proof) in parts {
            self.world.upload(&challenger, &proof);
            self.world.run(
                client::dispute_leaf(&who, &self.task, round, part),
                &[&challenger],
            );
        }
        self.world.run(
            client::dispute_close(&self.task, round, &who, &key(&self.aggregator)),
            &[],
        );
        challenger
    }
}

#[test]
fn an_honest_run_settles_and_a_dispute_against_it_fails() {
    let trained = train();
    assert_eq!(trained.replay.rounds.len(), ROUNDS as usize);
    let mut anchored = publish(&trained);

    let honest = |round| trained.replay.posted(round).unwrap().aggregate_root;
    anchored.post(&trained, 1, honest(1));

    // Someone disputes the honest round. Every leaf checks out, the sum
    // matches, and the bond goes to the aggregator.
    let before = anchored.world.balance(&key(&anchored.aggregator));
    let aggregate = trained.round_file(1, "aggregate.bin");
    assert_eq!(
        vector::first_wrong_leaf(
            trained.shape(),
            &(0..SITES)
                .map(|s| trained.round_file(1, &format!("site-{s}.bin")))
                .collect::<Vec<_>>()
                .iter()
                .map(PathBuf::as_path)
                .collect::<Vec<_>>(),
            &[&trained.round_file(1, "mask-0.bin")],
            &aggregate,
        )
        .unwrap(),
        None
    );
    let challenger = anchored.dispute(&trained, 1, 1, &aggregate);
    assert_eq!(
        anchored.world.round(&anchored.task, 1).state(),
        ROUND_POSTED
    );
    assert_eq!(
        anchored.world.balance(&key(&anchored.aggregator)),
        before + STAKE
    );
    assert!(anchored.world.balance(&key(&challenger)) < common::WALLET - STAKE + 1);

    anchored.acknowledge_and_settle(&trained, 1);
    anchored.post(&trained, 2, honest(2));
    anchored.acknowledge_and_settle(&trained, 2);

    let task = anchored.world.task(&anchored.task);
    assert_eq!(task.state(), TASK_FINISHED);
    assert_eq!(
        task.model(),
        &trained.replay.model.0,
        "the ledger ends on the log's model"
    );
    assert_eq!(anchored.world.round(&anchored.task, 2).state(), ROUND_FINAL);

    // Each site is paid its equal share of each round.
    let site = &anchored.sites[1];
    let paid = anchored.world.balance(&key(site));
    for round in 1..=ROUNDS {
        anchored
            .world
            .run(client::claim(&key(site), &anchored.task, 1, round), &[site]);
    }
    assert_eq!(
        anchored.world.balance(&key(site)),
        paid + u64::from(ROUNDS) * (REWARD / SITES as u64)
    );
}

#[test]
fn an_aggregate_that_is_not_the_sum_costs_the_aggregator_its_stake() {
    let trained = train();
    let mut anchored = publish(&trained);
    let shape = trained.shape();

    // The aggregator posts an aggregate with one coordinate moved.
    let wrong = trained.fed.root.join("wrong.bin");
    std::fs::copy(trained.round_file(1, "aggregate.bin"), &wrong).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&wrong)
        .unwrap();
    let mut chunk = Vec::new();
    vector::read_chunk(&file, shape, 1, &mut chunk).unwrap();
    chunk[17] ^= 1;
    vector::write_chunk_bytes(&file, shape, 1, vector::chunk_bytes(&chunk)).unwrap();
    let wrong_root = vector::manifest_of(&wrong, shape).unwrap().root;
    anchored.post(&trained, 1, wrong_root);

    // Anyone holding the vectors finds the leaf and proves it.
    let inputs: Vec<PathBuf> = (0..SITES)
        .map(|s| trained.round_file(1, &format!("site-{s}.bin")))
        .collect();
    let input_paths: Vec<&Path> = inputs.iter().map(PathBuf::as_path).collect();
    let mask = trained.round_file(1, "mask-0.bin");
    let leaf = vector::first_wrong_leaf(shape, &input_paths, &[&mask], &wrong)
        .unwrap()
        .unwrap();
    assert_eq!(leaf, 1);

    let challenger = anchored.dispute(&trained, 1, leaf, &wrong);
    let task = anchored.world.task(&anchored.task);
    assert_eq!((task.state(), task.fraud()), (TASK_ABORTED, true));
    assert_eq!(anchored.world.round(&anchored.task, 1).state(), ROUND_FRAUD);
    assert!(anchored.world.balance(&key(&challenger)) > common::WALLET + 4 * STAKE);
}
