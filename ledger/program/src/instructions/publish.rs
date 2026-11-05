//! Publish a task. The publisher signs, pays the rent of the task account
//! and deposits the reward of every round in it.
//!
//! The accounts are `[publisher, task, system_program]`.

use pinocchio::{error::ProgramError, AccountView, ProgramResult};

use crate::{
    error::{require, LedgerError},
    instructions::shared::*,
    privacy,
    reader::{Args, Reader},
    state::*,
};

/// What the publisher sets, in argument order.
struct Terms<'a> {
    aggregator: &'a [u8; 32],
    plan_id: &'a [u8; 32],
    sites_root: &'a [u8; 32],
    model: &'a [u8; 32],
    sites: u16,
    helpers: u8,
    coords: u64,
    rounds: u32,
    min_cohort: u16,
    sample: u16,
    sensitivity: u64,
    noise: u64,
    colluding: u16,
    budget: u64,
    site_stake: u64,
    aggregator_stake: u64,
    reward_per_round: u64,
    dispute_bond: u64,
    challenge_slots: u64,
    stall_slots: u64,
}

impl<'a> Terms<'a> {
    fn read(args: &'a [u8]) -> Result<Self, ProgramError> {
        let mut r = Reader::<Args>::new(args);
        let terms = Self {
            aggregator: r.array()?,
            plan_id: r.array()?,
            sites_root: r.array()?,
            model: r.array()?,
            sites: r.u16()?,
            helpers: r.u8()?,
            coords: r.u64()?,
            rounds: r.u32()?,
            min_cohort: r.u16()?,
            sample: r.u16()?,
            sensitivity: r.u64()?,
            noise: r.u64()?,
            colluding: r.u16()?,
            budget: r.u64()?,
            site_stake: r.u64()?,
            aggregator_stake: r.u64()?,
            reward_per_round: r.u64()?,
            dispute_bond: r.u64()?,
            challenge_slots: r.u64()?,
            stall_slots: r.u64()?,
        };
        r.finish()?;
        Ok(terms)
    }

    fn check(&self) -> ProgramResult {
        require(
            (1..=MAX_SITES as u16).contains(&self.sites),
            LedgerError::InvalidSites,
        )?;
        require(
            self.helpers as usize <= MAX_HELPERS,
            LedgerError::InvalidHelpers,
        )?;
        require(
            (1..=self.sites).contains(&self.min_cohort),
            LedgerError::InvalidMinCohort,
        )?;
        require(
            self.sample == 0 || (self.min_cohort..=self.sites).contains(&self.sample),
            LedgerError::InvalidSample,
        )?;
        require(self.rounds > 0, LedgerError::InvalidRounds)?;
        // Otherwise anyone could abort a task while a round waits out its window.
        require(
            self.stall_slots > self.challenge_slots,
            LedgerError::InvalidWindows,
        )?;
        require(
            self.noise == 0 || self.colluding < self.min_cohort,
            LedgerError::InvalidCollusion,
        )?;
        let cost = privacy::round_cost(
            self.sensitivity,
            self.noise,
            self.colluding,
            self.min_cohort,
        )?;
        require(self.budget >= cost, LedgerError::BudgetTooSmall)
    }

    /// The reward of every round, which the task holds from the start.
    fn escrow(&self) -> Result<u64, ProgramError> {
        (self.rounds as u64)
            .checked_mul(self.reward_per_round)
            .ok_or(ProgramError::ArithmeticOverflow)
    }
}

pub fn process(accounts: &mut [AccountView], args: &[u8]) -> ProgramResult {
    let [publisher, task_account, system_program, ..] = accounts else {
        return Err(ProgramError::NotEnoughAccountKeys);
    };
    check_payer(publisher)?;
    check_writable(task_account)?;
    check_system_program(system_program)?;
    let terms = Terms::read(args)?;
    terms.check()?;

    let bump = create(
        publisher,
        task_account,
        &[TASK_SEED, publisher.address().as_array(), terms.plan_id],
        TASK_SPACE,
        terms.escrow()?,
    )?;

    let mut task = Task::init(task_account.try_borrow_mut()?)?;
    task.set_bump(bump);
    task.set_state(TASK_OPEN);
    task.set_helpers(terms.helpers);
    task.set_sites(terms.sites);
    task.set_min_cohort(terms.min_cohort);
    task.set_sample(terms.sample);
    task.set_colluding(terms.colluding);
    task.set_rounds(terms.rounds);
    *task.publisher_mut() = *publisher.address().as_array();
    *task.aggregator_mut() = *terms.aggregator;
    *task.plan_id_mut() = *terms.plan_id;
    *task.sites_root_mut() = *terms.sites_root;
    *task.model_mut() = *terms.model;
    task.set_coords(terms.coords);
    task.set_sensitivity(terms.sensitivity);
    task.set_noise(terms.noise);
    task.set_budget(terms.budget);
    task.set_site_stake(terms.site_stake);
    task.set_aggregator_stake(terms.aggregator_stake);
    task.set_reward_per_round(terms.reward_per_round);
    task.set_dispute_bond(terms.dispute_bond);
    task.set_challenge_slots(terms.challenge_slots);
    task.set_stall_slots(terms.stall_slots);
    task.set_last_change(current_slot()?);
    Ok(())
}
