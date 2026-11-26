//! A tick that arrives while a store is answering the previous one is not lost.

use crate::prelude::*;
use crate::{Config, Engine, Instance, Module, Store, Trap, UpdateDeadline};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// A loop that ends by itself after two hundred million iterations, so that a
// lost tick fails the test instead of hanging it.
const LOOP: &str = r#"(module
    (func (export "run") (result i32)
        (local $i i32)
        (loop $again
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br_if $again (i32.lt_u (local.get $i) (i32.const 200000000))))
        (local.get $i)))"#;

struct State {
    stop: Arc<AtomicBool>,
    calls: Arc<AtomicU32>,
}

// The embedder signals a store by setting a flag the callback reads and
// ticking the epoch once. Here the signal arrives while the callback is
// answering an earlier tick: after it has read the flag, before it returns.
#[test]
fn a_tick_during_the_callback_makes_the_store_look_again() -> Result<()> {
    let mut config = Config::new();
    config.epoch_interruption(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, LOOP)?;
    let stop = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicU32::new(0));
    let mut store = Store::new(
        &engine,
        State {
            stop: stop.clone(),
            calls: calls.clone(),
        },
    );
    store.set_epoch_deadline(1);
    let ticker = engine.clone();
    store.epoch_deadline_callback(move |state| {
        let data = state.data();
        data.calls.fetch_add(1, Ordering::SeqCst);
        if data.stop.load(Ordering::SeqCst) {
            return Ok(UpdateDeadline::Interrupt);
        }
        // The signal lands now: the flag was read as not set.
        data.stop.store(true, Ordering::SeqCst);
        ticker.increment_epoch();
        Ok(UpdateDeadline::Continue(1))
    });
    let instance = Instance::new(&mut store, &module, &[])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    // An earlier tick, for some other store: it makes this one look.
    engine.increment_epoch();
    let result = run.call(&mut store, ());
    assert_eq!(calls.load(Ordering::SeqCst), 2, "the store did not look again");
    assert!(
        matches!(result.unwrap_err().downcast_ref::<Trap>(), Some(Trap::Interrupt)),
        "the loop ran to its end"
    );
    Ok(())
}

// Without a tick during the callback nothing changes: one tick, one call, and
// the deadline is one tick on.
#[test]
fn continue_still_extends_by_whole_ticks() -> Result<()> {
    let mut config = Config::new();
    config.epoch_interruption(true);
    let engine = Engine::new(&config)?;
    let module = Module::new(&engine, LOOP)?;
    let calls = Arc::new(AtomicU32::new(0));
    let mut store = Store::new(
        &engine,
        State {
            stop: Arc::new(AtomicBool::new(false)),
            calls: calls.clone(),
        },
    );
    store.set_epoch_deadline(1);
    store.epoch_deadline_callback(|state| {
        state.data().calls.fetch_add(1, Ordering::SeqCst);
        Ok(UpdateDeadline::Continue(1))
    });
    let instance = Instance::new(&mut store, &module, &[])?;
    let run = instance.get_typed_func::<(), i32>(&mut store, "run")?;
    engine.increment_epoch();
    assert_eq!(run.call(&mut store, ())?, 200000000);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}
