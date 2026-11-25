//! psizam's deterministic stack budget, enforced in compiled code.

use crate::prelude::*;
use crate::{AsContextMut, Caller, Config, Engine, Instance, Linker, Module, Store, Trap};

/// `text` assembled, with the stack cost of each defined function attached.
fn module_bytes(text: &str, costs: &[u32]) -> Vec<u8> {
    let mut bytes = wat::parse_str(text).unwrap();
    let name = wasmtime_environ::PSIZAM_STACK_COSTS_SECTION.as_bytes();
    let size = 1 + name.len() + 4 * costs.len();
    assert!(size < 128 && name.len() < 128);
    bytes.push(0);
    bytes.push(size as u8);
    bytes.push(name.len() as u8);
    bytes.extend_from_slice(name);
    for cost in costs {
        bytes.extend_from_slice(&cost.to_le_bytes());
    }
    bytes
}

fn engine() -> Result<Engine> {
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    Engine::new(&config)
}

fn overflowed(result: Result<()>) -> bool {
    matches!(
        result.unwrap_err().downcast_ref::<Trap>(),
        Some(Trap::StackOverflow)
    )
}

// Function 0 recurses `n` deep and costs 100 bytes a frame; function 1 is a
// leaf costing 40; function 2 calls the leaf through the table.
const RECURSE: &str = r#"(module
    (type $leaf (func (result i32)))
    (table 1 funcref)
    (elem (i32.const 0) $leaf)
    (func $down (export "down") (param i32)
        (if (local.get 0) (then (call $down (i32.sub (local.get 0) (i32.const 1))))))
    (func $leaf (export "leaf") (result i32) (i32.const 7))
    (func (export "indirect") (result i32) (call_indirect (type $leaf) (i32.const 0)))
    ;; Callees whose parameter lists differ from the caller's, one of them
    ;; with a vector parameter.
    (func $wide (param i64 f64 v128 i32) (result i32) (local.get 3))
    (func (export "mixed") (param i32) (result i32)
        (i32.add
            (call $leaf)
            (call $wide (i64.const 1) (f64.const 2) (v128.const i32x4 1 2 3 4) (local.get 0))))
)"#;
const RECURSE_COSTS: [u32; 5] = [100, 40, 10, 60, 20];

#[test]
fn psizam_stack_meter_fails_the_same_call_for_a_given_budget() -> Result<()> {
    let engine = engine()?;
    let module = Module::new(&engine, module_bytes(RECURSE, &RECURSE_COSTS))?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let down = instance.get_typed_func::<i32, ()>(&mut store, "down")?;
    let leaf = instance.get_typed_func::<(), i32>(&mut store, "leaf")?;
    let indirect = instance.get_typed_func::<(), i32>(&mut store, "indirect")?;

    // `down(n)` makes n + 1 frames of 100 bytes.
    for (budget, deepest) in [(100u64, 0i32), (199, 0), (200, 1), (1000, 9), (100_000, 999)] {
        store.psizam_set_stack_remaining(budget);
        down.call(&mut store, deepest)?;
        // A call leaves the budget as it found it.
        assert_eq!(store.psizam_stack_remaining(), budget);
        store.psizam_set_stack_remaining(budget);
        assert!(overflowed(down.call(&mut store, deepest + 1)));
        // So does a call that trapped.
        assert_eq!(store.psizam_stack_remaining(), budget);
    }

    // The budget is compared with the cost exactly.
    store.psizam_set_stack_remaining(39);
    assert!(overflowed(leaf.call(&mut store, ()).map(|_| ())));
    store.psizam_set_stack_remaining(40);
    assert_eq!(leaf.call(&mut store, ())?, 7);

    // An indirect call passes the budget on like a direct one.
    store.psizam_set_stack_remaining(49);
    assert!(overflowed(indirect.call(&mut store, ()).map(|_| ())));
    store.psizam_set_stack_remaining(50);
    assert_eq!(indirect.call(&mut store, ())?, 7);

    // Calls to functions of other shapes carry the budget too: `mixed`
    // (20) calls `leaf` (40) and then `wide` (60), so it needs 80.
    let mixed = instance.get_typed_func::<i32, i32>(&mut store, "mixed")?;
    store.psizam_set_stack_remaining(79);
    assert!(overflowed(mixed.call(&mut store, 5).map(|_| ())));
    store.psizam_set_stack_remaining(80);
    assert_eq!(mixed.call(&mut store, 5)?, 12);

    // Nothing remains by default: a store that was never given a budget
    // cannot run a function that costs anything.
    let mut fresh = Store::new(&engine, ());
    let instance = Instance::new(&mut fresh, &module, &[])?;
    let leaf = instance.get_typed_func::<(), i32>(&mut fresh, "leaf")?;
    assert!(overflowed(leaf.call(&mut fresh, ()).map(|_| ())));
    Ok(())
}

// Function 0 (cost 100) calls the host; function 1 (cost 30) is what the host
// calls back into.
const HOST: &str = r#"(module
    (import "env" "host" (func $host (param i32) (result i64)))
    (func (export "outer") (param i32) (result i64) (call $host (local.get 0)))
    (func (export "inner") (result i32) (i32.const 1))
)"#;
const HOST_COSTS: [u32; 2] = [100, 30];

#[test]
fn psizam_stack_meter_host_sees_and_sets_the_budget() -> Result<()> {
    let engine = engine()?;
    let module = Module::new(&engine, module_bytes(HOST, &HOST_COSTS))?;
    let mut linker = Linker::<Vec<u64>>::new(&engine);
    // The host records what remains; with a nonzero argument it calls back
    // into wasm with that many bytes of budget, and records what it finds
    // afterwards. It returns what remained when it was called.
    linker.func_wrap(
        "env",
        "host",
        |mut caller: Caller<'_, Vec<u64>>, reenter: i32| -> Result<i64> {
            let seen = caller.as_context_mut().psizam_stack_remaining();
            caller.data_mut().push(seen);
            if reenter != 0 {
                let inner = caller
                    .get_export("inner")
                    .and_then(|e| e.into_func())
                    .unwrap()
                    .typed::<(), i32>(&caller)?;
                caller
                    .as_context_mut()
                    .psizam_set_stack_remaining(reenter as u64);
                let result = inner.call(&mut caller, ());
                let after = caller.as_context_mut().psizam_stack_remaining();
                caller.data_mut().push(after);
                caller.data_mut().push(result.is_ok() as u64);
            }
            Ok(seen as i64)
        },
    )?;
    let mut store = Store::new(&engine, Vec::new());
    let instance = linker.instantiate(&mut store, &module)?;
    let outer = instance.get_typed_func::<i32, i64>(&mut store, "outer")?;

    // The host sees the caller's budget less the caller's own cost.
    store.psizam_set_stack_remaining(1000);
    assert_eq!(outer.call(&mut store, 0)?, 900);
    assert_eq!(store.psizam_stack_remaining(), 1000);

    // A call back into wasm starts with what the host set (30 is enough for
    // `inner`, 29 is not) and leaves that value in place. Neither changes
    // what the suspended caller has left.
    store.data_mut().clear();
    store.psizam_set_stack_remaining(1000);
    assert_eq!(outer.call(&mut store, 30)?, 900);
    assert_eq!(store.data().as_slice(), &[900, 30, 1]);
    assert_eq!(store.psizam_stack_remaining(), 1000);

    store.data_mut().clear();
    store.psizam_set_stack_remaining(1000);
    assert_eq!(outer.call(&mut store, 29)?, 900);
    assert_eq!(store.data().as_slice(), &[900, 29, 0]);
    assert_eq!(store.psizam_stack_remaining(), 1000);
    Ok(())
}

#[test]
fn psizam_stack_meter_requires_one_cost_per_function() -> Result<()> {
    let engine = engine()?;
    // No section at all, too few costs, and too many.
    assert!(Module::new(&engine, wat::parse_str(RECURSE)?).is_err());
    assert!(Module::new(&engine, module_bytes(RECURSE, &RECURSE_COSTS[..2])).is_err());
    assert!(Module::new(&engine, module_bytes(RECURSE, &[1, 2, 3, 4, 5, 6])).is_err());
    // The last section is the one in force: appending the real costs to a
    // module that already carries a section of that name overrides it.
    let mut twice = module_bytes(RECURSE, &[0, 0, 0, 0, 0]);
    let real = module_bytes("(module)", &RECURSE_COSTS);
    twice.extend_from_slice(&real[8..]);
    let module = Module::new(&engine, &twice)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let leaf = instance.get_typed_func::<(), i32>(&mut store, "leaf")?;
    store.psizam_set_stack_remaining(39);
    assert!(overflowed(leaf.call(&mut store, ()).map(|_| ())));
    // An engine that does not meter ignores the section, and its compiled
    // code cannot be loaded by one that does.
    let plain = Engine::default();
    let module = Module::new(&plain, module_bytes(RECURSE, &RECURSE_COSTS))?;
    let serialized = module.serialize()?;
    assert!(unsafe { Module::deserialize(&engine, &serialized) }.is_err());
    let metered = Module::new(&engine, module_bytes(RECURSE, &RECURSE_COSTS))?;
    assert!(unsafe { Module::deserialize(&plain, &metered.serialize()?) }.is_err());
    Ok(())
}

#[test]
fn psizam_stack_meter_leaves_the_start_function_to_the_embedder() -> Result<()> {
    let engine = engine()?;
    // The start function (cost 50) is not exported and sets a global.
    let text = r#"(module
        (global $started (mut i32) (i32.const 0))
        (func $init (global.set $started (i32.const 1)))
        (func (export "started") (result i32) (global.get $started))
        (start $init)
    )"#;
    // Without the request the start function runs at instantiation, where
    // nothing remains for it.
    let eager = Module::new(&engine, module_bytes(text, &[50, 0]))?;
    let mut store = Store::new(&engine, ());
    assert!(overflowed(Instance::new(&mut store, &eager, &[]).map(|_| ())));

    let mut deferring = module_bytes(text, &[50, 0]);
    let name = wasmtime_environ::PSIZAM_DEFER_START_SECTION.as_bytes();
    deferring.extend_from_slice(&[0, 1 + name.len() as u8, name.len() as u8]);
    deferring.extend_from_slice(name);
    let module = Module::new(&engine, deferring)?;
    let mut store = Store::new(&engine, ());
    // Instantiation succeeds with no budget set, because nothing of the
    // module's own code runs.
    let instance = Instance::new(&mut store, &module, &[])?;
    let started = instance.get_typed_func::<(), i32>(&mut store, "started")?;
    assert_eq!(started.call(&mut store, ())?, 0);

    let start = instance
        .psizam_deferred_start(&mut store)
        .unwrap()
        .typed::<(), ()>(&store)?;
    store.psizam_set_stack_remaining(49);
    assert!(overflowed(start.call(&mut store, ())));
    assert_eq!(started.call(&mut store, ())?, 0);
    store.psizam_set_stack_remaining(50);
    start.call(&mut store, ())?;
    assert_eq!(started.call(&mut store, ())?, 1);

    // A module without a start function has none to defer, and an engine
    // that does not meter runs it at instantiation as usual.
    let none = Module::new(&engine, module_bytes(RECURSE, &RECURSE_COSTS))?;
    let instance = Instance::new(&mut store, &none, &[])?;
    assert!(instance.psizam_deferred_start(&mut store).is_none());
    let plain = Engine::default();
    let module = Module::new(&plain, text)?;
    let mut store = Store::new(&plain, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    assert!(instance.psizam_deferred_start(&mut store).is_none());
    let started = instance.get_typed_func::<(), i32>(&mut store, "started")?;
    assert_eq!(started.call(&mut store, ())?, 1);
    Ok(())
}

#[test]
fn psizam_unfused_madd_rounds_the_product_before_adding() -> Result<()> {
    // With a = b = 1 + 2^-27 the exact product is 1 + 2^-26 + 2^-54, which
    // rounds to 1 + 2^-26. Adding c = -(1 + 2^-26) therefore gives 0 when the
    // product is rounded first and 2^-54 when the operation is fused.
    let text = r#"(module
        (func (export "madd") (param f64 f64 f64) (result f64)
            (f64x2.extract_lane 0 (f64x2.relaxed_madd
                (f64x2.splat (local.get 0)) (f64x2.splat (local.get 1)) (f64x2.splat (local.get 2)))))
        (func (export "nmadd") (param f64 f64 f64) (result f64)
            (f64x2.extract_lane 0 (f64x2.relaxed_nmadd
                (f64x2.splat (local.get 0)) (f64x2.splat (local.get 1)) (f64x2.splat (local.get 2)))))
    )"#;
    let a = 1.0 + 2f64.powi(-27);
    let c = 1.0 + 2f64.powi(-26);
    for (unfused, expected) in [(true, 0.0), (false, 2f64.powi(-54))] {
        let mut config = Config::new();
        config.psizam_unfused_madd(unfused);
        let engine = Engine::new(&config)?;
        let module = Module::new(&engine, text)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let madd = instance.get_typed_func::<(f64, f64, f64), f64>(&mut store, "madd")?;
        let nmadd = instance.get_typed_func::<(f64, f64, f64), f64>(&mut store, "nmadd")?;
        assert_eq!(madd.call(&mut store, (a, a, -c))?, expected);
        // -a * a + c is the same difference with the sign reversed.
        assert_eq!(nmadd.call(&mut store, (a, a, c))?, -expected);
    }
    Ok(())
}
