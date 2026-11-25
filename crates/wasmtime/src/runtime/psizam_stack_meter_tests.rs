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
