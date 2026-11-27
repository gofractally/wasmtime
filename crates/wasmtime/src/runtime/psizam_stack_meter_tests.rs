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
    for (budget, deepest) in [
        (100u64, 0i32),
        (199, 0),
        (200, 1),
        (1000, 9),
        (100_000, 999),
    ] {
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
    assert!(overflowed(
        Instance::new(&mut store, &eager, &[]).map(|_| ())
    ));

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

/// `text` assembled with one cost per function, for modules too large for
/// `module_bytes`' one-byte section size.
fn large_module_bytes(text: &str, costs: &[u32]) -> Vec<u8> {
    fn leb(mut n: usize, out: &mut Vec<u8>) {
        loop {
            let byte = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }
    let mut bytes = wat::parse_str(text).unwrap();
    let name = wasmtime_environ::PSIZAM_STACK_COSTS_SECTION.as_bytes();
    bytes.push(0);
    leb(1 + name.len() + 4 * costs.len(), &mut bytes);
    bytes.push(name.len() as u8);
    bytes.extend_from_slice(name);
    for cost in costs {
        bytes.extend_from_slice(&cost.to_le_bytes());
    }
    bytes
}

/// A function whose wasm operand stack never holds more than two values, but
/// which computes `values` pure values before a call and the same ones again
/// after it. An optimizer computes each once and keeps it across the call.
fn kept_across_a_call(values: usize) -> String {
    let mut body = String::new();
    for _round in 0..2 {
        for i in 0..values {
            body.push_str(&format!(
                "(global.set $g (i32.xor (global.get $g) \
                   (i32.mul (i32.rotl (local.get 0) (i32.const {})) (i32.const {}))))\n",
                i % 31 + 1,
                2 * i + 3
            ));
        }
        if _round == 0 {
            body.push_str(
                "(if (local.get 0) (then (call $f (i32.sub (local.get 0) (i32.const 1)))))\n",
            );
        }
    }
    format!(
        "(module (global $g (export \"g\") (mut i32) (i32.const 0))
           (func $f (export \"f\") (param i32) {body}))"
    )
}

fn engine_with_ratio(ratio: u32) -> Result<Engine> {
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    config.psizam_native_frame_ratio(ratio);
    config.collector(crate::Collector::Null);
    Engine::new(&config)
}

/// The native stack the test below gives wasm: 16 native bytes for each
/// byte of a 1 MiB budget, as an embedder that relies on the ratio would.
const RESERVED: usize = 16 << 20;

fn with_reserved_stack(config: &mut Config) -> &mut Config {
    config.async_stack_size(RESERVED + (1 << 20));
    config.max_wasm_stack(RESERVED)
}

#[test]
fn psizam_native_frame_ratio_holds_for_a_function_the_optimizer_inflates() -> Result<()> {
    // On a thread with room for the reservation and for the host's frames.
    std::thread::Builder::new()
        .stack_size(RESERVED + (8 << 20))
        .spawn(frame_ratio_holds)
        .unwrap()
        .join()
        .unwrap()
}

fn frame_ratio_holds() -> Result<()> {
    // 32 is what psizam charges this function: 16, a 4-byte parameter and a
    // deepest operand stack of 12 bytes.
    let wasm = large_module_bytes(&kept_across_a_call(2000), &[32]);
    let run = |engine: &Engine, depth: i32| -> Result<i32> {
        let module = Module::new(engine, &wasm)?;
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        store.psizam_set_stack_remaining(1 << 20);
        instance
            .get_typed_func::<i32, ()>(&mut store, "f")?
            .call(&mut store, depth)?;
        Ok(instance
            .get_global(&mut store, "g")
            .unwrap()
            .get(&mut store)
            .unwrap_i32())
    };

    // Unconstrained, the optimizer is free to keep all 2,000 values.
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    let unconstrained = run(&Engine::new(with_reserved_stack(&mut config))?, 50)?;

    // Constrained to 16 native bytes per cost byte, i.e. 512 bytes for this
    // function, it gets the determined frame and computes the same.
    let before = crate::psizam_unoptimized_for_frame();
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    config.psizam_native_frame_ratio(16);
    config.collector(crate::Collector::Null);
    let constrained = Engine::new(with_reserved_stack(&mut config))?;
    assert_eq!(run(&constrained, 50)?, unconstrained);
    assert!(crate::psizam_unoptimized_for_frame() > before);

    // With every frame at most 512 bytes, the whole budget fits 16 MiB of
    // native stack: the budget, not the native stack, ends this recursion.
    // (32,768 frames of 32 bytes are the budget; one more does not fit.)
    let module = Module::new(&constrained, &wasm)?;
    let mut store = Store::new(&constrained, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let f = instance.get_typed_func::<i32, ()>(&mut store, "f")?;
    store.psizam_set_stack_remaining(1 << 20);
    f.call(&mut store, 32_767)?;
    store.psizam_set_stack_remaining(1 << 20);
    assert!(overflowed(f.call(&mut store, 32_768)));

    // Unconstrained, the same depth inside the same budget does not fit the
    // same native stack: that is the failure the constraint excludes.
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    let engine = Engine::new(with_reserved_stack(&mut config))?;
    let module = Module::new(&engine, &wasm)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let f = instance.get_typed_func::<i32, ()>(&mut store, "f")?;
    store.psizam_set_stack_remaining(1 << 20);
    assert!(overflowed(f.call(&mut store, 32_767)));
    Ok(())
}

#[test]
fn psizam_native_frame_ratio_refuses_what_it_cannot_satisfy() -> Result<()> {
    // No frame is as small as its cost when the cost is 1. A ratio the
    // determined frame cannot meet is a misconfigured engine, not a property
    // of the module: it is reported as an internal error, and no code is
    // produced with a frame the embedder's reservation does not cover.
    let error = Module::new(
        &engine_with_ratio(1)?,
        module_bytes(RECURSE, &[1, 1, 1, 1, 1]),
    )
    .unwrap_err();
    let text = format!("{error:?}");
    assert!(
        text.contains("psizam: internal error: the determined native frame of"),
        "{text}"
    );

    // The same module with its real costs satisfies a ratio of 16 as it is.
    Module::new(
        &engine_with_ratio(16)?,
        module_bytes(RECURSE, &RECURSE_COSTS),
    )?;
    Ok(())
}

#[test]
fn psizam_native_frame_ratio_is_part_of_what_a_module_was_compiled_with() -> Result<()> {
    let wasm = module_bytes(RECURSE, &RECURSE_COSTS);
    let bytes = engine_with_ratio(16)?.precompile_module(&wasm)?;
    assert!(unsafe { Module::deserialize(&engine_with_ratio(16)?, &bytes) }.is_ok());
    assert!(unsafe { Module::deserialize(&engine_with_ratio(32)?, &bytes) }.is_err());
    assert!(unsafe { Module::deserialize(&engine()?, &bytes) }.is_err());
    Ok(())
}

/// Programs that use the operand stack and locals in the ways the determined
/// frame has to get right: values kept on the stack across calls and across
/// blocks, blocks, loops and ifs with parameters and results, branches out of
/// nested blocks carrying values, `br_table`, `select`, `local.tee`,
/// unreachable code followed by reachable code, multiple results, every
/// value type.
const SHAPES: &str = r#"(module
    (type $pair (func (param i32 i32) (result i32 i32)))
    (func $swap (type $pair) (local.get 1) (local.get 0))
    (func $id64 (param i64) (result i64) (local.get 0))
    (func (export "across_calls") (param i32) (result i32)
        (local.get 0) (i32.const 7)
        (call $swap (i32.const 1) (i32.const 2))
        (i32.sub) (i32.add) (i32.add))
    (func (export "fib") (param i32) (result i32)
        (if (result i32) (i32.lt_u (local.get 0) (i32.const 2))
            (then (local.get 0))
            (else (i32.add
                (call 3 (i32.sub (local.get 0) (i32.const 1)))
                (call 3 (i32.sub (local.get 0) (i32.const 2)))))))
    (func (export "loop_params") (param i32) (result i32)
        (i32.const 0) (local.get 0)
        (loop $again (param i32 i32) (result i32)
            (local.set 0)
            (local.get 0) (i32.add)
            (local.get 0) (i32.const 1) (i32.sub)
            (local.tee 0)
            (br_if $again (local.get 0))
            (drop)))
    (func (export "if_params") (param i32 i32) (result i32)
        (local.get 0) (local.get 1)
        (i32.gt_s (local.get 0) (local.get 1))
        (if (param i32 i32) (result i32)
            (then (i32.sub))
            (else (call $swap) (i32.sub))))
    (func (export "br_values") (param i32) (result i32)
        (i32.const 100)
        (block $out (result i32)
            (i32.const 1)
            (block $in (result i32)
                (i32.const 2)
                (br_if $out (i32.const 40) (local.get 0))
                (drop)
                (br $in (i32.const 50))
                (unreachable))
            (i32.add))
        (i32.add))
    (func (export "table") (param i32) (result i32)
        (block $c (result i32) (block $b (result i32) (block $a (result i32)
            (i32.const 5)
            (br_table $a $b $c (local.get 0)))
            (i32.const 10) (i32.add))
            (i32.const 100) (i32.add)))
    (func (export "select_tee") (param i32 i64) (result i64) (local i64)
        (local.tee 2 (i64.mul (local.get 1) (i64.const 3)))
        (call $id64 (local.get 1))
        (select (local.get 0))
        (local.get 2) (i64.add))
    (func (export "floats") (param f32 f64) (result f64) (local f32)
        (local.set 2 (f32.mul (local.get 0) (f32.const 2)))
        (f64.promote_f32 (local.get 2))
        (call $id64 (i64.const 0)) (drop)
        (local.get 1) (f64.add))
    (func (export "vectors") (param i32) (result i32) (local v128)
        (local.set 1 (i32x4.splat (local.get 0)))
        (local.get 1)
        (call $id64 (i64.const 0)) (drop)
        (i32x4.add (i32x4.splat (i32.const 1)))
        (i32x4.extract_lane 2))
    (func (export "early_return") (param i32) (result i32)
        (i32.const 9)
        (if (local.get 0) (then (return (i32.const 1))))
        (i32.const 3) (i32.mul))
    (memory 1)
    (func (export "memory") (param i32) (result i32)
        (i32.store (i32.const 8) (local.get 0))
        (i32.load (i32.const 8)) (i32.load8_u (i32.const 8)) (i32.add))
)"#;
const SHAPES_COSTS: [u32; 13] = [64; 13];

#[test]
fn psizam_determined_frames_compute_what_optimized_code_computes() -> Result<()> {
    let wasm = module_bytes(SHAPES, &SHAPES_COSTS);
    let mut results = Vec::new();
    for determined in [false, true] {
        let mut config = Config::new();
        config.psizam_stack_meter(true);
        config.psizam_determined_frames(determined);
        config.collector(crate::Collector::Null);
        let engine = Engine::new(&config)?;
        let module = Module::new(&engine, &wasm)?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        store.psizam_set_stack_remaining(1 << 16);
        let mut out = Vec::new();
        macro_rules! call {
            ($name:literal, $args:ty, $ret:ty, $($value:expr),*) => {{
                let f = instance.get_typed_func::<$args, $ret>(&mut store, $name)?;
                $( out.push(format!("{} {:?}", $name, f.call(&mut store, $value)?)); )*
            }};
        }
        call!("across_calls", i32, i32, 0, 5, -3);
        call!("fib", i32, i32, 0, 1, 2, 10, 15);
        call!("loop_params", i32, i32, 1, 2, 10);
        call!("if_params", (i32, i32), i32, (5, 2), (2, 5), (3, 3));
        call!("br_values", i32, i32, 0, 1);
        call!("table", i32, i32, 0, 1, 2, 7);
        call!("select_tee", (i32, i64), i64, (0, 5), (1, 5));
        call!("floats", (f32, f64), f64, (1.5, 2.25));
        call!("vectors", i32, i32, 41);
        call!("early_return", i32, i32, 0, 1);
        call!("memory", i32, i32, 0x1234);
        results.push(out);
    }
    assert_eq!(results[0], results[1]);
    // And they are what the functions mean.
    assert!(
        results[1].contains(&"fib 610".to_string()),
        "{:?}",
        results[1]
    );
    assert!(
        results[1].contains(&"loop_params 55".to_string()),
        "{:?}",
        results[1]
    );
    Ok(())
}

/// The determined frame's slots are not traced, so the constraint is refused
/// with a collector that collects.
#[test]
fn psizam_native_frame_ratio_requires_that_nothing_is_collected() {
    for collector in [
        crate::Collector::DeferredReferenceCounting,
        crate::Collector::Copying,
    ] {
        let mut config = Config::new();
        config.psizam_stack_meter(true);
        config.psizam_native_frame_ratio(16);
        config.collector(collector);
        let error = Engine::new(&config).err().expect("refused");
        assert!(error.to_string().contains("Collector::Null"), "{error}");
    }
    let mut config = Config::new();
    config.psizam_stack_meter(true);
    config.psizam_native_frame_ratio(16);
    config.gc_support(false);
    Engine::new(&config).expect("no GC support");
}
