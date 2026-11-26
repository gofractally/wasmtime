//! Cooperative cancellation of a compilation: [`CodeBuilder::cancel_handle`].

use crate::prelude::*;
use crate::{
    CodeBuilder, CompileCancel, CompileCancelled, Config, Engine, Instance, Module, Store,
};
use std::time::{Duration, Instant};

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

fn section(id: u8, body: &[u8], out: &mut Vec<u8>) {
    out.push(id);
    leb(body.len(), out);
    out.extend_from_slice(body);
}

/// A module of `bodies.len()` functions of type `(i32) -> i64`, the first
/// exported as `f`.
fn module(bodies: &[&[u8]]) -> Vec<u8> {
    let mut out = b"\0asm\x01\0\0\0".to_vec();
    section(1, &[1, 0x60, 1, 0x7f, 1, 0x7e], &mut out);
    let mut funcs = Vec::new();
    leb(bodies.len(), &mut funcs);
    funcs.resize(funcs.len() + bodies.len(), 0);
    section(3, &funcs, &mut out);
    section(7, &[1, 1, b'f', 0, 0], &mut out);
    let mut code = Vec::new();
    leb(bodies.len(), &mut code);
    for body in bodies {
        leb(body.len(), &mut code);
        code.extend_from_slice(body);
    }
    section(10, &code, &mut out);
    out
}

/// `count` functions that each return a constant: nothing in any one of them
/// takes time, the module does.
fn many_functions(count: usize) -> Vec<u8> {
    let body: &[u8] = &[0, 0x42, 7, 0x0b];
    module(&vec![body; count])
}

/// A small module with some arithmetic, a loop and a call.
const SMALL: &str = r#"(module
    (func $sum (param i32) (result i64)
        (local $i i32) (local $acc i64)
        (block $done
            (loop $next
                (br_if $done (i32.ge_u (local.get $i) (local.get 0)))
                (local.set $acc (i64.add (local.get $acc)
                    (i64.mul (i64.extend_i32_u (local.get $i)) (i64.const 3))))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $next)))
        (local.get $acc))
    (func (export "f") (param i32) (result i64)
        (i64.add (call $sum (local.get 0)) (i64.const 1))))"#;

fn engine(parallel: bool) -> Result<Engine> {
    let mut config = Config::new();
    #[cfg(feature = "parallel-compilation")]
    config.parallel_compilation(parallel);
    let _ = parallel;
    Engine::new(&config)
}

fn builder<'a>(engine: &'a Engine, wasm: &'a [u8], cancel: &CompileCancel) -> CodeBuilder<'a> {
    let mut builder = CodeBuilder::new(engine);
    builder.wasm_binary(wasm, None).unwrap();
    builder.cancel_handle(cancel);
    builder
}

#[track_caller]
fn assert_cancelled<T>(result: Result<T>) {
    match result {
        Ok(_) => panic!("the compilation completed"),
        Err(error) => assert!(
            error.is::<CompileCancelled>(),
            "not a cancellation: {error:?}"
        ),
    }
}

/// Compiles `wasm` on this thread while another thread sets the handle
/// `delay` after the start; returns the result and how long after the handle
/// was set the compilation returned.
fn cancel_after(engine: &Engine, wasm: &[u8], delay: Duration) -> (Result<Module>, Duration) {
    let cancel = CompileCancel::new();
    let builder = builder(engine, wasm, &cancel);
    std::thread::scope(|scope| {
        let setter = scope.spawn(|| {
            std::thread::sleep(delay);
            cancel.cancel();
            Instant::now()
        });
        let result = builder.compile_module();
        let returned = Instant::now();
        let set = setter.join().unwrap();
        (result, returned.saturating_duration_since(set))
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn cancelled_before_it_starts() -> Result<()> {
    let engine = engine(true)?;
    let wasm = wat::parse_str(SMALL)?;
    let cancel = CompileCancel::new();
    cancel.cancel();
    assert!(cancel.is_cancelled());

    let builder = builder(&engine, &wasm, &cancel);
    assert_cancelled(builder.compile_module());
    assert_cancelled(builder.compile_module_serialized());

    // The error is the bare marker, whatever noticed the handle.
    let error = builder.compile_module().unwrap_err();
    assert_eq!(error.to_string(), "compilation was cancelled");
    assert!(error.downcast_ref::<CompileCancelled>().is_some());

    // Not even an invalid module is looked at.
    let garbage = builder_garbage(&engine, &cancel);
    assert_cancelled(garbage);
    Ok(())
}

fn builder_garbage(engine: &Engine, cancel: &CompileCancel) -> Result<Module> {
    let mut builder = CodeBuilder::new(engine);
    builder.wasm_binary(b"\0asm\x01\0\0\0\xff\xff", None)?;
    builder.cancel_handle(cancel);
    builder.compile_module()
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_handle_that_is_never_set_changes_nothing() -> Result<()> {
    for parallel in [false, true] {
        let engine = engine(parallel)?;
        for wasm in [wat::parse_str(SMALL)?, many_functions(300)] {
            let plain = engine.precompile_module(&wasm)?;
            let cancel = CompileCancel::new();
            let with_handle = builder(&engine, &wasm, &cancel).compile_module_serialized()?;
            assert!(plain == with_handle, "the serialized module differs");
            assert!(!cancel.is_cancelled());
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn many_functions_cancelled_midway() -> Result<()> {
    for parallel in [false, true] {
        let engine = engine(parallel)?;
        // Enough functions that the handle is set long before the last of
        // them would have been compiled, in any build profile.
        let wasm = many_functions(if parallel { 200_000 } else { 60_000 });
        // Early (while the module is being parsed or the first functions are
        // compiled) and later (well into the functions).
        for delay in [5, 40, 150] {
            let (result, latency) = cancel_after(&engine, &wasm, Duration::from_millis(delay));
            assert_cancelled(result);
            // Each function is tiny, so the compilation is back as soon as
            // the functions in flight are done. The bound is loose for slow,
            // loaded test machines; what it excludes is running the module
            // to the end.
            assert!(
                latency < Duration::from_secs(2),
                "returned {latency:?} after a cancellation at {delay} ms (parallel: {parallel})"
            );
        }
    }
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn compiles_again_after_a_cancellation() -> Result<()> {
    for parallel in [false, true] {
        let engine = engine(parallel)?;
        let small = wat::parse_str(SMALL)?;
        let reference = engine.precompile_module(&small)?;

        // One cancelled before it started, one cancelled half-way.
        let cancel = CompileCancel::new();
        cancel.cancel();
        assert_cancelled(builder(&engine, &small, &cancel).compile_module());
        let big = many_functions(60_000);
        let (result, _) = cancel_after(&engine, &big, Duration::from_millis(10));
        assert_cancelled(result);

        // The engine compiles the same bytes, and others, as if nothing had
        // happened, with a new handle or with none ...
        let cancel = CompileCancel::new();
        let again = builder(&engine, &small, &cancel).compile_module_serialized()?;
        assert!(reference == again, "the serialized module differs");
        assert!(reference == engine.precompile_module(&small)?);

        // ... and the module runs.
        let module = builder(&engine, &small, &cancel).compile_module()?;
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[])?;
        let f = instance.get_typed_func::<i32, i64>(&mut store, "f")?;
        assert_eq!(f.call(&mut store, 10)?, 136);

        // A handle that was set stays set, for every builder that holds it.
        let (result, _) = cancel_after(&engine, &big, Duration::from_millis(10));
        assert_cancelled(result);
        let module = Module::new(&engine, &many_functions(1_000))?;
        assert!(module.get_export("f").is_some());
    }
    Ok(())
}

#[test]
#[cfg(feature = "cache")]
#[cfg_attr(miri, ignore)]
fn a_cancelled_compilation_is_not_cached() -> Result<()> {
    use crate::{Cache, CacheConfig};

    fn entries(dir: &std::path::Path) -> usize {
        let mut count = 0;
        let mut pending = vec![dir.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    pending.push(entry.path());
                } else {
                    count += 1;
                }
            }
        }
        count
    }

    let dir = tempfile::tempdir()?;
    let mut cache_config = CacheConfig::new();
    cache_config.with_directory(dir.path());
    let mut config = Config::new();
    config.cache(Some(Cache::new(cache_config)?));
    let engine = Engine::new(&config)?;

    let big = many_functions(60_000);
    let (result, _) = cancel_after(&engine, &big, Duration::from_millis(10));
    assert_cancelled(result);
    let cancel = CompileCancel::new();
    cancel.cancel();
    assert_cancelled(builder(&engine, &big, &cancel).compile_module());
    assert_eq!(entries(dir.path()), 0, "a cancelled compilation was cached");
    assert_eq!(engine.config().cache.as_ref().unwrap().cache_hits(), 0);

    // The handle is not part of the cache key: a module compiled with one is
    // found by a compilation without one.
    let small = wat::parse_str(SMALL)?;
    let cancel = CompileCancel::new();
    builder(&engine, &small, &cancel).compile_module()?;
    assert!(
        entries(dir.path()) > 0,
        "a completed compilation was not cached"
    );
    Module::new(&engine, &small)?;
    assert_eq!(engine.config().cache.as_ref().unwrap().cache_hits(), 1);
    Ok(())
}
