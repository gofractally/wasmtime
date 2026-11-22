use crate::prelude::*;
use crate::{
    Config, Engine, Extern, Instance, LinearMemory, Memory, MemoryCreator, MemoryRequest,
    MemoryType, Module, Store,
};
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};
use wasmtime_environ::{
    PSIZAM_MEMORY16_ACCESSIBLE_BYTES, PSIZAM_MEMORY16_P16, PSIZAM_MEMORY16_P32,
};

fn module_bytes(text: &str, profile: Option<u32>) -> Vec<u8> {
    let mut bytes = wat::parse_str(text).unwrap();
    if let Some(profile) = profile {
        bytes.extend_from_slice(&[0, 21, 16]);
        bytes.extend_from_slice(b"psizam.execution");
        bytes.extend_from_slice(&profile.to_le_bytes());
    }
    bytes
}

fn engine(force: bool) -> Result<Engine> {
    let mut config = Config::new();
    config.psizam_memory16(force);
    Engine::new(&config)
}

#[test]
fn psizam_memory16_logical_host_view_and_private_state() -> Result<()> {
    let engine = engine(false)?;
    let bytes = module_bytes(
        "(module (memory (export \"memory\") 1 1) (data (i32.const 65533) \"abc\"))",
        Some(PSIZAM_MEMORY16_P16),
    );
    let module = Module::new(&engine, bytes)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    assert_eq!(memory.data_size(&store), 65536);
    assert_eq!(&memory.data(&store)[65533..], b"abc");
    assert_eq!(
        memory.psizam_memory16_data_size(&store),
        Some(PSIZAM_MEMORY16_ACCESSIBLE_BYTES)
    );
    assert!(memory.write(&mut store, 65536, &[1]).is_err());
    // SAFETY: actual profile backing owns and initializes all private bytes;
    // no Wasm call or borrowed host view is active.
    let state = unsafe {
        core::slice::from_raw_parts_mut(memory.data_ptr(&store), PSIZAM_MEMORY16_ACCESSIBLE_BYTES)
    };
    assert!(state[65536..].iter().all(|byte| *byte == 0));
    state[131085] = 99;
    assert_eq!(state[131085], 99);
    assert!(memory.grow(&mut store, 1).is_err());
    assert_eq!(memory.size(&store), 1);
    Ok(())
}

#[test]
fn psizam_memory16_saved_profile_and_override_admission() -> Result<()> {
    let normal = engine(false)?;
    let forced = engine(true)?;
    let bytes = module_bytes("(module (memory (export \"memory\") 1 1))", None);
    let standard = Module::new(&normal, &bytes)?.serialize()?;
    // SAFETY: these are artifacts just produced by this exact fork.
    assert!(unsafe { Module::deserialize(&forced, &standard) }.is_err());
    let compiled = Module::new(&forced, &bytes)?.serialize()?;
    let restored = unsafe { Module::deserialize(&normal, &compiled) }?;
    let mut store = Store::new(&normal, ());
    let instance = Instance::new(&mut store, &restored, &[])?;
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    assert_eq!(memory.psizam_memory16_data_size(&store), Some(131086));
    let tagged = Module::new(
        &normal,
        module_bytes("(module (memory 1 1))", Some(PSIZAM_MEMORY16_P16)),
    )?
    .serialize()?;
    assert!(unsafe { Module::deserialize(&forced, &tagged) }.is_ok());
    Ok(())
}

#[test]
fn psizam_memory16_import_requires_explicit_capable_backing() -> Result<()> {
    let engine = engine(false)?;
    let module = Module::new(
        &engine,
        module_bytes(
            "(module (import \"env\" \"memory\" (memory 1 1)))",
            Some(PSIZAM_MEMORY16_P16),
        ),
    )?;
    let mut store = Store::new(&engine, ());
    let ordinary = Memory::new(&mut store, MemoryType::new(1, Some(1))?)?;
    assert!(Instance::new(&mut store, &module, &[Extern::Memory(ordinary)]).is_err());
    let capable = MemoryType::builder()
        .min(1)
        .max(Some(1))
        .psizam_memory16(16)
        .build()?;
    let capable = Memory::new(&mut store, capable)?;
    assert_eq!(capable.psizam_memory16_data_size(&store), Some(131086));
    assert!(Instance::new(&mut store, &module, &[Extern::Memory(capable)]).is_ok());
    assert!(
        MemoryType::builder()
            .min(1)
            .max(Some(1))
            .psizam_memory16(8)
            .build()
            .is_err()
    );
    Ok(())
}

#[test]
fn psizam_memory16_rejects_custom_creator_before_allocation() -> Result<()> {
    struct Custom(Arc<AtomicUsize>);
    // SAFETY: no memory is ever provided by this deliberately rejecting creator.
    unsafe impl MemoryCreator for Custom {
        fn new_memory(&self, _: &MemoryRequest<'_>) -> Result<Box<dyn LinearMemory>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            bail!("test creator must never be called for memory16")
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let mut config = Config::new();
    config.with_host_memory(Arc::new(Custom(calls.clone())));
    let engine = Engine::new(&config)?;
    assert!(
        Module::new(
            &engine,
            module_bytes("(module (memory 1 1))", Some(PSIZAM_MEMORY16_P16))
        )
        .is_err()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn psizam_memory16_actual_unexported_memory_during_start() -> Result<()> {
    for (force, profile, expected) in [
        (false, None, false),
        (false, Some(PSIZAM_MEMORY16_P16), true),
        (false, Some(PSIZAM_MEMORY16_P32), true),
        (true, None, true),
        (true, Some(PSIZAM_MEMORY16_P16), true),
        (true, Some(PSIZAM_MEMORY16_P32), true),
    ] {
        let engine = engine(force)?;
        let bytes = module_bytes(
            r#"(module
                (import "env" "observe" (func $observe))
                (memory 1 1)
                (data (i32.const 0) "*")
                (func $start call $observe)
                (start $start)
                (func (export "again") call $observe))"#,
            profile,
        );
        let module = Module::new(&engine, bytes)?;
        let mut store = Store::new(&engine, Vec::<Option<(usize, usize, u8)>>::new());
        let observe = crate::Func::wrap(
            &mut store,
            |mut caller: crate::Caller<'_, Vec<Option<(usize, usize, u8)>>>| {
                assert!(caller.get_export("memory").is_none());
                let record = caller.psizam_memory16_memory().map(|memory| {
                    (
                        memory.data_size(&caller),
                        memory.psizam_memory16_data_size(&caller).unwrap(),
                        memory.data(&caller)[0],
                    )
                });
                caller.data_mut().push(record);
            },
        );
        let instance = Instance::new(&mut store, &module, &[observe.into()])?;
        assert!(instance.get_memory(&mut store, "memory").is_none());
        let memory = instance.psizam_memory16_memory(&store);
        assert_eq!(memory.is_some(), expected);
        if let Some(memory) = memory {
            assert_eq!(memory.data_size(&store), 65536);
            assert_eq!(memory.psizam_memory16_data_size(&store), Some(131086));
            assert_eq!(memory.data(&store)[0], 42);
        }
        instance
            .get_typed_func::<(), ()>(&mut store, "again")?
            .call(&mut store, ())?;
        let wanted = expected.then_some((65536, 131086, 42));
        assert_eq!(store.data().as_slice(), &[wanted, wanted]);
    }
    Ok(())
}

#[cfg(feature = "winch")]
#[test]
fn psizam_memory16_rejects_unsupported_winch_compilation() -> Result<()> {
    let mut config = Config::new();
    config.strategy(crate::Strategy::Winch);
    let engine = Engine::new(&config)?;
    let tagged = module_bytes("(module (memory 1 1))", Some(PSIZAM_MEMORY16_P16));
    assert!(Module::validate(&engine, &tagged).is_err());
    assert!(Module::new(&engine, &tagged).is_err());
    let untagged = module_bytes("(module (memory 1 1))", None);
    assert!(Module::new(&engine, &untagged).is_ok());
    config.psizam_memory16(true);
    let forced = Engine::new(&config)?;
    assert!(Module::validate(&forced, &untagged).is_err());
    assert!(Module::new(&forced, &untagged).is_err());
    Ok(())
}
