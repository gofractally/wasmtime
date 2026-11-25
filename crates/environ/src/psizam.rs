//! Admission for psizam's explicitly selected memory16-v1 execution profile,
//! and the per-function costs of psizam's deterministic stack budget.
//!
//! # The stack budget
//!
//! How much native stack a wasm frame takes depends on the compiler, so the
//! native stack limit cannot decide which call overflows on a chain where
//! every node must agree. psizam charges each function a fixed number of
//! bytes computed from the wasm itself. With `Tunables::psizam_stack_meter`
//! the embedder supplies those costs in a `psizam.stack-costs` custom section
//! (one little-endian `u32` per defined function, in order) and compiled code
//! enforces them:
//!
//! * every wasm function takes the bytes remaining as one extra, trailing
//!   argument, traps with a stack overflow if that is less than its own
//!   cost, and passes the difference to whatever it calls;
//! * a host-to-wasm entry trampoline reads the starting value from
//!   `VMStoreContext::psizam_stack_remaining` and puts it back when the call
//!   returns;
//! * a wasm-to-host trampoline stores the value its caller passed, so that
//!   the host sees what remains and may set what a re-entrant call starts
//!   with.
//!
//! Nothing is loaded or stored on a wasm-to-wasm call or on a return.

use crate::error::{Result, bail};
use crate::{DefinedFuncIndex, Memory, PSIZAM_MEMORY16_P16, PSIZAM_MEMORY16_P32, PrimaryMap};
use wasmparser::{MemArgTarget, Operator, Parser, Payload, TypeRef};

/// Resolve and validate a core module's declared/default execution profile.
///
/// This prepass sees custom sections after code and checks offsets even in
/// unreachable code, before any compiler can discard those instructions.
pub fn psizam_execution_profile(data: &[u8], force_memory16: bool) -> Result<u32> {
    let mut declared = None;
    let mut memory_count = 0usize;
    let mut memory = None;
    for payload in Parser::new(0).parse_all(data) {
        match payload? {
            Payload::CustomSection(section) if section.name() == "psizam.execution" => {
                let data = section.data();
                if data.is_empty() || data.len() % 4 != 0 {
                    bail!("malformed psizam.execution declaration");
                }
                for record in data.chunks_exact(4) {
                    let profile = u32::from_le_bytes(record.try_into().unwrap());
                    if !matches!(profile, PSIZAM_MEMORY16_P16 | PSIZAM_MEMORY16_P32) {
                        bail!("unsupported psizam.execution declaration");
                    }
                    if declared.is_some_and(|old| old != profile) {
                        bail!("conflicting psizam.execution declarations");
                    }
                    declared = Some(profile);
                }
            }
            Payload::MemorySection(memories) => {
                for ty in memories {
                    memory_count += 1;
                    memory = Some(ty?);
                }
            }
            Payload::ImportSection(imports) => {
                for import in imports.into_imports() {
                    if let TypeRef::Memory(ty) = import?.ty {
                        memory_count += 1;
                        memory = Some(ty);
                    }
                }
            }
            _ => {}
        }
    }
    let profile = declared.unwrap_or(if force_memory16 {
        PSIZAM_MEMORY16_P32
    } else {
        0
    });
    if profile == 0 {
        return Ok(0);
    }
    if memory_count != 1 {
        bail!("memory16 requires exactly one memory");
    }
    let raw = memory.unwrap();
    if raw.shared
        || raw.memory64
        || raw.initial != 1
        || raw.maximum != Some(1)
        || raw.page_size_log2.unwrap_or(16) != 16
    {
        bail!("memory16 requires one nonshared memory32 with initial=max=1 page");
    }
    let mut memory = Memory::from(raw);
    memory.psizam_execution_profile = profile;
    memory.validate_psizam_execution_profile()?;
    for payload in Parser::new(0).parse_all(data) {
        if let Payload::CodeSectionEntry(body) = payload? {
            let mut operators = body.get_operators_reader()?;
            while !operators.eof() {
                check_operator(&operators.read()?)?;
            }
        }
    }
    Ok(profile)
}

fn check_operator(op: &Operator<'_>) -> Result<()> {
    macro_rules! check_field {
        (memarg, $value:ident) => {
            if $value.offset > 65535 || !matches!($value.target, MemArgTarget::Memory(0)) {
                bail!("memory16 requires memory0 instruction offsets <=65535");
            }
        };
        ($other:ident, $value:ident) => {
            let _ = $value;
        };
    }
    macro_rules! check_all {
        ($( @$proposal:ident $name:ident $({ $($arg:ident: $argty:ty),* })? => $visit:ident ($($ann:tt)*))*) => {
            match op {
                $(Operator::$name $({ $($arg),* })? => { $($(check_field!($arg, $arg);)*)? })*
                _ => bail!("unsupported memory16 operator"),
            }
        };
    }
    wasmparser::for_each_operator!(check_all);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wasm(text: &str, records: &[u32]) -> alloc::vec::Vec<u8> {
        let mut wasm = wat::parse_str(text).unwrap();
        for profile in records {
            wasm.extend_from_slice(&[0, 21, 16]);
            wasm.extend_from_slice(b"psizam.execution");
            wasm.extend_from_slice(&profile.to_le_bytes());
        }
        wasm
    }

    #[test]
    fn explicit_default_and_duplicate_profiles() {
        let text = "(module (memory 1 1))";
        assert_eq!(
            psizam_execution_profile(&wasm(text, &[]), false).unwrap(),
            0
        );
        assert_eq!(
            psizam_execution_profile(&wasm(text, &[]), true).unwrap(),
            PSIZAM_MEMORY16_P32
        );
        assert_eq!(
            psizam_execution_profile(&wasm(text, &[PSIZAM_MEMORY16_P16; 2]), true).unwrap(),
            PSIZAM_MEMORY16_P16
        );
        assert!(
            psizam_execution_profile(
                &wasm(text, &[PSIZAM_MEMORY16_P16, PSIZAM_MEMORY16_P32]),
                false
            )
            .is_err()
        );
        assert!(psizam_execution_profile(&wasm(text, &[0]), false).is_err());
    }

    #[test]
    fn validates_memory_and_unreachable_offsets() {
        for text in [
            "(module)",
            "(module (memory 0 1))",
            "(module (memory 1))",
            "(module (memory i64 1 1))",
            "(module (memory 1 1 shared))",
            "(module (memory 1 1) (memory 1 1))",
        ] {
            assert!(
                psizam_execution_profile(&wasm(text, &[]), true).is_err(),
                "{text}"
            );
        }
        for operator in [
            "i32.load offset=65536 drop",
            "v128.load offset=65536 drop",
            "i32.atomic.load offset=65536 drop",
        ] {
            let module = wasm(
                &alloc::format!("(module (memory 1 1) (func unreachable {operator}))"),
                &[PSIZAM_MEMORY16_P32],
            );
            assert!(
                psizam_execution_profile(&module, false).is_err(),
                "{operator}"
            );
        }
        assert!(
            psizam_execution_profile(
                &wasm(
                    "(module (memory 1 1) (func unreachable i32.load offset=65535 drop))",
                    &[PSIZAM_MEMORY16_P32]
                ),
                false
            )
            .is_ok()
        );
    }
}

/// The name of the custom section that carries psizam's stack costs.
pub const PSIZAM_STACK_COSTS_SECTION: &str = "psizam.stack-costs";

/// Reads the stack cost of each of a module's `defined` functions.
///
/// The section must be present exactly once and hold exactly one cost per
/// defined function: a module compiled for metering never runs unmetered.
pub fn psizam_stack_costs(data: &[u8], defined: usize) -> Result<PrimaryMap<DefinedFuncIndex, u32>> {
    let mut costs = None;
    for payload in Parser::new(0).parse_all(data) {
        if let Payload::CustomSection(section) = payload? {
            if section.name() != PSIZAM_STACK_COSTS_SECTION {
                continue;
            }
            if costs.is_some() {
                bail!("duplicate psizam.stack-costs section");
            }
            let bytes = section.data();
            if bytes.len() % 4 != 0 || bytes.len() / 4 != defined {
                bail!("psizam.stack-costs does not have one cost per defined function");
            }
            let mut map = PrimaryMap::with_capacity(defined);
            for cost in bytes.chunks_exact(4) {
                map.push(u32::from_le_bytes(cost.try_into().unwrap()));
            }
            costs = Some(map);
        }
    }
    match costs {
        Some(costs) => Ok(costs),
        None => bail!("stack metering requires a psizam.stack-costs section"),
    }
}
