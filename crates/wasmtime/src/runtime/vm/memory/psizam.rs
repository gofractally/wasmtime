//! Compact, zero-filled private backing for memory16-v1.

use crate::LinearMemory;
use crate::prelude::*;
use crate::runtime::vm::SendSyncPtr;
use core::ptr::NonNull;
use wasmtime_environ::{PSIZAM_MEMORY16_ACCESSIBLE_BYTES, PSIZAM_MEMORY16_LOGICAL_BYTES};

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Aligned([u8; 16]);

pub(super) struct Memory16Memory {
    _storage: Vec<Aligned>,
    base: SendSyncPtr<u8>,
}

impl Memory16Memory {
    pub fn new() -> Result<Self> {
        let elements = PSIZAM_MEMORY16_ACCESSIBLE_BYTES.div_ceil(16);
        let mut storage = Vec::new();
        storage.try_reserve_exact(elements)?;
        // Initialize the entire accessible allocation, including padding.
        storage.resize(elements, Aligned([0; 16]));
        let base = SendSyncPtr::new(NonNull::new(storage.as_mut_ptr()).unwrap()).cast();
        Ok(Self {
            _storage: storage,
            base,
        })
    }
}

// SAFETY: storage is owned, initialized, 16-aligned and immovable. Only the
// logical page is reported as the current length. The private profile admits
// precisely the additional bytes reported as capacity; no guard reservation,
// custom allocator or CoW image contract is used for this allocation.
unsafe impl LinearMemory for Memory16Memory {
    fn byte_size(&self) -> usize {
        PSIZAM_MEMORY16_LOGICAL_BYTES
    }

    fn byte_capacity(&self) -> usize {
        PSIZAM_MEMORY16_ACCESSIBLE_BYTES
    }

    fn grow_to(&mut self, new_size: usize) -> Result<()> {
        if new_size != PSIZAM_MEMORY16_LOGICAL_BYTES {
            bail!("memory16 cannot grow");
        }
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_state_is_zeroed_writable_and_logically_fixed() -> Result<()> {
        let mut memory = Memory16Memory::new()?;
        assert_eq!(memory.byte_size(), 65536);
        assert_eq!(memory.byte_capacity(), 131086);
        // SAFETY: the allocation owns and initializes the complete state.
        let state =
            unsafe { core::slice::from_raw_parts_mut(memory.as_ptr(), memory.byte_capacity()) };
        assert!(state.iter().all(|byte| *byte == 0));
        state[65536] = 42;
        state[131085] = 99;
        assert_eq!(state[65536], 42);
        assert_eq!(state[131085], 99);
        assert!(memory.grow_to(131072).is_err());
        assert_eq!(memory.byte_size(), 65536);
        let fresh = Memory16Memory::new()?;
        // SAFETY: the fresh allocation initializes its complete private state.
        assert!(
            unsafe { core::slice::from_raw_parts(fresh.as_ptr(), fresh.byte_capacity()) }
                .iter()
                .all(|byte| *byte == 0)
        );
        Ok(())
    }
}
