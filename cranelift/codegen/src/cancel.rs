//! Cooperative cancellation of a compilation.
//!
//! Compiling one function is normally quick, but its cost is not bounded by
//! the size of its input: some passes, register allocation above all, can
//! take seconds or longer on adversarial functions. An embedder that has
//! stopped waiting for a compilation hands in a flag and sets it; the
//! compiler looks at the flag between passes and inside its long loops and
//! then gives up with [`CodegenError::Cancelled`].
//!
//! The flag is passed down by reference. Looking at it is one relaxed atomic
//! load; a compilation whose flag is never set does exactly what it would do
//! without one.

use crate::result::{CodegenError, CodegenResult};
use core::sync::atomic::{AtomicBool, Ordering};

/// A flag that is never set, for compilations that cannot be cancelled.
static NEVER: AtomicBool = AtomicBool::new(false);

/// A borrowed cancellation flag; see the [module documentation](self).
#[derive(Clone, Copy, Debug)]
pub struct Cancel<'a>(&'a AtomicBool);

impl<'a> Cancel<'a> {
    /// Cancellation is requested by storing `true` in `flag`, from any
    /// thread and with any ordering.
    pub fn new(flag: &'a AtomicBool) -> Self {
        Cancel(flag)
    }

    /// A compilation that is never cancelled.
    pub fn never() -> Cancel<'static> {
        Cancel(&NEVER)
    }

    /// Has cancellation been requested?
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Returns `Err(CodegenError::Cancelled)` if cancellation has been
    /// requested.
    #[inline]
    pub fn check(&self) -> CodegenResult<()> {
        if self.is_cancelled() {
            Err(CodegenError::Cancelled)
        } else {
            Ok(())
        }
    }

    /// The flag itself.
    pub fn flag(&self) -> &'a AtomicBool {
        self.0
    }
}
