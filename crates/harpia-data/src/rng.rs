//! A tiny deterministic PRNG (SplitMix64) used for shuffling, splitting and sampling.
//!
//! Harpia ships its own generator so that a seed gives the same result on every platform and
//! every Harpia version, independently of `rand` crate releases.

/// SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeded generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Internal state (to checkpoint and resume a run exactly).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Generator continuing from a saved [`state`](Self::state).
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..n` (`n > 0`), without modulo bias.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0)");
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < zone {
                return x % n;
            }
        }
    }

    /// Fisher-Yates shuffle.
    ///
    /// ```
    /// use harpia_data::rng::SplitMix64;
    /// let mut a: Vec<u32> = (0..10).collect();
    /// let mut b = a.clone();
    /// SplitMix64::new(42).shuffle(&mut a);
    /// SplitMix64::new(42).shuffle(&mut b);
    /// assert_eq!(a, b);
    /// assert_ne!(a, (0..10).collect::<Vec<_>>());
    /// ```
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_first_value() {
        // reference value of SplitMix64 with seed 0
        assert_eq!(SplitMix64::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn floats_in_range() {
        let mut r = SplitMix64::new(1);
        assert!((0..1000).map(|_| r.next_f64()).all(|x| (0.0..1.0).contains(&x)));
    }
}
