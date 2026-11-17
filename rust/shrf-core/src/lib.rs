//! Batch-one CPU decoder for the Python SHRF equations. No external dependencies.
use std::slice;

#[no_mangle]
pub extern "C" fn shrf_abi_version() -> u32 {
    5
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Config {
    d: usize,
    trees: usize,
    depth: usize,
    r: usize,
    q: usize,
    m: usize,
    beam: usize,
    lag: usize,
    shrinkage: usize,
    context_lags: usize,
    rho: f64,
    sketch_rho: f64,
    tau: f64,
    kappa_min: f64,
    kappa_max: f64,
    gate_power: f64,
    decay: f64,
}

#[derive(Clone, Copy)]
struct Visit {
    node: usize,
    mass: f64,
    parent: usize,
}

struct Fit {
    pred: Vec<f64>,
    mean: Vec<f64>,
    variance: f64,
    n: f64,
}

pub struct State {
    c: Config,
    weights: Vec<f64>,
    heap: Vec<f64>,
    sketch: Vec<f64>,
    previous: Vec<f64>,
    value_origin: Vec<f64>,
    key_history: Vec<f64>,
    count: u64,
    times: Vec<u64>,
    scratch: Scratch,
}

impl Config {
    fn nodes(self) -> usize {
        1 << (self.depth + 1)
    }
    fn leaves(self) -> usize {
        1 << self.depth
    }
    fn stat_width(self) -> usize {
        2 + self.r + self.q + self.r * self.r + self.r * self.q
    }
    fn sketch_width(self) -> usize {
        self.m * (1 + self.r + self.m + self.q)
    }
    fn weights_width(self) -> usize {
        self.d * (self.trees * self.r + 2 * self.q + 1)
            + 1
            + self.trees * self.leaves() * (self.r + 1)
            + self.trees * self.m * (self.r + 1)
            + self.trees * self.context_lags
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    // Independent accumulators let LLVM vectorize the wide projection dot products.
    let (mut left, mut right) = (a.chunks_exact(8), b.chunks_exact(8));
    let mut sums = [0.; 8];
    for (u, v) in left.by_ref().zip(right.by_ref()) {
        for i in 0..8 {
            sums[i] += u[i] * v[i];
        }
    }
    sums.iter().sum::<f64>()
        + left
            .remainder()
            .iter()
            .zip(right.remainder())
            .map(|(x, y)| x * y)
            .sum::<f64>()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn route(
    c: Config,
    key: &[f64],
    dirs: &[f64],
    theta: &[f64],
    path: &mut Vec<Visit>,
    candidates: &mut Vec<Visit>,
) {
    path.clear();
    path.push(Visit {
        node: 1,
        mass: 1.,
        parent: 0,
    });
    let (mut start, mut end) = (0, 1);
    for _ in 0..c.depth {
        candidates.clear();
        for (offset, visit) in path[start..end].iter().enumerate() {
            let parent = start + offset;
            let direction = &dirs[visit.node * c.r..(visit.node + 1) * c.r];
            let right = sigmoid((dot(key, direction) - theta[visit.node]) / c.tau);
            candidates.push(Visit {
                node: 2 * visit.node,
                mass: visit.mass * (1. - right),
                parent,
            });
            candidates.push(Visit {
                node: 2 * visit.node + 1,
                mass: visit.mass * right,
                parent,
            });
        }
        candidates.sort_unstable_by(|a, b| {
            b.mass
                .total_cmp(&a.mass)
                .then(a.parent.cmp(&b.parent))
                .then(a.node.cmp(&b.node))
        });
        path.extend_from_slice(&candidates[..c.beam.min(candidates.len())]);
        start = end;
        end = path.len();
    }
}

// Factor in place. Failed pivots return before sqrt or division.
fn solve_spd(a: &mut [f64], rhs: &mut [f64], p: usize, q: usize) -> bool {
    for i in 0..p {
        for j in 0..=i {
            let mut value = a[i * p + j];
            for k in 0..j {
                value -= a[i * p + k] * a[j * p + k];
            }
            if i == j && (!value.is_finite() || value <= 0.) {
                return false;
            }
            a[i * p + j] = if i == j {
                value.sqrt()
            } else {
                value / a[j * p + j]
            };
        }
        for j in 0..i {
            let value = a[i * p + j];
            for col in 0..q {
                rhs[i * q + col] -= value * rhs[j * q + col];
            }
        }
        for col in 0..q {
            rhs[i * q + col] /= a[i * p + i];
        }
    }
    for i in (0..p).rev() {
        for j in i + 1..p {
            let value = a[j * p + i];
            for col in 0..q {
                rhs[i * q + col] -= value * rhs[j * q + col];
            }
        }
        for col in 0..q {
            rhs[i * q + col] /= a[i * p + i];
        }
    }
    rhs.iter().all(|value| value.is_finite())
}

struct FitWork {
    c: Config,
    gram: Vec<f64>,
    cross: Vec<f64>,
    coef: Vec<f64>,
    mean_key: Vec<f64>,
}

impl Fit {
    fn new(q: usize) -> Self {
        Self {
            pred: vec![0.; q],
            mean: vec![0.; q],
            variance: 0.,
            n: 0.,
        }
    }
}

impl FitWork {
    fn new(c: Config) -> Self {
        let p = c.r + c.m;
        Self {
            c,
            gram: vec![0.; p * p],
            cross: vec![0.; p * c.q],
            coef: vec![0.; p * c.q],
            mean_key: vec![0.; p],
        }
    }

    fn predict(
        &mut self,
        s: &[f64],
        scale: f64,
        key: &[f64],
        extra: &[f64],
        phi: &[f64],
        out: &mut Fit,
    ) {
        let c = self.c;
        let (r, q, m) = (c.r, c.q, phi.len());
        let n = s[0] * scale;
        out.n = n;
        out.variance = 0.;
        if n <= 1e-8 {
            out.pred.fill(0.);
            out.mean.fill(0.);
            return;
        }
        let p = r + m;
        let b = &s[1 + r..1 + r + q];
        let g = &s[1 + r + q..1 + r + q + r * r];
        let u = &s[1 + r + q + r * r..s.len() - 1];
        let means_k = &mut self.mean_key[..p];
        for (mean, sum) in means_k
            .iter_mut()
            .zip(s[1..1 + r].iter().chain(&extra[..m]))
        {
            *mean = sum * scale / n;
        }
        for (mean, sum) in out.mean.iter_mut().zip(b) {
            *mean = sum * scale / n;
        }
        out.pred.copy_from_slice(&out.mean);
        let mut rss = (s[s.len() - 1] * scale - dot(b, &out.mean) * scale).max(0.);
        out.variance = rss / (n * q as f64);
        let regularizer = |i| if i < r { c.rho } else { c.sketch_rho };
        let gram = &mut self.gram[..p * p];
        let cross = &mut self.cross[..p * q];
        let coef = &mut self.coef[..p * q];
        for i in 0..p {
            for j in 0..p {
                let moment = match (i < r, j < r) {
                    (true, true) => g[i * r + j],
                    (true, false) => extra[m + i * m + j - r],
                    (false, true) => extra[m + j * m + i - r],
                    (false, false) => extra[m + r * m + (i - r) * m + j - r],
                };
                gram[i * p + j] = moment * scale - n * means_k[i] * means_k[j];
            }
            gram[i * p + i] += regularizer(i);
            for j in 0..q {
                let moment = if i < r {
                    u[i * q + j]
                } else {
                    extra[m + r * m + m * m + (i - r) * q + j]
                };
                cross[i * q + j] = moment * scale - n * means_k[i] * out.mean[j];
            }
        }
        coef.copy_from_slice(cross);
        if !solve_spd(gram, coef, p, q) {
            return; // Intercept-only prediction, with variance from the centered values.
        }
        let mut penalty = 0.;
        for (i, query) in key.iter().chain(phi).enumerate() {
            for j in 0..q {
                let slope = coef[i * q + j];
                out.pred[j] += (query - means_k[i]) * slope;
                penalty += regularizer(i) * slope * slope;
            }
        }
        rss = (rss - dot(cross, coef) - penalty).max(0.);
        if out.pred.iter().all(|value| value.is_finite()) && rss.is_finite() {
            out.variance = rss / (n * q as f64);
        } else {
            out.pred.copy_from_slice(&out.mean);
        }
    }
}

fn affine_row(key: &[f64], value: &[f64], row: &mut Vec<f64>) {
    row.clear();
    row.push(1.);
    row.extend_from_slice(key);
    row.extend_from_slice(value);
    for a in key {
        row.extend(key.iter().map(|b| a * b));
    }
    for a in key {
        row.extend(value.iter().map(|b| a * b));
    }
    row.push(dot(value, value));
}

fn features(key: &[f64], omega: &[f64], beta: &[f64], phi: &mut [f64]) {
    if phi.is_empty() {
        return;
    }
    let m = phi.len();
    let norm = (2. / m as f64).sqrt();
    for (j, feature) in phi.iter_mut().enumerate() {
        let angle = beta[j]
            + key
                .iter()
                .enumerate()
                .map(|(i, x)| x * omega[i * m + j])
                .sum::<f64>();
        *feature = norm * angle.cos();
    }
}

fn extra_row(key: &[f64], phi: &[f64], value: &[f64], row: &mut Vec<f64>) {
    row.clear();
    row.extend_from_slice(phi);
    for a in key {
        row.extend(phi.iter().map(|b| a * b));
    }
    for a in phi {
        row.extend(phi.iter().map(|b| a * b));
    }
    for a in phi {
        row.extend(value.iter().map(|b| a * b));
    }
}

struct Scratch {
    raw: Vec<f64>,
    keys: Vec<f64>,
    value: Vec<f64>,
    avg: Vec<f64>,
    read: Vec<Visit>,
    write: Vec<Visit>,
    candidates: Vec<Visit>,
    fits: Vec<Fit>,
    leaf_fit: Fit,
    fit: FitWork,
    read_phi: Vec<f64>,
    write_phi: Vec<f64>,
    affine: Vec<f64>,
    nonlinear: Vec<f64>,
}

impl Scratch {
    fn new(c: Config) -> Self {
        let path_width = (0..=c.depth).map(|d| (1 << d).min(c.beam)).sum();
        Self {
            raw: vec![0.; c.trees * c.r],
            keys: vec![0.; c.trees * c.r],
            value: vec![0.; c.q],
            avg: vec![0.; c.q],
            read: Vec::with_capacity(path_width),
            write: Vec::with_capacity(path_width),
            candidates: Vec::with_capacity(2 * c.beam.min(c.leaves())),
            fits: (0..path_width).map(|_| Fit::new(c.q)).collect(),
            leaf_fit: Fit::new(c.q),
            fit: FitWork::new(c),
            read_phi: vec![0.; c.m],
            write_phi: vec![0.; c.m],
            affine: Vec::with_capacity(c.stat_width()),
            nonlinear: Vec::with_capacity(c.sketch_width()),
        }
    }
}

fn discount(decay: f64, elapsed: u64) -> f64 {
    if elapsed == 0 {
        1.
    } else {
        decay.powf(elapsed as f64)
    }
}

impl State {
    fn step(&mut self, x: &[f64], output: &mut [f64], write_weight: f64) -> bool {
        let c = self.c;
        if !(0.0..=1.0).contains(&write_weight) || self.count == u64::MAX {
            return false;
        }
        let w = &mut self.scratch;
        let (wk, rest) = self.weights.split_at(c.trees * c.r * c.d);
        let (wv, rest) = rest.split_at(c.q * c.d);
        let (wo, rest) = rest.split_at(c.d * c.q);
        let (wg, rest) = rest.split_at(c.d + 1);
        let (dirs, rest) = rest.split_at(c.trees * c.leaves() * c.r);
        let (theta, rest) = rest.split_at(c.trees * c.leaves());
        let (omega, rest) = rest.split_at(c.trees * c.r * c.m);
        let (beta, mix) = rest.split_at(c.trees * c.m);
        let logit = dot(&wg[..c.d], x) + wg[c.d];
        if !logit.is_finite() {
            return false;
        }
        let er = c.trees * c.r;
        let cap = (f64::MAX / (c.q as f64 * 2f64.powi(66))).sqrt();
        for (j, row) in wk.chunks_exact(c.d).enumerate() {
            let raw = dot(row, x);
            if !raw.is_finite() || raw.abs() > cap {
                return false;
            }
            w.raw[j] = raw;
            let e = j / c.r;
            let mut key = raw;
            for lag in 1..=c.context_lags {
                key += self.key_history[(c.context_lags - lag) * er + j]
                    * mix[e * c.context_lags + lag - 1];
            }
            if !key.is_finite() {
                return false;
            }
            w.keys[j] = key.tanh();
        }
        for (dst, row) in w.value.iter_mut().zip(wv.chunks_exact(c.d)) {
            *dst = dot(row, x);
            if !dst.is_finite() || dst.abs() > cap {
                return false;
            }
        }
        if self.count == 0 {
            self.value_origin.copy_from_slice(&w.value);
        }
        for (value, origin) in w.value.iter_mut().zip(&self.value_origin) {
            *value -= origin;
        }
        let gate = if c.lag == 1 && self.count == 0 {
            0.
        } else {
            sigmoid(logit).powf(c.gate_power) * write_weight
        };
        let write_keys = if c.lag == 1 { &self.previous } else { &w.keys };
        w.avg.fill(0.);
        for e in 0..c.trees {
            let k = &w.keys[e * c.r..(e + 1) * c.r];
            let kw = &write_keys[e * c.r..(e + 1) * c.r];
            let edges = &dirs[e * c.leaves() * c.r..(e + 1) * c.leaves() * c.r];
            let thresholds = &theta[e * c.leaves()..(e + 1) * c.leaves()];
            let om = &omega[e * c.r * c.m..(e + 1) * c.r * c.m];
            let phase = &beta[e * c.m..(e + 1) * c.m];
            features(k, om, phase, &mut w.read_phi);
            route(c, k, edges, thresholds, &mut w.read, &mut w.candidates);
            for (vi, visit) in w.read.iter().enumerate() {
                let node = e * c.nodes() + visit.node;
                let i = node * c.stat_width();
                let stats = &self.heap[i..i + c.stat_width()];
                let factor = if c.decay == 1. {
                    1.
                } else {
                    discount(c.decay, self.count - self.times[node])
                };
                let (prior, current) = w.fits.split_at_mut(vi);
                let fit = &mut current[0];
                w.fit.predict(stats, factor, k, &[], &[], fit);
                if visit.node >= c.leaves() && c.m > 0 {
                    let i = (e * c.leaves() + visit.node - c.leaves()) * c.sketch_width();
                    let extra = &self.sketch[i..i + c.sketch_width()];
                    w.fit
                        .predict(stats, factor, k, extra, &w.read_phi, &mut w.leaf_fit);
                    fit.pred.copy_from_slice(&w.leaf_fit.pred);
                }
                if vi == 0 {
                    for (a, p) in w.avg.iter_mut().zip(&fit.pred) {
                        *a += *p;
                    }
                    if fit.n > 1e-8 {
                        for (a, origin) in w.avg.iter_mut().zip(&self.value_origin) {
                            *a += origin;
                        }
                    }
                } else {
                    let parent = &prior[visit.parent];
                    let between = fit
                        .mean
                        .iter()
                        .zip(&parent.mean)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        / c.q as f64
                        + 0.001 * parent.variance
                        + 1e-5;
                    let lam = if c.shrinkage == 0 {
                        if fit.n > 1e-8 {
                            1.
                        } else {
                            0.
                        }
                    } else {
                        let kappa = (fit.variance / between).clamp(c.kappa_min, c.kappa_max);
                        fit.n / (fit.n + kappa)
                    };
                    for ((a, p), prev) in w.avg.iter_mut().zip(&mut fit.pred).zip(&parent.pred) {
                        let correction = lam * (*p - prev);
                        *p = prev + correction;
                        *a += visit.mass * correction;
                    }
                }
            }
            if gate > 0. {
                affine_row(kw, &w.value, &mut w.affine);
                features(kw, om, phase, &mut w.write_phi);
                extra_row(kw, &w.write_phi, &w.value, &mut w.nonlinear);
                if c.lag == 1 {
                    route(c, kw, edges, thresholds, &mut w.write, &mut w.candidates);
                }
                let route = if c.lag == 1 { &w.write } else { &w.read };
                for visit in route {
                    let influence = gate * visit.mass;
                    let node = e * c.nodes() + visit.node;
                    let mut factor = 1.;
                    if c.decay < 1. {
                        factor = discount(c.decay, self.count + 1 - self.times[node]);
                        self.times[node] = self.count + 1;
                    }
                    let i = node * c.stat_width();
                    update(
                        &mut self.heap[i..i + c.stat_width()],
                        &w.affine,
                        factor,
                        influence,
                    );
                    if visit.node >= c.leaves() && c.m > 0 {
                        let i = (e * c.leaves() + visit.node - c.leaves()) * c.sketch_width();
                        update(
                            &mut self.sketch[i..i + c.sketch_width()],
                            &w.nonlinear,
                            factor,
                            influence,
                        );
                    }
                }
            }
        }
        for (dst, row) in output.iter_mut().zip(wo.chunks_exact(c.q)) {
            *dst = dot(row, &w.avg) / c.trees as f64;
        }
        self.previous.copy_from_slice(&w.keys);
        if c.context_lags != 0 {
            self.key_history.copy_within(er.., 0);
            self.key_history[(c.context_lags - 1) * er..].copy_from_slice(&w.raw);
        }
        self.count += 1;
        true
    }
}

fn update(target: &mut [f64], row: &[f64], decay: f64, weight: f64) {
    for (slot, value) in target.iter_mut().zip(row) {
        *slot = *slot * decay + weight * value;
    }
}

/// # Safety
/// The returned pointer must be released once by shrf_free.
#[no_mangle]
pub extern "C" fn shrf_new(c: Config) -> *mut State {
    if c.d == 0
        || c.trees == 0
        || c.r == 0
        || c.q == 0
        || c.beam == 0
        || c.depth > 20
        || c.lag > 1
        || c.context_lags > 16
        || !(1.0..=16.0).contains(&c.gate_power)
        || !(c.decay > 0. && c.decay <= 1.)
        || ![c.rho, c.sketch_rho, c.tau, c.kappa_min, c.kappa_max]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
        || c.kappa_min > c.kappa_max
    {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(State {
        c,
        weights: vec![0.0; c.weights_width()],
        heap: vec![0.0; c.trees * c.nodes() * c.stat_width()],
        sketch: vec![0.0; c.trees * c.leaves() * c.sketch_width()],
        previous: vec![0.0; c.trees * c.r],
        value_origin: vec![0.0; c.q],
        key_history: vec![0.0; c.context_lags * c.trees * c.r],
        count: 0,
        times: if c.decay < 1. {
            vec![0; c.trees * c.nodes()]
        } else {
            Vec::new()
        },
        scratch: Scratch::new(c),
    }))
}

/// # Safety
/// state is a live handle; weights points to len contiguous f64 values.
#[no_mangle]
pub unsafe extern "C" fn shrf_set_weights(
    state: *mut State,
    weights: *const f64,
    len: usize,
) -> bool {
    if state.is_null() || !state.is_aligned() || weights.is_null() || !weights.is_aligned() {
        return false;
    }
    let s = &mut *state;
    if len != s.weights.len() || s.count != 0 {
        return false;
    }
    let weights = slice::from_raw_parts(weights, len);
    if !weights.iter().all(|w| w.abs() <= 1e100) {
        return false;
    }
    s.weights.copy_from_slice(weights);
    true
}

/// # Safety
/// state is a live, exclusively borrowed handle. x and out are disjoint d-element buffers.
#[no_mangle]
pub unsafe extern "C" fn shrf_step(
    state: *mut State,
    x: *const f64,
    out: *mut f64,
    weight: f64,
) -> bool {
    if state.is_null()
        || !state.is_aligned()
        || x.is_null()
        || out.is_null()
        || !x.is_aligned()
        || !out.is_aligned()
    {
        return false;
    }
    let s = &mut *state;
    let d = s.c.d;
    s.step(
        slice::from_raw_parts(x, d),
        slice::from_raw_parts_mut(out, d),
        weight,
    )
}

/// # Safety
/// state is exclusive. x/out are disjoint len*d-element buffers. weight is null
/// for unit weights, or points to len contiguous f64 values.
#[no_mangle]
pub unsafe extern "C" fn shrf_decode(
    state: *mut State,
    x: *const f64,
    out: *mut f64,
    len: usize,
    weight: *const f64,
) -> usize {
    if len == 0
        || state.is_null()
        || !state.is_aligned()
        || x.is_null()
        || out.is_null()
        || !x.is_aligned()
        || !out.is_aligned()
        || (!weight.is_null() && !weight.is_aligned())
    {
        return 0;
    }
    let s = &mut *state;
    let d = s.c.d;
    let Some(items) = len
        .checked_mul(d)
        .filter(|size| *size <= isize::MAX as usize / std::mem::size_of::<f64>())
    else {
        return 0;
    };
    let input = slice::from_raw_parts(x, items);
    let output = slice::from_raw_parts_mut(out, items);
    let weights = if weight.is_null() {
        None
    } else {
        Some(slice::from_raw_parts(weight, len))
    };
    if !input.iter().all(|x| x.is_finite())
        || weights.is_some_and(|w| !w.iter().all(|x| (0. ..=1.).contains(x)))
    {
        return 0;
    }
    for (i, (row, dst)) in input
        .chunks_exact(d)
        .zip(output.chunks_exact_mut(d))
        .enumerate()
    {
        if !s.step(row, dst, weights.map_or(1.0, |w| w[i])) {
            return i;
        }
    }
    len
}

/// # Safety
/// state is a live, exclusively borrowed handle.
#[no_mangle]
pub unsafe extern "C" fn shrf_reset(state: *mut State) {
    if state.is_null() || !state.is_aligned() {
        return;
    }
    let s = &mut *state;
    s.heap.fill(0.0);
    s.sketch.fill(0.0);
    s.previous.fill(0.0);
    s.value_origin.fill(0.0);
    s.key_history.fill(0.0);
    s.times.fill(0);
    s.count = 0;
}

/// # Safety
/// state is a live handle from shrf_new and must not be used again.
#[no_mangle]
pub unsafe extern "C" fn shrf_free(state: *mut State) {
    if !state.is_null() {
        drop(Box::from_raw(state));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! { static TRACK: Cell<Option<usize>> = const { Cell::new(None) }; }
    struct TestAllocator;
    #[global_allocator]
    static ALLOCATOR: TestAllocator = TestAllocator;
    unsafe impl GlobalAlloc for TestAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = TRACK.try_with(|count| {
                if let Some(n) = count.get() {
                    count.set(Some(n + 1));
                }
            });
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            let _ = TRACK.try_with(|count| {
                if let Some(n) = count.get() {
                    count.set(Some(n + 1));
                }
            });
            System.dealloc(ptr, layout)
        }
    }

    #[test]
    fn cholesky_solves_two_right_hand_sides() {
        let mut x = [6., 4., 5., 2.];
        assert!(solve_spd(&mut [4., 2., 2., 3.], &mut x, 2, 2));
        for (actual, expected) in x.iter().zip([1., 1., 1., 0.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_rejects_indefinite_nonfinite_and_singular_inputs() {
        for diagonal in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(!solve_spd(&mut [diagonal], &mut [1.], 1, 1));
        }
    }

    #[test]
    fn invalid_gram_uses_intercept_only() {
        let c = test_config();
        let mut fit = Fit::new(c.q);
        FitWork::new(c).predict(&[1., 1., 2., 0., 2., 4.], 1., &[0.], &[], &[], &mut fit);
        assert_eq!(fit.pred, [2.]);
        assert_eq!(fit.variance, 0.);
    }

    #[test]
    fn ffi_rejects_null_and_overflowing_inputs() {
        let state = shrf_new(test_config());
        let x = [0.];
        let mut out = [0.];
        unsafe {
            use std::ptr::{null, null_mut};
            assert!(!shrf_set_weights(null_mut(), x.as_ptr(), 1));
            assert!(!shrf_set_weights(state, null(), 1));
            assert!(!shrf_step(state, null(), out.as_mut_ptr(), 1.));
            assert!(!shrf_step(null_mut(), x.as_ptr(), out.as_mut_ptr(), 1.));
            assert_eq!(shrf_decode(state, x.as_ptr(), null_mut(), 1, null()), 0);
            assert_eq!(
                shrf_decode(state, x.as_ptr(), out.as_mut_ptr(), usize::MAX, null()),
                0
            );
            shrf_reset(null_mut());
            shrf_free(state);
        }
    }

    #[test]
    fn tied_children_keep_stable_beam_order() {
        let config = test_config();
        let mut path = Vec::new();
        route(
            config,
            &[0.],
            &[0.; 4],
            &[0.; 4],
            &mut path,
            &mut Vec::new(),
        );
        assert_eq!(path[3..].iter().map(|v| v.node).collect::<Vec<_>>(), [4, 5]);
        assert_eq!(path[3..].iter().map(|v| v.mass).sum::<f64>(), 0.5);
    }

    fn test_config() -> Config {
        Config {
            d: 1,
            trees: 1,
            depth: 2,
            r: 1,
            q: 1,
            m: 0,
            beam: 2,
            lag: 1,
            shrinkage: 1,
            context_lags: 0,
            rho: 0.1,
            sketch_rho: 0.1,
            tau: 0.2,
            kappa_min: 0.25,
            kappa_max: 100.0,
            gate_power: 1.0,
            decay: 1.0,
        }
    }

    #[test]
    fn decoder_step_never_allocates_after_state_creation() {
        let c = Config {
            d: 64,
            trees: 4,
            depth: 6,
            r: 8,
            q: 32,
            m: 8,
            context_lags: 2,
            beam: 4,
            decay: 0.99,
            ..test_config()
        };
        let state = shrf_new(c);
        let state = unsafe { &mut *state };
        state.weights.fill(0.01);
        let mut out = [0.; 64];
        TRACK.with(|count| count.set(Some(0)));
        for _ in 0..16 {
            assert!(state.step(&[0.5; 64], &mut out, 1.));
        }
        let count = TRACK.with(|count| count.replace(None));
        assert_eq!(count, Some(0));
        assert!(out.iter().all(|x| x.is_finite()));
        unsafe { shrf_free(state) };
    }
}
