//! Batch-one CPU decoder for the Python SHRF equations. No external dependencies.
use std::slice;

#[no_mangle]
pub extern "C" fn shrf_abi_version() -> u32 {
    3
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
    key_history: Vec<f64>,
    occupied: bool,
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

fn route(c: Config, key: &[f64], dirs: &[f64], theta: &[f64]) -> Vec<Vec<Visit>> {
    let mut levels = Vec::with_capacity(c.depth + 1);
    levels.push(vec![Visit {
        node: 1,
        mass: 1.0,
        parent: 0,
    }]);
    for depth in 0..c.depth {
        let parents = &levels[depth];
        let mut next = Vec::with_capacity(2 * parents.len());
        for (parent, v) in parents.iter().enumerate() {
            let dir = &dirs[v.node * c.r..(v.node + 1) * c.r];
            let p = sigmoid((dot(key, dir) - theta[v.node]) / c.tau);
            next.push(Visit {
                node: 2 * v.node,
                mass: v.mass * (1.0 - p),
                parent,
            });
            next.push(Visit {
                node: 2 * v.node + 1,
                mass: v.mass * p,
                parent,
            });
        }
        next.sort_by(|a, b| b.mass.total_cmp(&a.mass));
        next.truncate(c.beam);
        levels.push(next);
    }
    levels
}

// SPD solve with Cholesky. The Python implementation uses LU.
fn solve_spd(mut a: Vec<f64>, mut rhs: Vec<f64>, p: usize, q: usize) -> Vec<f64> {
    for i in 0..p {
        for j in 0..=i {
            let mut v = a[i * p + j];
            for k in 0..j {
                v -= a[i * p + k] * a[j * p + k];
            }
            a[i * p + j] = if j == i { v.sqrt() } else { v / a[j * p + j] };
        }
        for j in 0..i {
            let v = a[i * p + j];
            for col in 0..q {
                rhs[i * q + col] -= v * rhs[j * q + col];
            }
        }
        for col in 0..q {
            rhs[i * q + col] /= a[i * p + i];
        }
    }
    for i in (0..p).rev() {
        for j in i + 1..p {
            let v = a[j * p + i];
            for col in 0..q {
                rhs[i * q + col] -= v * rhs[j * q + col];
            }
        }
        for col in 0..q {
            rhs[i * q + col] /= a[i * p + i];
        }
    }
    rhs
}

fn predict(c: Config, s: &[f64], key: &[f64], extra: &[f64], phi: &[f64]) -> Fit {
    let (r, q, m) = (c.r, c.q, phi.len());
    let n = s[0];
    if n <= 1e-8 {
        return Fit {
            pred: vec![0.0; q],
            mean: vec![0.0; q],
            variance: 0.0,
            n,
        };
    }
    let p = r + m;
    let a = &s[1..1 + r];
    let b = &s[1 + r..1 + r + q];
    let g = &s[1 + r + q..1 + r + q + r * r];
    let u = &s[1 + r + q + r * r..s.len() - 1];
    let mean: Vec<_> = b.iter().map(|v| v / n).collect();
    let mut means_k: Vec<_> = a.iter().chain(extra[..m].iter()).map(|v| v / n).collect();
    let mut gram = vec![0.0; p * p];
    let mut cross = vec![0.0; p * q];
    let mut reg = vec![c.rho.max(1e-5 * n); p];
    reg[r..].fill(c.sketch_rho.max(1e-5 * n));
    let mut query = Vec::with_capacity(p);
    query.extend_from_slice(key);
    query.extend_from_slice(phi);
    for i in 0..p {
        for j in 0..p {
            let moment = match (i < r, j < r) {
                (true, true) => g[i * r + j],
                (true, false) => extra[m + i * m + (j - r)],
                (false, true) => extra[m + j * m + (i - r)],
                (false, false) => extra[m + r * m + (i - r) * m + (j - r)],
            };
            gram[i * p + j] = moment - n * means_k[i] * means_k[j];
        }
        gram[i * p + i] += reg[i];
        for j in 0..q {
            let moment = if i < r {
                u[i * q + j]
            } else {
                extra[m + r * m + m * m + (i - r) * q + j]
            };
            cross[i * q + j] = moment - n * means_k[i] * mean[j];
        }
    }
    let coef = solve_spd(gram, cross.clone(), p, q);
    let mut pred = mean.clone();
    let mut penalty = 0.0;
    for i in 0..p {
        means_k[i] = query[i] - means_k[i];
        for j in 0..q {
            let slope = coef[i * q + j];
            pred[j] += means_k[i] * slope;
            penalty += reg[i] * slope * slope;
        }
    }
    let rss = (s[s.len() - 1] - dot(b, &mean) - dot(&cross, &coef) - penalty).max(0.0);
    Fit {
        pred,
        mean,
        variance: rss / (n * q as f64),
        n,
    }
}

fn affine_row(key: &[f64], v: &[f64], capacity: usize) -> Vec<f64> {
    let mut row = Vec::with_capacity(capacity);
    row.push(1.0);
    row.extend_from_slice(key);
    row.extend_from_slice(v);
    for a in key {
        row.extend(key.iter().map(|b| a * b));
    }
    for a in key {
        row.extend(v.iter().map(|b| a * b));
    }
    row.push(dot(v, v));
    row
}

fn features(c: Config, key: &[f64], omega: &[f64], beta: &[f64]) -> Vec<f64> {
    (0..c.m)
        .map(|j| {
            let angle = beta[j]
                + key
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * omega[i * c.m + j])
                    .sum::<f64>();
            (2.0 / c.m as f64).sqrt() * angle.cos()
        })
        .collect()
}

fn extra_row(key: &[f64], phi: &[f64], v: &[f64], capacity: usize) -> Vec<f64> {
    let mut row = Vec::with_capacity(capacity);
    row.extend_from_slice(phi);
    for a in key {
        row.extend(phi.iter().map(|b| a * b));
    }
    for a in phi {
        row.extend(phi.iter().map(|b| a * b));
    }
    for a in phi {
        row.extend(v.iter().map(|b| a * b));
    }
    row
}

impl State {
    fn step(&mut self, x: &[f64], output: &mut [f64], write_weight: f64) {
        let c = self.c;
        let (wk, rest) = self.weights.split_at(c.trees * c.r * c.d);
        let (wv, rest) = rest.split_at(c.q * c.d);
        let (wo, rest) = rest.split_at(c.d * c.q);
        let (wg, rest) = rest.split_at(c.d + 1);
        let (dirs, rest) = rest.split_at(c.trees * c.leaves() * c.r);
        let (theta, rest) = rest.split_at(c.trees * c.leaves());
        let (omega, rest) = rest.split_at(c.trees * c.r * c.m);
        let (beta, mix) = rest.split_at(c.trees * c.m);
        let raw: Vec<_> = wk.chunks_exact(c.d).map(|row| dot(row, x)).collect();
        let er = c.trees * c.r;
        let keys: Vec<_> = raw
            .iter()
            .enumerate()
            .map(|(j, raw_key)| {
                let mut key = *raw_key;
                let e = j / c.r;
                for lag in 1..=c.context_lags {
                    key += self.key_history[(c.context_lags - lag) * er + j]
                        * mix[e * c.context_lags + lag - 1];
                }
                key.tanh()
            })
            .collect();
        let value: Vec<_> = wv.chunks_exact(c.d).map(|row| dot(row, x)).collect();
        let gate = if c.lag == 1 && !self.occupied {
            0.0
        } else {
            let probability = sigmoid(dot(&wg[..c.d], x) + wg[c.d]);
            (if c.gate_power == 1.0 {
                probability
            } else {
                probability.powf(c.gate_power)
            }) * write_weight
        };
        let write_keys = if c.lag == 1 { &self.previous } else { &keys };
        let mut avg = vec![0.0; c.q];
        for e in 0..c.trees {
            let k = &keys[e * c.r..(e + 1) * c.r];
            let kw = &write_keys[e * c.r..(e + 1) * c.r];
            let edges = &dirs[e * c.leaves() * c.r..(e + 1) * c.leaves() * c.r];
            let thresholds = &theta[e * c.leaves()..(e + 1) * c.leaves()];
            let om = &omega[e * c.r * c.m..(e + 1) * c.r * c.m];
            let phase = &beta[e * c.m..(e + 1) * c.m];
            let leaf_phi = features(c, k, om, phase);
            let read = route(c, k, edges, thresholds);
            let mut last: Vec<Fit> = Vec::new();
            for (depth, level) in read.iter().enumerate() {
                let mut next = Vec::with_capacity(level.len());
                for visit in level {
                    let i = (e * c.nodes() + visit.node) * c.stat_width();
                    let stats = &self.heap[i..i + c.stat_width()];
                    let mut fit = predict(c, stats, k, &[], &[]);
                    if depth == c.depth && c.m > 0 {
                        let i = (e * c.leaves() + visit.node - c.leaves()) * c.sketch_width();
                        let extra = &self.sketch[i..i + c.sketch_width()];
                        fit.pred = predict(c, stats, k, extra, &leaf_phi).pred;
                    }
                    if depth == 0 {
                        for (a, p) in avg.iter_mut().zip(&fit.pred) {
                            *a += *p;
                        }
                    } else {
                        let parent = &last[visit.parent];
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
                                1.0
                            } else {
                                0.0
                            }
                        } else {
                            let kappa = (fit.variance / between).clamp(c.kappa_min, c.kappa_max);
                            fit.n / (fit.n + kappa)
                        };
                        for ((a, p), prev) in avg.iter_mut().zip(&mut fit.pred).zip(&parent.pred) {
                            let correction = lam * (*p - prev);
                            *p = prev + correction;
                            *a += visit.mass * correction;
                        }
                    }
                    next.push(fit);
                }
                last = next;
            }
            if gate > 0.0 {
                let raw = affine_row(kw, &value, c.stat_width());
                let phi = features(c, kw, om, phase);
                let nonlinear = extra_row(kw, &phi, &value, c.sketch_width());
                for (depth, level) in route(c, kw, edges, thresholds).iter().enumerate() {
                    for visit in level {
                        let influence = gate * visit.mass;
                        let i = (e * c.nodes() + visit.node) * c.stat_width();
                        for (slot, row) in self.heap[i..i + c.stat_width()].iter_mut().zip(&raw) {
                            *slot += influence * row;
                        }
                        if depth == c.depth && c.m > 0 {
                            let i = (e * c.leaves() + visit.node - c.leaves()) * c.sketch_width();
                            for (slot, row) in self.sketch[i..i + c.sketch_width()]
                                .iter_mut()
                                .zip(&nonlinear)
                            {
                                *slot += influence * row;
                            }
                        }
                    }
                }
            }
        }
        for (dst, row) in output.iter_mut().zip(wo.chunks_exact(c.q)) {
            *dst = dot(row, &avg) / c.trees as f64;
        }
        self.previous.copy_from_slice(&keys);
        if c.context_lags > 0 {
            self.key_history.copy_within(er.., 0);
            self.key_history[(c.context_lags - 1) * er..].copy_from_slice(&raw);
        }
        self.occupied = true;
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
        key_history: vec![0.0; c.context_lags * c.trees * c.r],
        occupied: false,
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
    let s = &mut *state;
    if len != s.weights.len() || s.occupied {
        return false;
    }
    s.weights
        .copy_from_slice(slice::from_raw_parts(weights, len));
    true
}

/// # Safety
/// state is a live, exclusively borrowed handle. x and out are disjoint d-element buffers.
#[no_mangle]
pub unsafe extern "C" fn shrf_step(state: *mut State, x: *const f64, out: *mut f64, weight: f64) {
    let s = &mut *state;
    let d = s.c.d;
    s.step(
        slice::from_raw_parts(x, d),
        slice::from_raw_parts_mut(out, d),
        weight,
    );
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
) {
    if len == 0 {
        return;
    }
    let s = &mut *state;
    let d = s.c.d;
    let input = slice::from_raw_parts(x, len * d);
    let output = slice::from_raw_parts_mut(out, len * d);
    let weights = if weight.is_null() {
        None
    } else {
        Some(slice::from_raw_parts(weight, len))
    };
    for (i, (row, dst)) in input
        .chunks_exact(d)
        .zip(output.chunks_exact_mut(d))
        .enumerate()
    {
        s.step(row, dst, weights.map_or(1.0, |w| w[i]));
    }
}

/// # Safety
/// state is a live, exclusively borrowed handle.
#[no_mangle]
pub unsafe extern "C" fn shrf_reset(state: *mut State) {
    let s = &mut *state;
    s.heap.fill(0.0);
    s.sketch.fill(0.0);
    s.previous.fill(0.0);
    s.key_history.fill(0.0);
    s.occupied = false;
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

    #[test]
    fn cholesky_solves_two_right_hand_sides() {
        let x = solve_spd(vec![4., 2., 2., 3.], vec![6., 4., 5., 2.], 2, 2);
        for (actual, expected) in x.iter().zip([1., 1., 1., 0.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn tied_children_keep_stable_beam_order() {
        let config = Config {
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
        };
        let visits = route(config, &[0.], &[0.; 4], &[0.; 4]);
        assert_eq!(visits[2].iter().map(|v| v.node).collect::<Vec<_>>(), [4, 5]);
        assert_eq!(visits[2].iter().map(|v| v.mass).sum::<f64>(), 0.5);
    }
}
