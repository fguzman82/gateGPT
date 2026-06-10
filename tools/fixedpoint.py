"""
Fixed-point integer reference for NamesGPT — the authoritative spec the RTL core
must reproduce bit-for-bit. Everything is Q5.11 signed 16-bit (FRAC=11). All ops
use only integer arithmetic that maps directly to hardware: wide MAC + arithmetic
right shift, integer isqrt + reciprocal for RMSNorm, a table+interp exp, and a
32-bit LCG for sampling. (These are our own choices, designed for this project.)
"""
import math
import numpy as np
from model import ModelConfig

FRAC = 11
SCALE = 1 << FRAC                 # 2048
QMAX, QMIN = 32767, -32768


def sat16(v):
    return QMAX if v > QMAX else (QMIN if v < QMIN else int(v))


def tdiv(a, b):
    """Truncate toward zero (matches a sign-magnitude hardware divider)."""
    qq = abs(int(a)) // abs(int(b))
    return -qq if (a < 0) != (b < 0) else qq


def q(x):
    """float -> Q5.11 saturated int16 (round to nearest)."""
    return sat16(int(math.floor(x * SCALE + 0.5)))


# ---- exp table: EXP_TAB[k] = round(exp(-k) * 2048), k = 0..EXP_K -------------
EXP_K = 16
EXP_TAB = [int(math.floor(math.exp(-k) * SCALE + 0.5)) for k in range(EXP_K + 1)]


def exp_neg_q11(z):
    """exp(z) in Q11 for z <= 0 (z in Q11). Table lookup + linear interpolation."""
    if z >= 0:
        return SCALE
    u = -z                                  # >= 0, Q11
    ui = u >> FRAC                           # integer part
    if ui >= EXP_K:
        return 0
    uf = u & (SCALE - 1)                     # fractional part, Q11
    lo, hi = EXP_TAB[ui], EXP_TAB[ui + 1]    # hi <= lo (decreasing)
    e = lo + ((hi - lo) * uf >> FRAC)        # arithmetic shift (uf>=0)
    return e if e > 0 else 0


def matvec(W_q, x_q):
    """y[o] = sat16( (sum_i W[o,i]*x[i]) >> FRAC ). W_q:[out,in] int, x_q:[in] int."""
    acc = W_q.astype(np.int64) @ x_q.astype(np.int64)
    return np.array([sat16(int(a) >> FRAC) for a in acc], dtype=np.int64)


def rmsnorm(x_q, gain_q):
    """y = x / sqrt(mean(x^2)) * gain, all Q11, via integer isqrt + reciprocal."""
    n = len(x_q)
    ss = int((x_q.astype(np.int64) ** 2).sum())      # Q22
    mean_sq = ss // n                                 # Q22
    if mean_sq < 1:
        mean_sq = 1
    r = math.isqrt(mean_sq)                           # Q11 (sqrt of Q22)
    if r < 1:
        r = 1
    scale = (1 << (2 * FRAC)) // r                    # 2^22 / r  -> Q11 reciprocal-sqrt
    if scale > QMAX:
        scale = QMAX
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = sat16((int(x_q[i]) * scale) >> FRAC)
        y[i] = sat16((t * int(gain_q[i])) >> FRAC)
    return y


class QModel:
    """Quantized NamesGPT; integer forward identical to the planned RTL."""

    def __init__(self, sd, cfg: ModelConfig):
        self.cfg = cfg
        self.tok = np.array([[q(v) for v in row] for row in sd["tok_embed.weight"]], dtype=np.int64)
        self.pos = np.array([[q(v) for v in row] for row in sd["pos_embed.weight"]], dtype=np.int64)
        b = "blocks.0."
        self.g1 = np.array([q(v) for v in sd[b + "norm1.gain"]], dtype=np.int64)
        self.g2 = np.array([q(v) for v in sd[b + "norm2.gain"]], dtype=np.int64)
        self.gf = np.array([q(v) for v in sd["norm_f.gain"]], dtype=np.int64)
        qz = lambda name: np.array([[q(v) for v in row] for row in sd[name]], dtype=np.int64)
        self.wq = qz(b + "attn.wq.weight")
        self.wk = qz(b + "attn.wk.weight")
        self.wv = qz(b + "attn.wv.weight")
        self.wo = qz(b + "attn.wo.weight")
        self.fc1 = qz(b + "mlp.fc1.weight")
        self.fc2 = qz(b + "mlp.fc2.weight")
        self.lm = qz("lm_head.weight")
        self.attn_scale = q(1.0 / math.sqrt(cfg.head_dim))   # 1/sqrt(head_dim) in Q11

    def attn_debug(self, ctx):
        """Return (qlast[24], k[T,24], v[T,24], attn_out[24]) for a ctx, for RTL testing."""
        cfg = self.cfg
        T, H, D = cfg.block_size, cfg.n_head, cfg.head_dim
        x = np.empty((T, cfg.n_embed), dtype=np.int64)
        for t in range(T):
            x[t] = np.array([sat16(int(self.tok[ctx[t]][i]) + int(self.pos[t][i]))
                             for i in range(cfg.n_embed)], dtype=np.int64)
        xn = np.array([rmsnorm(x[t], self.g1) for t in range(T)], dtype=np.int64)
        k = np.array([matvec(self.wk, xn[t]) for t in range(T)], dtype=np.int64)
        v = np.array([matvec(self.wv, xn[t]) for t in range(T)], dtype=np.int64)
        qlast = matvec(self.wq, xn[T - 1])
        attn_out = np.zeros(cfg.n_embed, dtype=np.int64)
        for h in range(H):
            sl = slice(h * D, (h + 1) * D)
            scores = []
            for s in range(T):
                acc = int((qlast[sl].astype(np.int64) * k[s][sl].astype(np.int64)).sum())
                sc = sat16(acc >> FRAC)
                sc = sat16((sc * self.attn_scale) >> FRAC)
                scores.append(sc)
            mm = max(scores)
            e = [exp_neg_q11(sc - mm) for sc in scores]
            se = sum(e)
            if se < 1:
                se = 1
            for d in range(D):
                num = sum(e[s] * int(v[s][h * D + d]) for s in range(T))
                attn_out[h * D + d] = sat16(tdiv(num, se))
        return qlast, k, v, attn_out

    def logits_last(self, ctx):
        """ctx: list of block_size token ids (left-padded). Returns Q11 logits at last pos."""
        cfg = self.cfg
        T, H, D = cfg.block_size, cfg.n_head, cfg.head_dim
        # embeddings for all positions
        x = np.empty((T, cfg.n_embed), dtype=np.int64)
        for t in range(T):
            x[t] = np.array([sat16(int(self.tok[ctx[t]][i]) + int(self.pos[t][i]))
                             for i in range(cfg.n_embed)], dtype=np.int64)
        # --- attention sub-layer (we only need the last position's output) ---
        xn = np.array([rmsnorm(x[t], self.g1) for t in range(T)], dtype=np.int64)
        k = np.array([matvec(self.wk, xn[t]) for t in range(T)], dtype=np.int64)
        v = np.array([matvec(self.wv, xn[t]) for t in range(T)], dtype=np.int64)
        qlast = matvec(self.wq, xn[T - 1])
        attn_out = np.zeros(cfg.n_embed, dtype=np.int64)
        for h in range(H):
            sl = slice(h * D, (h + 1) * D)
            scores = []
            for s in range(T):
                acc = int((qlast[sl].astype(np.int64) * k[s][sl].astype(np.int64)).sum())
                sc = sat16(acc >> FRAC)
                sc = sat16((sc * self.attn_scale) >> FRAC)
                scores.append(sc)
            m = max(scores)
            e = [exp_neg_q11(sc - m) for sc in scores]
            sum_e = sum(e)
            if sum_e < 1:
                sum_e = 1
            for d in range(D):
                num = sum(e[s] * int(v[s][h * D + d]) for s in range(T))
                attn_out[h * D + d] = sat16(tdiv(num, sum_e))
        wo = matvec(self.wo, attn_out)
        x1 = np.array([sat16(int(x[T - 1][i]) + int(wo[i])) for i in range(cfg.n_embed)], dtype=np.int64)
        # --- MLP sub-layer ---
        xn2 = rmsnorm(x1, self.g2)
        h1 = matvec(self.fc1, xn2)
        h1 = np.array([hh if hh > 0 else 0 for hh in h1], dtype=np.int64)   # ReLU
        h2 = matvec(self.fc2, h1)
        x2 = np.array([sat16(int(x1[i]) + int(h2[i])) for i in range(cfg.n_embed)], dtype=np.int64)
        # --- final norm + LM head ---
        xf = rmsnorm(x2, self.gf)
        return matvec(self.lm, xf)


# ---- deterministic sampler (32-bit LCG, Numerical Recipes constants) ----------
def lcg_next(state):
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def generate(model: QModel, seed, inv_temp_q11, max_len=None, greedy=False):
    """Generate one name. Returns (token_ids, string)."""
    cfg = model.cfg
    max_len = max_len or cfg.block_size
    rng = seed & 0xFFFFFFFF
    ctx = [0] * cfg.block_size
    toks = []
    for _ in range(max_len):
        logits = model.logits_last(ctx)
        if greedy:
            nxt = int(np.argmax(logits))
        else:
            scaled = [sat16((int(l) * inv_temp_q11) >> FRAC) for l in logits]
            m = max(scaled)
            e = [exp_neg_q11(s - m) for s in scaled]
            total = sum(e)
            if total < 1:
                total = 1
            rng = lcg_next(rng)
            r = rng % total
            acc, nxt = 0, len(e) - 1
            for i, ei in enumerate(e):
                acc += ei
                if acc > r:
                    nxt = i
                    break
        if nxt == 0:
            break
        toks.append(nxt)
        ctx = ctx[1:] + [nxt]
    s = "".join(chr(ord("a") + t - 1) for t in toks)
    return toks, s
