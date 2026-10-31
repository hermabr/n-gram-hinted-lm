"""The count mixture: the model's probability of each target mixed with exact n-gram counts of its contexts
(build.py), its weights fit on the clock on held-out training positions (the fit positions), never on validation.

Per position and level j (ascending) the counts tot_j (its context's occurrences), dist_j (their distinct next tokens)
and cnt_j (those followed by the target); a level whose context never occurred (tot_j = 0) is skipped. Witten-Bell
backoff onto the model: q_0 = p_model, q_j = (cnt_j + G_j q_{j-1}) / (tot_j + G_j) with
G_j = dist_j exp(gamma[j, bin_j] + gamma_sparse[j, dist_j == 1]), bin_j = floor(log2 tot_j) (at most NB - 1); p = q_L.
A proper distribution over the vocabulary (the weights see only the contexts' counts, never the target), evaluated at
the target.

The fit minimizes the mean -log p over the fit positions: Adam, full batch, with the gradient written out (the
per-parameter sums by atomic index_add_ into SPREAD copies: autograd's sort-based scatter of a gather's gradient is
~100x slower here), each step one compiled graph."""
import torch

NB = 25
SPREAD = 64  # copies of the gradient sums, by position % SPREAD (atomics on fewer hot addresses)
ITERS = 100
LR = 0.1


def features(counts: torch.Tensor):
    """counts int32 [3, L, n] (tot as uint32 bits, dist, cnt) -> per position and level: tot, dist, cnt (float32), the
    parameter index of its bin, of its sparse flag."""
    tot = (counts[0].t().to(torch.int64) & 0xFFFFFFFF).float()
    dist = counts[1].t().float()
    cnt = torch.minimum(counts[2].t().float(), tot)
    L = tot.shape[1]
    b = torch.clamp(torch.floor(torch.log2(tot.clamp_min(1))), 0, NB - 1).long() + NB * torch.arange(L, device=tot.device)
    s = (dist == 1).long() + 2 * torch.arange(L, device=tot.device) + NB * L
    return tot.contiguous(), dist.contiguous(), cnt.contiguous(), b.contiguous(), s.contiguous()


def _forward(theta: torch.Tensor, p, tot, dist, cnt, b, s):
    """q_L, every q_j (q_{-1} = p first), and the per-level G."""
    has = tot > 0
    w = dist * torch.exp((theta[b] + theta[s]).clamp(-20, 20))
    q, qs = p, [p]
    for j in range(tot.shape[1]):
        q = torch.where(has[:, j], (cnt[:, j] + w[:, j] * q) / (tot[:, j] + w[:, j]).clamp_min(1e-30), q)
        qs.append(q)
    return q, qs, w, has


def _step(theta, m, v, t, p, tot, dist, cnt, b, s, spread):
    """One Adam step on the mean -log q_L; returns the loss before the step."""
    n, L = tot.shape
    P = theta.numel()
    q, qs, w, has = _forward(theta, p, tot, dist, cnt, b, s)
    loss = -torch.log(q.clamp_min(1e-30)).mean()
    a = -1.0 / (n * q.clamp_min(1e-30))  # d loss / d q_j, from j = L - 1 down
    gw = []
    for j in reversed(range(L)):
        D = (tot[:, j] + w[:, j]).clamp_min(1e-30)
        gw.append(torch.where(has[:, j], a * (qs[j] - qs[j + 1]) / D * w[:, j], 0.0))  # d/dlog G
        a = torch.where(has[:, j], a * w[:, j] / D, a)
    gw = torch.stack(gw[::-1], 1).flatten()  # [n, L]
    grad = torch.zeros(SPREAD * P, device=theta.device)
    grad.index_add_(0, (b + spread).flatten(), gw)
    grad.index_add_(0, (s + spread).flatten(), gw)
    grad = grad.view(SPREAD, P).sum(0)
    m.mul_(0.9).add_(grad, alpha=0.1)
    v.mul_(0.999).addcmul_(grad, grad, value=0.001)
    theta.sub_(LR * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)).sqrt() + 1e-8))
    return loss


class CountMix:
    def __init__(self, levels: int, device):
        self.L = levels
        self.theta = torch.zeros(NB * levels + 2 * levels, device=device)  # gamma [NB * L], then gamma_sparse [2 * L]
        self.step = torch.compile(_step, dynamic=False, fullgraph=True)

    @torch.no_grad()
    def nll(self, p: torch.Tensor, feats) -> torch.Tensor:
        """Per position -log p_mix (float32 [n]) with the fit weights."""
        return -torch.log(_forward(self.theta, p.float(), *feats)[0].clamp_min(1e-30))

    @torch.no_grad()
    def fit(self, p: torch.Tensor, feats, iters: int = ITERS) -> float:
        """Fit the weights on the fit positions' model probabilities `p` [n] and features; returns the last loss."""
        theta = torch.zeros_like(self.theta)
        m, v = torch.zeros_like(theta), torch.zeros_like(theta)
        p = p.float()
        spread = (torch.arange(p.numel(), device=p.device) % SPREAD * theta.numel())[:, None]
        for t in range(1, iters + 1):
            loss = self.step(theta, m, v, torch.tensor(float(t), device=p.device), p, *feats, spread)
        self.theta = theta
        return float(loss)

    def warm(self, n: int) -> None:
        """Compile the fit's graph for n positions (before the clock)."""
        z = torch.zeros((3, self.L, n), dtype=torch.int32, device=self.theta.device)
        z[0], z[1], z[2] = 3, 2, 1
        self.fit(torch.full((n,), 0.1, device=z.device), features(z), iters=2)
        self.theta.zero_()
