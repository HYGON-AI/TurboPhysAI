# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""moge 的 focal / shift 求解搬到 GPU。

官方实现（moge/utils/geometry_torch.py::recover_focal_shift → moge/utils/geometry_numpy.py::solve_optimal_focal_shift）逐样本调用::

    scipy.optimize.least_squares(fn, x0=0, ftol=1e-3, method='lm')

这里把 lmdif 及其子程序 fdjac2 / qrfac / qrsolv / lmpar / enorm / dpmpar 逐行翻译成 batched torch，
算法、迭代步长、阻尼参数更新和三个收敛判据都与上游一致，只是把 m 个样本并行算。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moge.utils.geometry_torch import normalized_view_plane_uv
from moge.utils.geometry_torch import recover_focal_shift as _original

# dpmpar(1) / dpmpar(2) from MINPACK dpmpar.f
_EPSMCH = 2.22044604926e-16
_DWARF = 2.22507385852e-308
# constants in enorm.f
_RDWARF = 3.834e-20
_RGIANT = 1.304e19

# scipy/_lsq/least_squares.py::call_minpack with least_squares(method='lm')
_FTOL = 1e-3
_XTOL = 1e-8
_GTOL = 1e-8
_MAXFEV = 200
_EPSFCN = 2.220446049250313e-16
_FACTOR = 100.0


def _enorm(x):
    """enorm.f, batched over dim 1 (same three-sum splitting)."""
    _, m = x.shape
    xabs = x.abs()
    agiant = _RGIANT / m
    large = xabs >= agiant
    small = xabs <= _RDWARF
    mid = ~(large | small)

    s2 = torch.where(mid, xabs * xabs, torch.zeros_like(xabs)).sum(1)

    x1max = torch.where(large, xabs, torch.zeros_like(xabs)).max(1).values
    x1s = torch.where(x1max > 0, x1max, torch.ones_like(x1max))
    s1 = torch.where(
        large, (xabs / x1s.unsqueeze(1)) ** 2, torch.zeros_like(xabs)
    ).sum(1)
    s1 = torch.where(x1max > 0, s1, torch.zeros_like(s1))

    x3max = torch.where(small, xabs, torch.zeros_like(xabs)).max(1).values
    x3s = torch.where(x3max > 0, x3max, torch.ones_like(x3max))
    s3 = torch.where(
        small, (xabs / x3s.unsqueeze(1)) ** 2, torch.zeros_like(xabs)
    ).sum(1)
    s3 = torch.where(x3max > 0, s3, torch.zeros_like(s3))

    out = torch.zeros_like(s2)
    out = torch.where(s1 != 0, x1max * torch.sqrt(s1 + (s2 / x1s) / x1s), out)
    s2p = torch.where(s2 > 0, s2, torch.ones_like(s2))
    big = torch.sqrt(s2p * (1 + (x3s / s2p) * (x3s * s3)))
    smallb = torch.sqrt(x3s * ((s2 / x3s) + (x3s * s3)))
    out = torch.where(
        (s1 == 0) & (s2 != 0), torch.where(s2 >= x3max, big, smallb), out
    )
    out = torch.where((s1 == 0) & (s2 == 0), x3max * torch.sqrt(s3), out)
    return out


def _lmdif(resid, x0):
    """MINPACK lmdif specialised to n == 1, mode == 2 (diag == 1)."""
    b = x0.shape[0]
    zero = torch.zeros(b, dtype=x0.dtype, device=x0.device)
    one = torch.ones(b, dtype=x0.dtype, device=x0.device)

    x = x0.clone()
    fvec = resid(x)
    fnorm = _enorm(fvec)

    par = zero.clone()
    delta = zero.clone()
    xnorm = zero.clone()
    gnorm = zero.clone()
    r11 = zero.clone()
    qtf1 = zero.clone()

    nfev = torch.ones(b, dtype=torch.long, device=x0.device)
    it = torch.ones(b, dtype=torch.long, device=x0.device)
    info = torch.zeros(b, dtype=torch.long, device=x0.device)
    active = torch.ones(b, dtype=torch.bool, device=x0.device)
    has_jac = torch.ones(b, dtype=torch.bool, device=x0.device)

    eps = max(_EPSFCN, _EPSMCH) ** 0.5
    tiny = torch.full_like(zero, _DWARF)
    tenth = torch.full_like(zero, 0.1)
    half = torch.full_like(zero, 0.5)
    quarter = torch.full_like(zero, 0.25)

    for _ in range(_MAXFEV + 8):
        if not bool(active.any()):
            break

        # ---- outer loop: fdjac2 + qrfac(n=1) + qtf + gnorm -------------
        need = active & has_jac
        if bool(need.any()):
            h = eps * x.abs()
            h = torch.where(h == 0, torch.full_like(h, eps), h)
            jvec = (resid(x + h) - fvec) / h.unsqueeze(1)

            # qrfac: one Householder column
            acnorm_n = _enorm(jvec)
            aj = torch.where(jvec[:, 0] < 0, -acnorm_n, acnorm_n)
            nz = aj != 0
            ajs = torch.where(nz, aj, one)
            hh_n = torch.where(nz.unsqueeze(1), jvec / ajs.unsqueeze(1), jvec)
            hh_n = hh_n.clone()
            hh_n[:, 0] += torch.where(nz, one, zero)

            # form (Q^T)*fvec, and keep R11 = -ajnorm
            ssum = (hh_n * fvec).sum(1)
            temp = torch.where(hh_n[:, 0] != 0, -ssum / hh_n[:, 0], zero)
            wa4 = fvec + hh_n * temp.unsqueeze(1)
            qtf1_n = wa4[:, 0]
            r11_n = -aj

            gnorm_n = torch.where(
                (fnorm != 0) & (acnorm_n != 0),
                (r11_n * (qtf1_n / fnorm) / acnorm_n).abs(),
                zero,
            )

            r11 = torch.where(need, r11_n, r11)
            qtf1 = torch.where(need, qtf1_n, qtf1)
            gnorm = torch.where(need, gnorm_n, gnorm)
            nfev = nfev + need.long()

            # iter == 1: xnorm = |x| (initially 0), delta = factor * xnorm
            first = need & (it == 1)
            delta = torch.where(first, _FACTOR * xnorm, delta)
            delta = torch.where(
                first & (delta == 0), torch.full_like(delta, _FACTOR), delta
            )

            conv = need & (gnorm <= _GTOL)
            info = torch.where(conv, torch.full_like(info, 4), info)
            active = active & ~conv

        # ---- lmpar (10 inner iterations) ------------------------------
        rnz = r11 != 0
        r11s = torch.where(rnz, r11, one)
        x_gn = torch.where(rnz, qtf1 / r11s, zero)
        dxn = x_gn.abs()
        dxs = torch.where(dxn > 0, dxn, one)
        deltas = torch.where(delta > 0, delta, tiny)

        fp = dxn - deltas
        done = fp <= 0.1 * deltas

        wa1 = torch.where(rnz, x_gn / dxs, zero) / r11s
        tt = wa1.abs()
        parl = torch.where(rnz & (tt > 0), ((fp / deltas) / tt) / tt, zero)

        g_lm = (r11 * qtf1).abs()
        paru = g_lm / deltas
        paru = torch.where(paru == 0, _DWARF / torch.minimum(deltas, tenth), paru)

        par_new = torch.maximum(par, parl)
        par_new = torch.minimum(par_new, paru)
        par_new = torch.where(par_new == 0, g_lm / dxs, par_new)
        # lmdif: Gauss-Newton accepted before the inner loop -> go to 220,
        # where `if (iter == 0) par = 0`.
        par = torch.where(done, zero, par_new)

        # Gauss-Newton is often accepted outright (lmdif jumps to 220 before
        # entering the inner loop); skip the whole damping search then.
        inner = range(10) if not bool(done.all()) else range(0)
        for j in inner:
            prev = done.clone()
            upd = ~prev
            par = torch.where(
                upd & (par == 0), torch.maximum(tiny, 1e-3 * paru), par
            )
            d = par.sqrt()

            # qrsolv(n=1): one Givens rotation
            big = r11.abs() >= d.abs()
            tan = torch.where(big, d / r11s, zero)
            cos_a = half / torch.sqrt(quarter + quarter * tan * tan)
            sin_a = cos_a * tan
            cot = torch.where(~big, r11 / torch.where(d > 0, d, one), zero)
            sin_b = half / torch.sqrt(quarter + quarter * cot * cot)
            cos_b = sin_b * cot
            cos = torch.where(big, cos_a, cos_b)
            sin = torch.where(big, sin_a, sin_b)

            rn = cos * r11 + sin * d
            rns = torch.where(rn != 0, rn, one)
            x_new = torch.where(rn != 0, cos * qtf1 / rns, zero)

            dxn = x_new.abs()
            dxs = torch.where(dxn > 0, dxn, one)
            temp_old = fp
            fp = dxn - deltas

            stop = (
                (fp.abs() <= 0.1 * deltas)
                | ((parl == 0) & (fp <= temp_old) & (temp_old < 0))
                | (j == 9)
            )
            done = done | stop
            x_gn = torch.where(prev, x_gn, x_new)

            wa1 = x_new / dxs / rns
            tv = wa1.abs()
            parc = ((fp / deltas) / tv) / tv
            parl = torch.where(fp > 0, torch.maximum(parl, par), parl)
            paru = torch.where(fp < 0, torch.minimum(paru, par), paru)
            par = torch.where(~done, torch.maximum(parl, par + parc), par)

        # ---- trial step ------------------------------------------------
        p = -x_gn
        x_try = x + p
        pnorm = p.abs()
        first = active & (it == 1)
        delta = torch.where(first, torch.minimum(delta, pnorm), delta)

        wa4 = resid(x_try)
        nfev = nfev + active.long()
        fnorm1 = _enorm(wa4)

        actred = torch.where(
            0.1 * fnorm1 < fnorm, 1 - (fnorm1 / fnorm) ** 2, -torch.ones_like(fnorm)
        )
        temp1 = (r11 * p).abs() / fnorm
        temp2 = par.sqrt() * pnorm / fnorm
        prered = temp1 ** 2 + temp2 ** 2 / 0.5
        dirder = -(temp1 ** 2 + temp2 ** 2)
        ratio = torch.where(prered != 0, actred / prered, zero)

        # ---- update the step bound (lmdif 240-260) ---------------------
        ok25 = ratio > 0.25
        tA = torch.where(
            actred >= 0,
            torch.full_like(actred, 0.5),
            0.5 * dirder / (dirder + 0.5 * actred),
        )
        tA = torch.where(
            (0.1 * fnorm1 >= fnorm) | (tA < 0.1), torch.full_like(tA, 0.1), tA
        )
        keep = (par != 0) & (ratio < 0.75)
        delta = torch.where(
            ok25, torch.where(keep, delta, pnorm / 0.5), tA * torch.minimum(delta, pnorm / 0.1)
        )
        par = torch.where(
            ok25,
            torch.where(keep, par, 0.5 * par),
            par / tA,
        )

        # ---- successful iteration --------------------------------------
        success = active & (ratio >= 1e-4)
        x = torch.where(success, x_try, x)
        fvec = torch.where(success.unsqueeze(1), wa4, fvec)
        fnorm = torch.where(success, fnorm1, fnorm)
        xnorm = torch.where(success, x.abs(), xnorm)
        it = it + success.long()

        # ---- convergence tests -----------------------------------------
        c1 = (actred.abs() <= _FTOL) & (prered <= _FTOL) & (0.5 * ratio <= 1)
        info = torch.where(active & c1, torch.full_like(info, 1), info)
        info = torch.where(
            active & (delta <= _XTOL * xnorm), torch.full_like(info, 2), info
        )
        info = torch.where(active & c1 & (info == 2), torch.full_like(info, 3), info)
        active = active & (info == 0)

        info = torch.where(active & (nfev >= _MAXFEV), torch.full_like(info, 5), info)
        c6 = (
            (actred.abs() <= _EPSMCH)
            & (prered <= _EPSMCH)
            & (0.5 * ratio <= 1)
        )
        info = torch.where(active & c6, torch.full_like(info, 6), info)
        info = torch.where(
            active & (delta <= _EPSMCH * xnorm), torch.full_like(info, 7), info
        )
        info = torch.where(active & (gnorm <= _EPSMCH), torch.full_like(info, 8), info)
        active = active & (info == 0)

        has_jac = success & active

    return x


def _solve_batch(uv, xy, z, w):
    b, n = z.shape

    # scipy runs the least squares in float64 (`z + shift` promotes the
    # residual); keep the same precision here.
    uv64, xy64, z64 = uv.double(), xy.double(), z.double()
    w64 = None if w is None else w.double()

    def resid(s):
        p = xy64 / (z64 + s.unsqueeze(1)).unsqueeze(-1)
        if w64 is None:
            num = (p * uv64).sum((1, 2))
            den = (p * p).sum((1, 2))
            r = p * (num / den).unsqueeze(1).unsqueeze(1) - uv64
        else:
            num = (p * uv64 * w64.unsqueeze(-1)).sum((1, 2))
            den = (p * p * w64.unsqueeze(-1)).sum((1, 2))
            r = (p * (num / den).unsqueeze(1).unsqueeze(1) - uv64) * w64.unsqueeze(-1)
        return r.reshape(b, -1)

    shift64 = _lmdif(resid, torch.zeros(b, dtype=torch.float64, device=z.device))
    shift = shift64.to(z.dtype)

    # focal is recomputed in float32 from the float32 shift, as upstream does.
    d = z + shift.unsqueeze(-1)
    p = xy / d.unsqueeze(-1)
    weight = 1.0 if w is None else w.unsqueeze(-1)
    num = (p * uv * weight).sum(dim=(1, 2))
    den = (p * p * weight).sum(dim=(1, 2)).clamp(min=1e-12)
    return shift, num / den


@torch.no_grad()
def recover_focal_shift(points, mask=None, focal=None, downsample_size=(64, 64)):
    if focal is not None:
        return _original(points, mask=mask, focal=focal, downsample_size=downsample_size)

    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = F.interpolate(
        uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).squeeze(0).permute(1, 2, 0)
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    b = points_lr.shape[0]
    points_f = points_lr.float()
    xy = points_f[..., :2].reshape(b, -1, 2)
    z = points_f[..., 2].reshape(b, -1)
    uvf = uv_lr.float().reshape(-1, 2).unsqueeze(0).expand(b, -1, -1)
    w = None if mask_lr is None else mask_lr.reshape(b, -1).to(torch.float32)

    shift, focal_out = _solve_batch(uvf, xy, z, w)

    if w is not None:
        valid = w.sum(1) >= 2
        shift = torch.where(valid, shift, torch.zeros_like(shift))
        focal_out = torch.where(valid, focal_out, torch.ones_like(focal_out))

    return (
        focal_out.to(points.dtype).reshape(shape[:-3]),
        shift.to(points.dtype).reshape(shape[:-3]),
    )
