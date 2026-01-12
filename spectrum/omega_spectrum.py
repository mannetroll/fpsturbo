import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_pgm_p5(path: Path) -> np.ndarray:
    """
    Read a binary PGM P5 file into a numpy array.
    Handles comments in header and maxval up to 65535.
    """
    data = path.read_bytes()
    n = len(data)
    i = 0

    def _read_token() -> bytes:
        nonlocal i
        while i < n and data[i] in b" \t\r\n":
            i += 1
        if i < n and data[i] == ord("#"):
            while i < n and data[i] not in b"\r\n":
                i += 1
            return _read_token()
        j = i
        while j < n and data[j] not in b" \t\r\n":
            j += 1
        tok = data[i:j]
        i = j
        return tok

    magic = _read_token()
    if magic != b"P5":
        raise ValueError(f"Not a P5 PGM file: {magic!r}")

    w = int(_read_token())
    h = int(_read_token())
    maxval = int(_read_token())

    while i < n and data[i] in b" \t\r\n":
        i += 1

    pixels = data[i:]
    expected_8 = w * h
    expected_16 = w * h * 2

    if maxval <= 255:
        if len(pixels) < expected_8:
            raise ValueError("PGM data is shorter than expected")
        arr = np.frombuffer(pixels[:expected_8], dtype=np.uint8).reshape(h, w)
        return arr
    else:
        if len(pixels) < expected_16:
            raise ValueError("PGM data is shorter than expected for 16 bit")
        arr = np.frombuffer(pixels[:expected_16], dtype=">u2").reshape(h, w)
        return arr


def radial_mean_power_spectrum(a2d: np.ndarray, nbins: int | None = None):
    """
    Returns r_centers, pmean, good_mask.
    r is normalized radius k / k_Nyquist where k_Nyquist = N/2 (axis Nyquist).
    """
    a = np.asarray(a2d, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Input must be 2D")

    nz, nx = a.shape

    a = a - float(a.mean())

    w = np.fft.fft2(a)
    p = (w.real * w.real + w.imag * w.imag)

    kx = np.fft.fftfreq(nx) * nx
    kz = np.fft.fftfreq(nz) * nz
    kz_grid, kx_grid = np.meshgrid(kz, kx, indexing="ij")

    nmin = float(min(nx, nz))
    k_nyq = 0.5 * nmin
    r = np.sqrt(kx_grid * kx_grid + kz_grid * kz_grid) / k_nyq

    mask = r > 0.0
    r1 = r[mask].ravel()
    p1 = p[mask].ravel()

    r_max = math.sqrt(2.0)

    if nbins is None:
        nbins = max(64, int(2 * min(nx, nz)))

    idx = np.floor((r1 / r_max) * nbins).astype(np.int64)
    idx = np.clip(idx, 0, nbins - 1)

    psum = np.bincount(idx, weights=p1, minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins).astype(np.float64)

    good = cnt > 0.0
    pmean = np.zeros(nbins, dtype=np.float64)
    pmean[good] = psum[good] / cnt[good]

    r_edges = np.linspace(0.0, r_max, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    return r_centers, pmean, good


def add_reference_line(ax, x, y, slope, x1, x2, label):
    """
    Draw y = y1 * (x/x1)^slope through data point at x1.
    """
    if x1 <= 0.0 or x2 <= 0.0 or x2 == x1:
        return
    i0 = int(np.argmin(np.abs(x - x1)))
    y1 = float(y[i0])
    if not np.isfinite(y1) or y1 <= 0.0:
        return
    y2 = y1 * (x2 / x1) ** slope
    ax.loglog([x1, x2], [y1, y2], "--", linewidth=2)
    ax.text(x2, y2 * 1.7, label, fontsize=11, ha="left", va="center", color="black")


def fit_power_law(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float):
    """
    Fit y = A * x^m on log scale over x in [x_min, x_max].
    Returns A, m, n where n = -m (so y = A * x^{-n}).
    """
    sel = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y) & (y > 0.0) & (x > 0.0)
    if not np.any(sel):
        return None

    xf = x[sel]
    yf = y[sel]

    lx = np.log10(xf)
    ly = np.log10(yf)

    m, b = np.polyfit(lx, ly, 1)
    a = 10.0 ** b
    n = -m
    return a, m, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgm", type=str, help="Path to omega.pgm (P5)")
    ap.add_argument("--out", type=str, default="omega_spectrum.png", help="Output PNG")
    ap.add_argument("--nbins", type=int, default=0, help="Radial bins, 0 means auto")
    ap.add_argument("--fit_min", type=float, default=1.0e-3, help="Fit start x")
    ap.add_argument("--fit_max", type=float, default=0.5, help="Fit end x")
    ap.add_argument("--x1", type=float, default=1.0e-3, help="Anchor x for reference lines")
    ap.add_argument("--x2", type=float, default=0.7, help="End x for reference lines")
    args = ap.parse_args()

    pgm_path = Path(args.pgm)
    out_path = Path(args.out)

    img = read_pgm_p5(pgm_path)
    nbins = None if args.nbins <= 0 else int(args.nbins)

    r_centers, pmean, good = radial_mean_power_spectrum(img, nbins=nbins)

    x = r_centers[good]
    y = pmean[good]

    fit = fit_power_law(x, y, float(args.fit_min), float(args.fit_max))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.loglog(x, y, linewidth=1.5)
    ax.set_ylim(bottom=1.0)
    ax.set_title("Omega: radially averaged FFT power spectrum")
    ax.set_xlabel("normalized radius  k / k_Nyquist")
    ax.set_ylabel("radially averaged power")

    if fit is not None:
        a, m, n = fit
        y_fit = a * (x ** m)
        ax.loglog(x, y_fit, "--", linewidth=2)
        print(f"Fitted exponent n = {n:.6f} over [{args.fit_min:g}, {args.fit_max:g}]")

    x1 = float(args.x1)
    x2 = float(args.x2)

    add_reference_line(ax, x, y, slope=-2.0, x1=x1, x2=x2, label=r"$k^{-2}$")
    add_reference_line(ax, x, y, slope=-3.0, x1=x1, x2=x2, label=r"$k^{-3}$")

    meta = f"{pgm_path.name}\nshape={img.shape[1]} x {img.shape[0]}\nnbins={len(r_centers)}"
    ax.text(
        0.02,
        0.02,
        meta,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        linespacing=1.5,
        color="black",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(str(out_path))


if __name__ == "__main__":
    main()