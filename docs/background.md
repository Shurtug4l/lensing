# Gravitational lensing — theoretical background

This document is a self-contained primer on the physics implemented in this
package, structured to follow the **Meneghetti** *Lensing Gravitazionale*
graduate-course lecture notes (UNIBO MSc Astrophysics & Cosmology). For
each section we point at the module that implements the corresponding
quantity, so a reader of the code knows where to look.

> **Reference texts** — primary: Meneghetti, *Introduction to Gravitational
> Lensing*, Springer (UNIBO MSc lecture notes, the basis for this section).
> Secondary: Bartelmann & Schneider (2001) *Phys. Rep.*; Schneider, Kochanek
> & Wambsganss (2006) *Saas-Fee 33*; Congdon & Keeton (2018) *Principles of
> Gravitational Lensing*; Kormann, Schneider & Bartelmann (1994) *A&A*;
> Wright & Brainerd (2000) *ApJ* (NFW); Refsdal (1964) *MNRAS* (time
> delays); Tessore & Metcalf (2015) *A&A* (elliptical power-law).

## 0. Conventions and units

| Symbol               | Quantity                                          | Unit                            |
|----------------------|---------------------------------------------------|---------------------------------|
| `θ, β, α, x, y`      | angular position / deflection on the sky          | **arcsec**                      |
| `ξ`                  | physical impact parameter on the lens plane       | **kpc**  (Mpc for clusters)     |
| `D_L, D_S, D_LS`     | angular-diameter distances                        | **Mpc**                         |
| `D_Δt`               | time-delay distance ``(1+z_L) D_L D_S / D_LS``    | **Mpc**                         |
| `r_s`                | NFW scale radius (3-D)                            | **Mpc**                         |
| `θ_s`                | NFW angular scale radius ``r_s / D_L``            | **arcsec**                      |
| `θ_E`                | Einstein radius                                   | **arcsec**                      |
| `σ_v`                | velocity dispersion of the lens galaxy            | **km/s**                        |
| `M`                  | lens mass (microlensing)                          | **M_⊙** (solar masses)          |
| `v_rel`              | lens-source relative transverse velocity (galactic micro) | **km/s**                |
| `t, t_0, t_E`        | time axis, peak time, Einstein crossing time      | **days**                        |
| `Δt`                 | time delay between images                         | **days** (s in raw API output)  |
| `H_0`                | Hubble constant                                   | **km/s/Mpc**                    |
| `Ψ`, `τ`             | lensing potential, Fermat potential               | **arcsec²**                     |
| `κ, γ, μ`            | convergence, shear, magnification                 | **dimensionless**               |
| `q`                  | axis ratio of an ellipsoidal lens / light model   | dimensionless ∈ (0, 1]          |
| `e1, e2`             | ellipticity components, ``e = (1-q)/(1+q)``       | dimensionless                   |
| `Σ_crit`             | critical surface density                          | M_⊙ / Mpc²                      |

The package converts between unit systems explicitly inside
`lensing.cosmology` (uses astropy under the hood) and exposes plain
torch tensors with the units listed above; **no quantity has hidden
units**. A formula in a docstring will state the units for both inputs
and outputs.

**Coordinate convention**: image-plane (x, y) and source-plane (β₁, β₂)
are sky-aligned right-handed Cartesian coordinates; positive *x* is
east, positive *y* is north. The function `gl.data.coordinate_grid`
returns a `(2, npix, npix)` tensor with `xy[0]` the x-grid and `xy[1]`
the y-grid — the convention used by every model's `forward(xy)`.

**Sign convention**: deflection α is the *direction the source
appears to move*, i.e. β = θ − α. Convergence is positive everywhere
inside the lens (κ > 0). The shear sign in the SIE follows
Kormann+1994 (γ₁ = −κ cos 2φ, γ₂ = −κ sin 2φ); some authors use the
opposite sign for shear, so check before mixing with external code.

**Cosmology**: a flat ΛCDM with H₀ = 70 km/s/Mpc, Ω_m = 0.3, Ω_b = 0.05
is the default in `gl.cosmology.DEFAULT_COSMOLOGY`. All angular-
diameter distances are obtained via `astropy.cosmology.FlatLambdaCDM`.

---

## Outline (mirrors the Meneghetti lecture notes)

| Chapter | Topic                                | Implementation                                                                |
|---------|--------------------------------------|-------------------------------------------------------------------------------|
| 1       | Brief history of lensing             | —                                                                             |
| 2       | Light deflection                     | `lens.PointMassMicrolens`                                                     |
| 3       | The general lens                     | `lens.SIE`, `lens.timedelay`                                                  |
| 4       | Microlenses                          | `lens.PaczynskiLightcurve`, `lens.BinaryPointMass`                            |
| 5       | Extended lenses                      | `lens.SIE`, `lens.NIE`, `lens.NFW`, `lens.PowerLawSpherical`, `lens.SIS`      |
| 5.6/5.7 | External pert. & multiple components | `lens.ExternalShear`, `lens.CompositeLens`                                    |
| 5.8     | Time delays                          | `lens.timedelay`                                                              |
| 6       | Galaxies and clusters                | composite NFW + members + shear; data ``MACS J1206``                          |

---

## 1. Light deflection (Ch. 2)

A photon passing at impact parameter ξ from a point mass M is deflected by
an angle (in **General Relativity**, twice the Newtonian value)

$$
\hat{\alpha}(\xi) = \frac{4 G M}{c^2 \xi}.
$$

The factor of two comes from the GR prediction for the perihelion-shift
geodesic; it was confirmed by Eddington's 1919 expedition and is the
historical opening of the gravitational-lensing field.

For a continuous mass distribution Σ(ξ) projected onto a thin lens plane,
the deflection generalizes by linear superposition,

$$
\hat{\boldsymbol\alpha}(\boldsymbol\xi)
= \frac{4 G}{c^2} \int d^2\xi'\, \Sigma(\boldsymbol\xi')\,
  \frac{\boldsymbol\xi - \boldsymbol\xi'}{|\boldsymbol\xi - \boldsymbol\xi'|^2},
$$

an exact result for thin-screen lenses (the lens's line-of-sight extent is
much smaller than its angular-diameter distance to source and observer).

**Implemented**: `PointMassMicrolens` (point mass), `SIE.deflection`,
`NFW.deflection`, `BinaryPointMass.deflection`, `NIE.deflection`,
`PowerLawSpherical.deflection`, `ExternalShear.deflection`,
`CompositeLens.deflection` (linear superposition).

## 2. The lens equation (Ch. 3.1)

In angular coordinates, the **reduced deflection** is

$$
\boldsymbol\alpha(\boldsymbol\theta) = \frac{D_{LS}}{D_S}\,
\hat{\boldsymbol\alpha}(D_L \boldsymbol\theta),
$$

and the **lens equation** maps observed image positions θ to the true
source position β,

$$
\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha(\boldsymbol\theta).
$$

Distances `D_L, D_S, D_{LS}` are *angular-diameter distances* to lens,
source, and from lens to source, computed by `lensing.cosmology.Cosmology`
through astropy. The lens equation is a 2-D non-linear root-finding
problem that admits a unique solution for weak lensing and a finite
multi-image set for strong lensing.

**Implemented**: every lens model exposes `ray_trace(x, y) -> (β_x, β_y)`.
For SIE the inverse problem (solve for θ given β) is in
`SIE.solve_image_positions`.

## 3. Lensing potential, convergence, shear, magnification (Ch. 3.2-3.4)

Every thin lens admits a (dimensionless) lensing potential `Ψ̂` such that
``α = ∇ Ψ̂``. Its second derivatives organise into the **convergence** κ
and the two-component **shear** γ:

$$
\kappa = \tfrac12 \nabla^2 \Psi, \qquad
\gamma_1 = \tfrac12 (\partial_{xx} - \partial_{yy})\Psi, \qquad
\gamma_2 = \partial_{xy}\Psi.
$$

The Jacobian of the lens equation is

$$
A_{ij}(\theta) = \frac{\partial \beta_i}{\partial \theta_j}
              = \delta_{ij} - \frac{\partial^2 \Psi}{\partial \theta_i \partial \theta_j}
              = \begin{pmatrix} 1 - \kappa - \gamma_1 & -\gamma_2 \\ -\gamma_2 & 1 - \kappa + \gamma_1 \end{pmatrix}.
$$

**Magnification**: a small source patch is mapped onto an image patch
whose area is multiplied by `|det A|^{-1}`; equivalently,

$$
\mu = \frac{1}{(1-\kappa)^2 - |\gamma|^2}.
$$

**Critical curves** are the loci where det A = 0 (μ → ∞); their image
through the lens equation is the **caustic**, which delimits the multi-
image regions in source plane.

**Implemented**: `SIE.kappa`, `SIE.shear`, `SIE.magnification`,
`NFW.kappa`, `BinaryPointMass.magnification_map` (via inverse ray-shooting),
`SIE.tangential_caustic`, `SIE.tangential_critical`. The general
*finite-difference detA* recipe used in notebooks 09 and 14 is in the
`critical_curves` helper of those notebooks.

## 4. Second-order distortions and flexion (Ch. 3.5)

A small circular source at β with radius R becomes an ellipse on the image
plane to first order. The shear γ tilts and stretches it; the convergence κ
scales it isotropically. To **second order** in the source size, the same
map produces *flexion* — the bending of the source's image into a comma
shape, with two components

$$
F = (\partial_x + i \partial_y)(\kappa + \gamma_*),
\quad
G = (\partial_x + i \partial_y)\gamma,
$$

(the first and third moments of the third derivatives of Ψ). Flexion is
the workhorse of *cluster-scale weak lensing* (Bacon+ 2006, Goldberg &
Bacon 2005) where it picks up small-scale density fluctuations missed by
shear alone.

We do not currently have a closed-form `flexion()` API, but every
deflection in the package is differentiable, so the third-order
derivatives can be obtained on demand via repeated `torch.autograd.grad`.

## 5. Microlenses (Ch. 4)

For small lens masses (M ≲ 10⁵ M☉) the image separation is below the
PSF, so we observe a single, time-varying brightness:

$$
\mu(t) = \frac{u^2 + 2}{u\sqrt{u^2 + 4}},\qquad
u(t) = \sqrt{u_0^2 + \big[(t - t_0)/t_E\big]^2}.
$$

The **Paczynski light curve** depends on four observables: source flux
`f_S`, peak time `t_0`, impact parameter `u_0` (in units of θ_E), and the
**Einstein crossing time** `t_E`. The physical parameters (M, v_rel, D_L,
D_S) are degenerate at the level of the light curve — notebook 01
demonstrates this explicitly.

For **binary lenses** the deflection has up to 5 images (5th-order
complex polynomial roots), and the caustic structure has three regimes
(close, intermediate, wide) controlled by the projected separation `d` in
units of the system Einstein radius. Caustic-crossing events produce
sharp light-curve spikes that are the basis of microlensing exoplanet
detection (OGLE, KMTNet, Roman).

**Optical depth and event rate** (Meneghetti Ch. 4.5): the probability
that a randomly chosen background star is currently magnified above some
threshold is τ = π θ_E² · n_lenses, with the line-of-sight integral over
the lens distribution. This is what the MACHO and EROS surveys
measured to constrain the dark-matter fraction in compact halo objects.

**Implemented**: `PointMassMicrolens` (with physical params for didactic
plots), `PaczynskiLightcurve` (minimal model for fitting),
`BinaryPointMass.magnification_map` (inverse ray-shooting),
`BinaryPointMass.critical_curves` (numerical det A = 0).

## 6. Extended lenses: SIE, NIE, NFW, power-law (Ch. 5)

### 6.1 Singular Isothermal Sphere (SIS) and power-law

The **SIS** is the n = 2 case of the spherical power law

$$
\kappa(x) = \tfrac{3-n}{2}\, x^{1-n}, \quad
\alpha(x) = x^{2-n}, \quad
\Psi(x) = \frac{x^{3-n}}{3-n} \;.
$$

It is famous because the deflection is *constant* in magnitude (equal to
θ_E), so any source displacement is encoded entirely in the geometric
side of the lens equation. The 2-image configuration of an SIS lens has
images on opposite sides of the lens, with separation ≈ 2 θ_E.

For 1 < n < 2 the lens is shallower (extra dark-matter halo); for 2 < n
< 3 it is steeper (more concentrated baryonic core).

**Implemented**: `lens.PowerLawSpherical`, `lens.SIS`. The elliptical
generalisation (Tessore & Metcalf 2015) is closed-form via `2F1`; we
defer it because `lens.SIE` and `lens.NIE` already cover the most
commonly used elliptical isothermal lenses.

### 6.2 Singular Isothermal Ellipsoid (SIE)

Kormann, Schneider & Bartelmann 1994 derive the closed-form deflection of
the elliptical isothermal lens, which we reproduce verbatim in
`SIE.alpha_polar`. The Einstein radius is

$$
\theta_E = 4\pi (\sigma_v/c)^2\, D_{LS}/D_S,
$$

where σ_v is the velocity dispersion of the lens galaxy. The image
positions for a given source β are roots of a 1-D function F(ϕ) = 0
along the lens-aligned polar angle: this is the **bisection-on-sign-
changes** strategy implemented in `SIE.solve_image_positions`. It is
robust (no derivatives required) and gives the 2 or 4 images of the
standard SIE multi-image regions.

### 6.3 Non-singular Isothermal Ellipsoid (NIE)

Adding a finite core radius `xi_c` regularises the central singularity
of the SIE, producing an extra **radial** caustic and (for sufficiently
small core) a third image close to the lens centre. The caustic
topology has four regimes controlled by the dimensionless ratio
`x_c / q^{3/2}`; see notebook 14 for a numerical exploration.

**Implemented**: `lens.NIE` with closed-form deflection (Kormann+ 1994).

### 6.4 Navarro-Frenk-White (NFW) halo

Galaxy clusters are modelled to first order as NFW halos:

$$
\rho(r) = \frac{\rho_s}{(r/r_s)\,(1 + r/r_s)^2}.
$$

The projected surface density and convergence are due to Wright &
Brainerd 2000 — the formulas live in `NFW.kappa` and `NFW.deflection`,
parameterised by the angular scale radius `θ_s = r_s / D_L` and the
characteristic convergence `κ_s`. The NFW concentration ``c =
r_{200}/r_s`` is the link to the halo mass M_{200} and the Einstein
radius (cluster-scale Einstein radii are typically ~10–50 arcsec).

### 6.5 External shear

A galaxy lens often lives inside a tidal field from a more extended
structure (group, filament). The first-order expansion is a constant
external shear,

$$
\Psi_\text{ext} = \tfrac{\gamma_1}{2}(x^2 - y^2) + \gamma_2\, xy,
$$

added linearly via `CompositeLens([SIE(...), ExternalShear(g1, g2)])`.

## 7. Time delays and Fermat potential (Ch. 3.6, 5.8)

Different images of the same source have different arrival-time delays:

$$
\Delta t(\theta) = \frac{(1+z_L)}{c} \frac{D_L D_S}{D_{LS}}\,
\bigg[\tfrac12 (\theta - \beta)^2 - \hat\Psi(\theta)\bigg]
\equiv \frac{D_{\Delta t}}{c}\,\tau(\theta; \beta).
$$

The square-bracket term is the **Fermat potential** τ. Its stationary
points are the image positions (Fermat's principle); its second-derivative
matrix is exactly A (the magnification Jacobian). The dimensional
prefactor `D_{Δt} = (1+z_L)D_L D_S / D_LS` is the **time-delay distance**;
it scales as 1/H₀.

Inverting an observed Δt for D_{Δt} (and hence H₀) is **Refsdal
cosmography** (Refsdal 1964), today applied to ~6 quasar-pair lenses by
H0LiCOW / TDCOSMO with ~2% precision per system.

The dominant systematic is the **mass-sheet degeneracy** (Falco,
Gorenstein & Shapiro 1985): replacing κ → λ κ + (1-λ) leaves the image
configuration invariant but rescales τ → λ τ. External information
(stellar dynamics, line-of-sight κ from N-body sims) is required to
break it.

**Implemented**: `lens.timedelay.fermat_potential`,
`lens.timedelay.time_delay_distance`,
`lens.timedelay.time_delay_seconds`, `lens.timedelay.refsdal_H0`. See
notebook 15 for a worked example.

## 8. Light models (Sérsic family)

Galaxy surface brightness profiles obey the Sérsic (1968) law

$$
I(R) = I_e \exp\!\left[-b_n\!\left((R/R_e)^{1/n} - 1\right)\right],
$$

with R the elliptical radius. n = 1 ↔ exponential disk; n = 4 ↔ de
Vaucouleurs; n ~ 2-2.5 ↔ typical spheroid. The constant `b_n` is fixed by
the half-light condition; we use the Ciotti & Bertin (1999) polynomial
expansion in `utils.sersic_bn`.

The **core-Sérsic** profile (Graham et al. 2003) replaces the inner
power-law cusp with a smooth break at `R_b` — relevant for the most
massive ellipticals where binary-black-hole inspirals deplete the core.

We parameterise the ellipse via (e1, e2), continuous and NaN-free at e=0;
q and PA are derived for reporting only. See `light/base.py` for the
algebraic derivation.

## 9. Inference pipeline

* **Maximum-likelihood / chi-squared**: differentiable models +
  autodiff ⇒ trivial gradient descent. We use Adam followed by an
  L-BFGS polish; see `inference.fit`. Constraint enforcement is done
  via `enforce_constraints()` on each model after each step.
* **Bayesian posteriors**: Pyro NUTS (`inference.run_nuts`).
* **Weak-lensing ellipticity**: forward-fit Sérsic + PSF
  (`inference.fit_ellipticity`) or non-parametric quadrupole moments
  (`inference.kaiser_squires_estimator`).
* **Deep learning**: `lensing.ml` for CNN/DNN/U-Net pipelines on
  synthetic lensed images.
* **CPU vs MPS performance**: `lensing.benchmarks` and notebook 13
  measure the trade-offs.
