# Astrophysical applications of gravitational lensing

A focused survey of *why* gravitational lensing matters, mapped onto the
case studies in `notebooks/` and the helpers in `lensing.lens`. Sources
and notation follow the Meneghetti UNIBO lecture notes (Ch. 4–6) plus the
review papers cited inline.

## 1. Cosmography from time-delay quasar lenses

**Question**: what is the value of the Hubble constant H₀?

**Answer via lensing**: a lensed quasar produces ≥ 2 images at different
arrival times because of two effects: (a) the geometric path length
differs between images; (b) the gravitational Shapiro delay accumulated
near the lens differs. Combined, they give the Fermat-potential
difference Δτ between images. The corresponding Δt is

$$
\Delta t = \frac{(1+z_L)}{c}\frac{D_L D_S}{D_{LS}}\, \Delta\tau
       \equiv \frac{D_{\Delta t}}{c}\, \Delta\tau,
$$

where `D_{Δt} ∝ 1/H₀`. Monitoring quasar variability gives Δt directly;
modelling the lens gives Δτ; ratio fixes D_{Δt} and hence H₀.

**State of the art**: H0LiCOW / TDCOSMO have measured H₀ = 73.3 ± 1.8
km/s/Mpc from 6 systems (Wong+ 2020) — a value in tension with the CMB
inference from Planck (67.4 ± 0.5). Whether this tension is "new
physics" or systematics in lens modelling is an active question.

**Implementation**: `lensing.lens.timedelay.refsdal_H0` and notebook 15.

**Limitations**:
* **Mass-sheet degeneracy**: replacing κ → λ κ + (1-λ) preserves image
  positions but rescales Δτ. Only external priors (line-of-sight
  convergence, stellar dynamics) break it.
* Lens density-profile slope. A single power-law lens can reach 2%
  precision; relaxing to a free profile can broaden the posterior to 5–8%
  per system (Birrer+ 2020, hierarchical TDCOSMO).

## 2. Dark matter from substructure detection

**Question**: how lumpy is the dark matter on sub-galactic scales?

**Answer via lensing**: extended Einstein rings carry tiny perturbations
from sub-halos along the line of sight. Comparing the observed arc to a
smooth lens model leaves residuals that pin down sub-halo positions and
masses (Vegetti & Koopmans 2009, Vegetti+ 2010 detection at 10⁸ M☉).

**State of the art**: ~10 confirmed detections in HST + Keck data;
Euclid, Rubin LSST and JWST will multiply this by orders of magnitude.
The mass function below 10⁹ M☉ tests cold-dark-matter (CDM) vs. warm
DM scenarios — WDM predicts fewer low-mass sub-halos.

**Implementation**: notebook 09 demonstrates the cluster + member +
shear composite that you would extend with sub-halos
(`CompositeLens([NFW, SIE, NIE_subhalo, ExternalShear])`).

## 3. Exoplanet hunting via microlensing

**Question**: what is the demographics of cold, bound and free-floating
planets in the Galaxy?

**Answer via lensing**: a star with a planet, acting as a binary lens
on a background star, produces caustic-crossing **light-curve
anomalies** with characteristic durations (a few hours for Earth-mass
companions). Because microlensing is sensitive to **mass** (not light),
it samples the demographics across distances and stellar types complementary
to RV/transit surveys.

**State of the art**: ~150 microlensing planets to date from
KMTNet/MOA/OGLE; Sumi+ 2011 announced ~1.8 free-floating Jupiter-mass
planets per main-sequence star. The Roman Space Telescope (launch 2027)
will multiply detections by ~100.

**Implementation**: `lensing.lens.BinaryPointMass` + notebook 08.

## 4. Primordial black holes from microlensing

**Question**: can compact dark-matter objects (e.g. primordial black
holes) make up a significant fraction of the dark matter?

**Answer via lensing**: monitoring millions of stars in M31 / SMC for
microlensing events constrains the abundance of compact lenses across
~10⁻⁹ to 10⁵ M☉. Subaru HSC's 7 hours of M31 imaging (Niikura+ 2019)
ruled out PBHs as 100% of the DM in 10⁻¹¹ – 10⁻⁶ M☉.

**Implementation**: the optical-depth and event-rate machinery is left
as an exercise for now (Meneghetti Ch. 4.5 has the formulas);
`lensing.lens.PaczynskiLightcurve` + a Monte-Carlo over event geometries
gives the underlying single-event probability.

## 5. Weak-lensing cosmic shear

**Question**: what is the matter power spectrum P(k, z) on cosmological
scales?

**Answer via lensing**: small (~1%) coherent distortions in the shapes
of background galaxies trace the projected line-of-sight matter
distribution. Two-point correlations of the shear field give P(k)
directly; combined with photometric redshifts, the redshift dependence
follows.

**State of the art**: KiDS-1000 (Asgari+ 2021), DES Y3 (Amon+ 2022) and
HSC Y3 measure σ₈ to ~3%. The Euclid mission (launched 2023) will tighten
this to <1%, opening a precision-cosmology test of dark energy.

**Implementation**: notebook 05 measures shear at the per-galaxy level
via `inference.fit_ellipticity`. A full weak-lensing pipeline would
average shear over millions of galaxies and propagate it through a
power-spectrum estimator; that is beyond this package's scope but the
per-galaxy ingredient is here.

## 6. Galaxy clusters as cosmic telescopes

**Question**: what does the high-redshift universe look like?

**Answer via lensing**: a strong-lensing galaxy cluster behind a faint
high-z source magnifies it by factors 10–100. The Hubble Frontier Fields
campaign (2013-2017) used this technique to push optical photometry to
mag 30, finding the most distant galaxies known at the time.

**Implementation**: notebook 09 with the MACS J1206 stack from
`data/raw/`. The mass model (NFW + members + shear) gives the local
magnification, hence the unmagnified luminosity of any source behind it.

## 7. Astrometric microlensing

**Question**: black-hole mass without electromagnetic counterpart?

**Answer via lensing**: even when the photometric magnification is
small (low μ but non-zero), the **image centroid** moves on the sky in
a measurable way. Gaia + HST observations of MOA-2011-BLG-191 yielded
the first **direct mass measurement of an isolated stellar-mass black
hole** (Sahu+ 2022, M = 7.1 ± 1.3 M☉, distance 1.6 kpc).

**Implementation**: not currently in the package — adding it would
require a binary-position helper that returns the magnitude-weighted
image centroid (a few lines on top of `BinaryPointMass`).

## 8. Multi-plane lensing and cosmic structure

**Question**: how do multiple structures along the line of sight
combine to lens a single source?

**Answer via lensing**: when the line-of-sight integral is not
dominated by a single lens, the deflection becomes path-ordered and
must be tracked plane by plane (Schneider, Ehlers & Falco 1992). For
typical galaxy-scale lenses the dominant contribution is still one
plane; for galaxy-cluster lenses a 2-plane treatment is increasingly
important for percent-precision modelling.

**Implementation**: not currently in the package; would require a
``MultiPlaneLens`` wrapper that ray-traces sequentially through a list
of lens planes, applying angular-diameter rescalings between each. This
is a natural extension once the rest of the package is solid.

---

## Reading list

* Meneghetti M., *Introduction to Gravitational Lensing*, Springer 2021
  (chapters 1-6 of the UNIBO course).
* Bartelmann M. & Schneider P., *Weak gravitational lensing*, Phys. Rep.
  340, 291 (2001).
* Schneider P., Kochanek C. S. & Wambsganss J., *Gravitational Lensing:
  Strong, Weak and Micro*, Saas-Fee 33, Springer 2006.
* Congdon A. B. & Keeton C. R., *Principles of Gravitational Lensing*,
  Springer 2018.
* Treu T., *Strong lensing by galaxies*, ARA&A 48, 87 (2010).
* Mandelbaum R., *Weak lensing for precision cosmology*, ARA&A 56, 393
  (2018).
