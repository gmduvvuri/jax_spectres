import numpy as np
import jax

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_compilation_cache", False)
jax.clear_caches()
from jax.lax import fori_loop
from jax_spectres.jax_spectres import (
    make_bins,
    get_resampling_arrays,
    jax_spectres_lean,
    jax_spectres,
    jnp,
)
from spectres.spectral_resampling import spectres as original_spectres
from spectres.spectral_resampling_numba import spectres_numba
from matplotlib import pyplot as plt
from timeit import default_timer

# Large portions of code copied nearly verbatim from
# https://github.com/ACCarnall/SpectRes/blob/master/examples/example_hizquasar.py
# obtained 2026-06-15 15:27:00 UTC
# with the intent to demonstrate its equivalence.

# Load the spectral data, the first column is wavelength,
# second is flux density, and third is flux density uncertainty
spectrum = np.loadtxt("./VST-ATLAS_J025.6821-33.4627.txt")

# Specify the grid of wavelengths onto which you wish to sample
regrid = np.arange(6510.0, 9385, 5.0) + 2.5

# Call the spectres function to resample the input spectrum
# or spectra to the new wavelength grid
original_spec_resample, original_spec_errs_resample = original_spectres(
    regrid,
    spectrum[:, 0],
    spectrum[:, 1],
    spectrum[:, 2],
)

jax_spec_resample, jax_spec_errs_resample = jax_spectres(
    regrid,
    spectrum[:, 0],
    spectrum[:, 1],
    spectrum[:, 2],
)

# Plotting code
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

ax1.plot(
    spectrum[:, 0],
    spectrum[:, 1] * 1e19,
    color="blue",
    label="Source",
)
ax1.plot(
    regrid,
    original_spec_resample * 1e19,
    color="red",
    label="Original",
)
ax1.plot(
    regrid,
    jax_spec_resample * 1e19,
    color="magenta",
    label="JAX",
)

ax2.plot(spectrum[:, 0], spectrum[:, 2] * 1e19, color="blue")
ax2.plot(regrid, original_spec_errs_resample * 1e19, color="red")
ax2.plot(regrid, jax_spec_errs_resample * 1e19, color="magenta")

fdens_str = r" [$10^{-19}\ \mathrm{W}\,\mathrm{m}^{-2}\,\mathrm{\AA}^{-1}$]"

ax1.set_ylabel(r"Flux Density" + fdens_str)
ax2.set_ylabel(r"Flux Density Uncertainty" + fdens_str)
ax2.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax1.set_xlim(6800, 9000)
ax1.legend()
plt.tight_layout()
plt.show()


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

ax1.plot(
    regrid,
    jax_spec_resample / original_spec_resample,
    color="k",
    label="JAX / Original",
)
ax2.plot(
    regrid,
    jax_spec_errs_resample / original_spec_errs_resample,
    color="k",
)

ax1.set_ylabel(r"Flux Density Ratio [--]")
ax2.set_ylabel(r"Flux Density Uncertainty Ratio [--]")
ax2.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax1.set_xlim(6800, 9000)
ax1.legend()
plt.tight_layout()
plt.show()

demo_sample_num = 1000000
old_edges, old_widths = make_bins(spectrum[:, 0])
new_edges, _ = make_bins(regrid)
(
    start_pos,
    stop_pos,
    width_factors,
    width_factor_sums,
    fill_indices,
    eval_mask,
) = get_resampling_arrays(regrid, old_edges, new_edges, old_widths)


def original_single_iter(i, val):
    _ = original_spectres(
        regrid,
        spectrum[:, 0],
        spectrum[:, 1],
        spectrum[:, 2],
    )
    return None


pre_original_spectres = default_timer()
_ = fori_loop(0, demo_sample_num, original_single_iter, None)
post_original_spectres = default_timer() - pre_original_spectres


def numba_single_iter(i, val):
    _ = spectres_numba(
        regrid,
        spectrum[:, 0],
        spectrum[:, 1],
        spectrum[:, 2],
    )
    return None


pre_numba_spectres = default_timer()
_ = fori_loop(0, demo_sample_num, numba_single_iter, None)
post_numba_spectres = default_timer() - pre_numba_spectres


def normal_single_iter(i, val):
    _ = jax_spectres(
        regrid,
        spectrum[:, 0],
        spectrum[:, 1],
        spectrum[:, 2],
    )
    return None


pre_jax_spectres_normal = default_timer()
_ = fori_loop(0, demo_sample_num, normal_single_iter, None)
post_jax_spectres_normal = default_timer() - pre_jax_spectres_normal


def lean_single_iter(i, val):
    _ = jax_spectres_lean(
        regrid,
        spectrum[:, 1],
        spectrum[:, 2],
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
        fill_indices,
        eval_mask,
        jnp.nan,
    )
    return None


pre_jax_spectres_lean = default_timer()
_ = fori_loop(0, demo_sample_num, lean_single_iter, None)
post_jax_spectres_lean = default_timer() - pre_jax_spectres_lean

print("Time for original: ", post_original_spectres)
print("Time for original (numba): ", post_numba_spectres)
print("Time for JAX default: ", post_jax_spectres_normal)
print("Time for JAX lean: ", post_jax_spectres_lean)
