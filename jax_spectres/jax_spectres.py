from __future__ import print_function, division, absolute_import
import warnings

from jax import numpy as jnp


def make_bins(waves):
    """
    Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins. Assumes series is sorted in
    ascending order.

    Parameters
    ----------

    waves : jax.numpy.array
        Array containing the wavelength points to find edges and
        widths for.

    Returns
    -------

    edges : jax.numpy.array
        Array containing the bin edges of points listed in waves.

    widths : jax.numpy.array
        Array containing the widths of the bins bounded by edges.
    """
    zeroth_edge = waves[0] - ((waves[1] - waves[0]) / 2)
    inner_edges = (waves[1:] + waves[:-1]) / 2
    last_edge = waves[-1] + ((waves[-1] - waves[-2]) / 2)
    edges = jnp.vstack([jnp.array([zeroth_edge], inner_edges[:-1], jnp.array([last_edge]])
    widths = jnp.diff(edges)
    return edges, widths


def original_spectres(new_waves, spec_waves, spec_fluxes, spec_errs=None, fill=None,):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    Parameters
    ----------

    new_waves : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra. Assumes array is sorted in ascending order.

    spec_waves : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra. Assumes array is sorted in ascending order.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_waves, last dimension must correspond to the shape of
        spec_waves. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_waves extends outside the wavelength range in spec_waves
        this value will be used as a filler in new_fluxes and new_errs.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, last dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_waves = spec_waves
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    old_edges, old_widths = make_bins(old_waves)
    new_edges, new_widths = make_bins(new_waves)

    # Generate output arrays to be populated
    # Set any values where new_edges are beyond bounds to fill,
    # and zero otherwise.
    new_fluxes = jnp.zeros(old_fluxes[..., 0].shape + new_waves.shape)
    new_fluxes = new_fluxes.at[.., :].set(jnp.where(jnp.logical_or((new_edges[:-1] < old_edges[0]), (new_edges[1:] > old_edges[-1])), fill, 0.0))

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = jnp.copy(new_fluxes)

    start = 0
    stop = 0
    width_factors = jnp.ones((old_widths.shape[0], new_waves.shape[0])) * old_widths

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_waves.shape[0]):
        
        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

      # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes = new_fluxes.at[..., j].set(old_fluxes[..., start])
            if old_errs is not None:
                new_errs = new_errs.at[..., j].set(old_errs[..., start])

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            width_factors = width_factors.at[start, j].multiply(start_factor)
            width_factors = width_factors.at[stop, j].multiply(end_factor)

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = width_factors[start:stop+1, j]*old_fluxes[..., start:stop+1]
            width_factor_sum = jnp.sum(width_factors[start:stop + 1, j])
            new_fluxes[..., j] = jnp.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= width_factor_sum

            if old_errs is not None:
                e_wid = width_factors[start:stop+1, j]*old_errs[..., start:stop+1]
                new_errs[..., j] = jnp.sqrt(jnp.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= width_factor_sum

  return new_fluxes, new_errs


def jax_spectres_single_bin(bin_index, old_fluxes, old_errs, start_pos, stop_pos, width_factors, width_factor_sums,):
  new_flux = jnp.sum(width_factors[start_pos[bin_index] : stop_pos[bin_index] + 1, bin_index] * old_fluxes[..., start_pos[bin_index] : stop_pos[bin_index] + 1], axis=-1) / width_factor_sums[bin_index]
  new_err = jnp.sqrt(jnp.sum((width_factors[start_pos[bin_index] : stop_pos[bin_index] + 1, bin_index] * old_errs[..., start_pos[bin_index] : stop_pos[bin_index] + 1])**2, axis=-1)) / width_factor_sums[bin_index]
  return new_flux, new_err


def get_resampling_arrays(new_waves, old_edges, new_edges,):
  fill_indices = jnp.where(jnp.logical_or((new_edges[:-1] < old_edges[0]), (new_edges[1:] > old_edges[-1])), True, False)
  width_factors = jnp.ones((old_widths.shape[0], new_waves.shape[0])) * old_widths
  width_factor_sums = jnp.ones((new_waves.shape[0]))
  start_pos = jnp.ones_like(width_factor_sums)
  stop_pos = jnp.ones_like(width_factor_sums)
  pos_equal_flux_vals = jnp.zeros_like(width_factors)
  pos_equal_errs_vals = jnp.zeros_like(width_factors)
  eval_mask = jnp.copy(fill_indices)

  start = 0
  stop = 0
  # Calculate new flux and uncertainty values, looping over new bins
  for j in range(new_waves.shape[0]):
      
      # Find first old bin which is partially covered by the new bin
      while old_edges[start+1] <= new_edges[j]:
          start += 1
      start_pos = start_pos.at[j].set(start)
      # Find last old bin which is partially covered by the new bin
      while old_edges[stop+1] < new_edges[j+1]:
          stop += 1
      stop_pos = stop_pos.at[j].set(stop)

    # If new bin is fully inside an old bin start and stop are equal
      if stop == start:
        pos_equal_flux_vals = pos_equal_flux_vals.at[..., j].set(old_fluxes[..., start])
        pos_equal_errs_vals = pos_equal_errs_vals.at[..., j].set(old_errs[..., start])
        eval_mask = eval_mask.at[j].set(False)
                                                              

      # Otherwise multiply the first and last old bin widths by P_ij
      else:
          start_factor = ((old_edges[start+1] - new_edges[j])
                          / (old_edges[start+1] - old_edges[start]))

          end_factor = ((new_edges[j+1] - old_edges[stop])
                        / (old_edges[stop+1] - old_edges[stop]))

          width_factors = width_factors.at[start, j].multiply(start_factor)
          width_factors = width_factors.at[stop, j].multiply(end_factor)

          # Populate new_fluxes spectrum and uncertainty arrays
          width_factor_sums = width_factor_sums.at[j].set(jnp.sum(width_factors[start:stop + 1, j]))
  return start_pos, stop_pos, width_factors_ width_factor_sums, fill_indices, pos_equal_flux_vals, pos_equal_errs_vals, eval_mask


def jax_spectres_lean(new_waves, spec_fluxes, spec_errs, start_pos, stop_pos, width_factors, width_factor_sums, fill_indices, pos_equal_flux_vals, pos_equal_errs_vals, eval_mask, fill=jnp.nan,):
  new_fluxes = pos_equal_flux_vals.at[..., fill_indices].set(fill)
  new_errs = pos_equal_errs_vals.at[..., fill_indices].set(jnp.nan)

  for bin_index in range(new_waves.shape[0]):
    if eval_mask[bin_index]:
      new_flux, new_err = jax_spectres_single_bin(bin_index, spec_fluxes, spec_errs, start_pos, stop_pos, width_factors, width_factor_sums)
      new_fluxes = new_fluxes.at[bin_index].set(new_flux)
      new_errs = new_errs.at[bin_index].set(new_err)
  return new_fluxes, new_errs


def jax_spectres(new_waves, spec_waves, spec_fluxes, spec_errs, fill=jnp.nan,):
  old_edges, old_widths = make_bins(spec_waves)
  new_edges, new_widths = make_bins(new_waves)

  start_pos, stop_pos, width_factors, width_factor_sums, fill_indices, pos_equal_flux_vals, pos_equal_errs_vals, eval_mask = get_resampling_arrays(old_edges, new_edges)
  new_fluxes, new_errs = jax_spectres_lean(new_waves, spec_fluxes, spec_errs, start_pos, stop_pos, width_factors, width_factor_sums, fill_indices, pos_equal_flux_vals, pos_equal_errs_vals, eval_mask, fill)
  return new_fluxes, new_errs
