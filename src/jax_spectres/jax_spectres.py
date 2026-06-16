from jax import numpy as jnp
from jax import jit
from jax.lax import while_loop, fori_loop


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
    edges = jnp.zeros(waves.shape[0] + 1)
    widths = jnp.zeros(waves.shape[0])
    edges = edges.at[0].set(waves[0] - (waves[1] - waves[0]) / 2)
    widths = widths.at[-1].set(waves[-1] - waves[-2])
    edges = edges.at[-1].set(waves[-1] + (waves[-1] - waves[-2]) / 2)
    edges = edges.at[1:-1].set((waves[1:] + waves[:-1]) / 2)
    widths = widths.at[:-1].set(edges[1:-1] - edges[:-2])
    return edges, widths


def create_width_factor_array(out_shape, old_widths):
    return jnp.ones(out_shape) * old_widths[:, jnp.newaxis]


def edge_check(edge_state):
    counter, old_edges, edge_limit = edge_state
    return old_edges[counter] < edge_limit


def add_counter(edge_state):
    counter, old_edges, edge_limit = edge_state
    return (counter + 1, old_edges, edge_limit)


@jit
def jax_spectres_single_bin(
    old_fluxes_bin_slice,
    old_errs_bin_slice,
    width_factors_bin_slice,
    width_factor_sum,
):
    new_flux = (
        jnp.sum(
            width_factors_bin_slice * old_fluxes_bin_slice,
            axis=-1,
        )
        / width_factor_sum
    )
    new_err = (
        jnp.sqrt(
            jnp.sum(
                (width_factors_bin_slice * old_errs_bin_slice) ** 2,
                axis=-1,
            )
        )
        / width_factor_sum
    )
    return new_flux, new_err


def get_resampling_arrays(
    new_waves,
    old_edges,
    new_edges,
    old_widths,
):
    fill_indices = jnp.where(
        jnp.logical_or(
            (new_edges[:-1] < old_edges[0]), (new_edges[1:] > old_edges[-1])
        ),
        True,
        False,
    )
    width_factors = create_width_factor_array(
        (old_widths.shape[0], new_waves.shape[0]), old_widths
    )
    width_factor_sums = jnp.ones((new_waves.shape[0]))
    start_pos = jnp.zeros((width_factor_sums.shape[0]), dtype=jnp.int64)
    stop_pos = jnp.zeros_like(start_pos)
    eval_mask = jnp.where(fill_indices, False, jnp.full_like(new_waves, True))
    false_mask = jnp.full_like(eval_mask, False)
    old_wave_indices = jnp.arange(old_widths.shape[0])

    start = 0
    stop = 0
    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_waves.shape[0]):
        # Find first old bin which is partially covered by the new bin
        # while old_edges[start + 1] <= new_edges[j]:
        #     start += 1
        start_mod, _, _ = while_loop(
            edge_check, add_counter, (start + 1, old_edges, new_edges[j])
        )
        start = start_mod - 1
        start_pos = start_pos.at[j].set(start)

        # Find last old bin which is partially covered by the new bin
        # while old_edges[stop + 1] < new_edges[j + 1]:
        #     stop += 1
        stop_mod, _, _ = while_loop(
            edge_check, add_counter, (stop + 1, old_edges, new_edges[j + 1])
        )
        stop = stop_mod - 1
        stop_pos = stop_pos.at[j].set(stop)

        # If new bin is fully inside an old bin start and stop are equal
        # Otherwise multiply the first and last old bin widths by P_ij
        start_factor = (old_edges[start + 1] - new_edges[j]) / (
            old_edges[start + 1] - old_edges[start]
        )

        end_factor = (new_edges[j + 1] - old_edges[stop]) / (
            old_edges[stop + 1] - old_edges[stop]
        )

        width_factors = width_factors.at[start, j].multiply(start_factor)
        width_factors = width_factors.at[stop, j].multiply(end_factor)
        width_factors_wave_slice = width_factors[:, j]
        slice_condition = jnp.logical_and(
            old_wave_indices >= start, old_wave_indices < stop + 1
        )
        width_factors_bin_slice = jnp.where(
            slice_condition, width_factors_wave_slice, 0.0
        )
        width_factor_sums = width_factor_sums.at[j].set(
            jnp.sum(width_factors_bin_slice)
        )
    eval_mask = jnp.where(start_pos == stop_pos, false_mask, eval_mask)
    return (
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
        fill_indices,
        eval_mask,
    )


@jit
def spectres_loop_iter(
    bin_index,
    loop_state,
):
    (
        width_factor_flux_vals,
        width_factor_errs_vals,
        pos_flux,
        pos_errs,
        old_flux_indices,
        spec_fluxes,
        spec_errs,
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
    ) = loop_state
    start_index = start_pos[bin_index]
    stop_index = stop_pos[bin_index]
    width_factor_sum = width_factor_sums[bin_index]
    slice_condition = jnp.logical_and(
        old_flux_indices >= start_index,
        old_flux_indices < stop_index + 1,
    )
    spec_fluxes_bin_slice = jnp.where(slice_condition, spec_fluxes, 0.0)
    spec_errs_bin_slice = jnp.where(slice_condition, spec_errs, 0.0)
    width_slice = width_factors[:, bin_index]
    width_factors_bin_slice = jnp.where(slice_condition, width_slice, 0.0)
    new_flux, new_err = jax_spectres_single_bin(
        spec_fluxes_bin_slice,
        spec_errs_bin_slice,
        width_factors_bin_slice,
        width_factor_sum,
    )
    width_factor_flux_vals = width_factor_flux_vals.at[bin_index].set(new_flux)
    width_factor_errs_vals = width_factor_errs_vals.at[bin_index].set(new_err)
    pos_flux = pos_flux.at[bin_index].set(spec_fluxes[start_index])
    pos_errs = pos_errs.at[bin_index].set(spec_errs[start_index])
    new_loop_state = (
        width_factor_flux_vals,
        width_factor_errs_vals,
        pos_flux,
        pos_errs,
        old_flux_indices,
        spec_fluxes,
        spec_errs,
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
    )
    return new_loop_state


@jit
def jax_spectres_lean(
    new_waves,
    spec_fluxes,
    spec_errs,
    start_pos,
    stop_pos,
    width_factors,
    width_factor_sums,
    fill_indices,
    eval_mask,
    fill=jnp.nan,
):
    pos_flux = jnp.where(fill_indices, fill, jnp.zeros_like(new_waves))
    pos_errs = jnp.where(fill_indices, fill, jnp.zeros_like(new_waves))
    width_factor_flux_vals = jnp.copy(pos_flux)
    width_factor_errs_vals = jnp.copy(pos_errs)
    old_flux_indices = jnp.arange(spec_fluxes.shape[0])
    loop_state = (
        width_factor_flux_vals,
        width_factor_errs_vals,
        pos_flux,
        pos_errs,
        old_flux_indices,
        spec_fluxes,
        spec_errs,
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
    )
    out_loop = fori_loop(0, new_waves.shape[0], spectres_loop_iter, loop_state)
    new_fluxes = jnp.where(eval_mask, out_loop[0], out_loop[2])
    new_errs = jnp.where(eval_mask, out_loop[1], out_loop[3])
    return new_fluxes, new_errs


def jax_spectres(
    new_waves,
    spec_waves,
    spec_fluxes,
    spec_errs,
    fill=jnp.nan,
):
    old_edges, old_widths = make_bins(spec_waves)
    new_edges, _ = make_bins(new_waves)
    (
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
        fill_indices,
        eval_mask,
    ) = get_resampling_arrays(new_waves, old_edges, new_edges, old_widths)
    new_fluxes, new_errs = jax_spectres_lean(
        new_waves,
        spec_fluxes,
        spec_errs,
        start_pos,
        stop_pos,
        width_factors,
        width_factor_sums,
        fill_indices,
        eval_mask,
        fill,
    )
    return new_fluxes, new_errs
