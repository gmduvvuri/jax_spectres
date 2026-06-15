JAX_SpectRes: A JAX Rewrite of "SpectRes"
=========================================

The following is a direct quotation of the original description of ``SpectRes``:

    "SpectRes is a Python function which efficiently resamples spectra and their associated uncertainties onto an arbitrary wavelength grid. The function works with any grid of wavelength values, including non-uniform sampling, and preserves the integrated flux. This may be of use for binning data to increase the signal to noise ratio, obtaining synthetic photometry, or resampling model spectra to match the sampling of observational data."

    --- `SpectRes <https://github.com/ACCarnall/SpectRes>`_.

This project attempts to rewrite ``SpectRes`` with two major differences:

1. This project writes all code to be usable with ``JAX`` to benefit from that framework's list of features that speed up computation. The original package has a ``numba`` option for similar reasons, but this is not always compatible with fitting tools like ``NumPyro``.
2. The function that performs resampling in the original ``spectres.spectres`` has many steps that are independent of the flux values of the spectrum being resampled, only having to do with evaluating overlaps between the original source and target new wavelength/frequency/spectral axis grid. In applications like fitting line profiles, the same wavelength grids are used repeatedly while the flux values of the spectrum to be resampled may change between iterations. This project aims to split these steps so that those wavelength resampling steps are only performed once, cutting computational cost for scenarios like the line-fitting problem.

Installation
------------

The goal is for ``JAX_SpectRes`` to be installed using ``pip`` BUT THIS IS NOT IMPLEMENTED YET.

.. code::

	pip install jax_spectres


Documentation
-------------

The code is developed at `github.com/gmduvvuri/jax_spectres <https://github.com/gmduvvuri/jax_spectres>`_. The examples folder has a direct analog to an example use of the code available at `github.com/ACCarnall/spectres <https://github.com/ACCarnall/spectres>`_.

The Astrophysics Data System reference for the resampling algorithm is available at `ui.adsabs.harvard.edu/abs/2017arXiv170505165C/abstract <https://ui.adsabs.harvard.edu/abs/2017arXiv170505165C/abstract>`_. Please cite this publication if you use either the original SpectRes or this rewrite in your research.


API Documentation
-----------------

.. autofunction:: jax_spectres.jax_spectres
