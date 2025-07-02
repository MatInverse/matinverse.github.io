
Welcome to MatInverse's documentation!
=======================================

MatInverse is a JAX-based GPU-accelerated framework for differentiable multiphysics using the finite-volume method. Application include inverse design and automatic parameter extraction. Currently, we feature a heat conduction solver, with a mechanical solver in development. The framework is designed to be extensible, allowing users to implement their own physics and optimization algorithms. 

MatInverse is developed by Giuseppe Romano (romanog@mit.edu). Paper coming soon.

.. image:: https://docs.google.com/drawings/d/e/2PACX-1vRknGL9b64-biw__Z6S0PtE2voNBWakZH9OomnKOnyk3dUlqqZSFzJNQYZ7Owk4V-EZfy03G7UD1MFQ/pub?w=1440&h=1080
   :alt: MatInverse
   :width: 500px



\*     Under development


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 1
   :caption: Solvers

   heat_conduction

.. toctree::
   :maxdepth: 1
   :caption: Examples

   2Dmetamaterial
   3Dmetamaterial
   nonlinear
   transient

.. toctree::
   :maxdepth: 1
   :caption: API

   api/matinverse.fourier
