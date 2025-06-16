Heat Conduction Module
======================

The heat conduction module solves the following model

.. math::

 C\frac{\partial T}{\partial t} - \nabla \cdot \kappa\nabla T = Q


where :math:`C(\mathbf{r},t)` is the heat capacity, :math:`\kappa(T,t)` is the thermal conductivity tensor, :math:`T(\mathbf{r},t)` is the temperature, and :math:`Q(\mathbf{r},t)` is the heat source. Boundary conditions include:

- Dirichlet boundary conditions, where the temperature is fixed at the boundary.

.. math::

 T = f(t)

- Neumann boundary conditions, where the heat flux is fixed at the boundary.

.. math::

  \mathbf{J}\cdot \mathbf{\hat{n}} = f(T,t)


- Robin boundary conditions, where a linear combination of temperature and heat flux is fixed at the boundary.

.. math::

   \mathbf{J}\cdot \mathbf{\hat{n}} = h(t)\left[T - T_0(t)\right]
  
where :math:`h(t)` is the heat conductance and :math:`T_0(t)` is the reference temperature.


Nonlinear boundary conditions are also supported, where the boundary condition depends on the temperature at the boundary. For example, thermal radiation can be modeled as:

.. math::

   \mathbf{J}\cdot \mathbf{\hat{n}} = \sigma(T^4 - T_0^4)

where :math:`\sigma` is the Stefan-Boltzmann constant, and :math:`T_0` is the reference temperature. The module support linear and nonlinear heat conduction, as well as steady state and transient transport.



.. note::

   The model is fully differentiable, and enables to take gradients against any variable listed above, including time-dependent and nonlinear boundary conditions. 

