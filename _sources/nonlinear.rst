Nonlinear Thermal Transport
============================


In this example, we consider a material with the temperature-dependent thermal conductivity:

.. math::

   \kappa(T) = \kappa_A + \frac{\kappa_A - \kappa_B}{1+e^{\alpha \left(T-T_c \right)}},

where :math:`\kappa_A` and :math:`\kappa_B` are the low- and high-temperature limits, respectively, and :math:`T` is the transition temperature. This sigmoidal form is typical of phase-changing materials (PCMs). We design a material with prescribed cross-plane thermal conductivity. Note that we don't apply periodic boundary conditions since nonlinearity breaks translational symmetry.


.. code-block:: python

   from matinverse import Fourier
   from matinverse import BoundaryConditions
   from matinverse import Geometry2D
   from jax import numpy as jnp
   from matinverse.projection import projection
   from matinverse.filtering import Conic2D
   from matinverse import projection
   from matinverse.optimizer import MMA, State
   from matinverse import Movie2D
   import jax

   L = 10
   N = 70
   size = [L, L]
   grid = [N, N]

   resolution = N / L
   eta = 0.5

   geo = Geometry2D(grid, size, periodic=[False, False]) 

   T1 = 10
   T2 = 20

   DeltaT = T2 - T1

   bcs = BoundaryConditions(geo)

   bcs.temperature(lambda p: jnp.isclose(p[0], -size[0]/2), lambda batch, x, t: T1)
   bcs.temperature(lambda p: jnp.isclose(p[0],  size[0]/2), lambda batch, x, t: T2)

   # Add "nonlinear" to take into account temperature-dependent properties
   f = Fourier(geo, mode='nonlinear')

   # Given parameters
   TA = T1      
   KA = 1    
   TB = T2      
   KB = 2
   epsilon = 0.01  
   T_c = (TA + TB) / 2
   a = (2 / (TB - TA)) * jnp.log((1 - epsilon) / epsilon)

   R = L / 20
   filtering = Conic2D(geo, R)

   def transform(x, beta):
       x = filtering(x)
       return projection(x, beta)

   def get_kappa(T):
       return KA + (KB - KA) / (1 + jnp.exp(-a * (T - T_c)))

   kd = jnp.array([0.25]) 

   @jax.jit
   def objective(x, beta):
       x = transform(x, beta)

       thermal_conductivity = lambda batch, space, temp, t: get_kappa(temp) * jnp.eye(2) * x[space]

       out = f(thermal_conductivity=thermal_conductivity,
               boundary_conditions=bcs,
               batch_size=1)

       out['projected_rho'] = x

       kappa = out['kappa_effective'] / DeltaT

       g = jnp.linalg.norm(kappa - kd)

       return g, ({'kappa': [kappa]}, out)

   state = State()
   betas = [16, 32, 64, 128, 256, 512]

   x = jax.random.uniform(jax.random.PRNGKey(0), N**2)
   for beta in betas:   
       print(beta)

       x = MMA(lambda x: objective(x, beta),
               x0=x,
               state=state,
               nDOFs=N**2,
               maxiter=30)

   evolution = jnp.array([aux['projected_rho'] for aux in state.aux])

   Movie2D(evolution, geo, cmap='binary')
