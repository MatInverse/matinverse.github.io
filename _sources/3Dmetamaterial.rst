Thermal metamaterial (3D)
===========================
In this example, we design a metamaterial with effective thermal conductivity

.. math::

    \bar{\kappa}_{\textrm{eff}} = \begin{bmatrix}
        0.2 & 0.0 & 0.0 \\
        0.0 & 0.3 & 0.0 \\
        0.0 & 0.0 & 0.4
    \end{bmatrix}


We solve the heat conduction for follwing six different directions of the unit sphere, 


:math:`\hat{\mathbf{n}}_0 = \hat{\mathbf{x}}`, 
:math:`\hat{\mathbf{n}}_1 = \hat{\mathbf{y}}`, 
:math:`\hat{\mathbf{n}}_2 = \hat{\mathbf{z}}`, 
:math:`\hat{\mathbf{n}}_3 = \frac{\sqrt{2}}{2}\hat{\mathbf{x}} + \frac{\sqrt{2}}{2}\hat{\mathbf{y}}`, 
:math:`\hat{\mathbf{n}}_4 = \frac{\sqrt{2}}{2}\hat{\mathbf{y}} + \frac{\sqrt{2}}{2}\hat{\mathbf{z}}`, and
:math:`\hat{\mathbf{n}}_5 = \frac{\sqrt{2}}{2}\hat{\mathbf{x}} + \frac{\sqrt{2}}{2}\hat{\mathbf{z}}`, 

leading to the linear system

.. math::

    \begin{bmatrix}
        \kappa_{xx} \\
        \kappa_{yy} \\
        \kappa_{zz} \\
        \kappa_{xy} \\
        \kappa_{xz} \\
        \kappa_{yz}
    \end{bmatrix}
    = 
    \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0\\
        \frac{1}{2} & \frac{1}{2} & 0 & 1 & 0 & 0\\
        0 &\frac{1}{2} & \frac{1}{2} & 0 & 1 & 0\\
        \frac{1}{2} & 0 & \frac{1}{2} & 0 & 0 & 1\\
    \end{bmatrix}
    \begin{bmatrix}
        \kappa_0 \\
        \kappa_1 \\
        \kappa_2 \\
        \kappa_3 \\
        \kappa_4 \\
        \kappa_5 \\
    \end{bmatrix},


where :math:`\kappa_i` is the effective thermal conductivity in the direction :math:`\hat{\mathbf{n}}_i`, computed by :math:`\kappa_i = \hat{\mathbf{n}}_i^T\kappa \hat{\mathbf{n}}_i`.
Density-based topology optimization is performed using the three-field approach, as outlined 
`here <https://link.springer.com/article/10.1007/s00158-022-03392-w>`_ for thermal materials. 

Notes:

- We use the recently introduced `subpixel smoothed projection <https://arxiv.org/abs/2503.20189>`_. 

- We specify ``batch_size=3`` when creating the ``Fourier`` object.

- With ``collapse_direct=True``, we use the same matrix assembly for each direction, which is more efficient. This option is only available for the direct solver. 


.. code-block:: python

    from matinverse import Geometry3D,BoundaryConditions,Fourier
    from matinverse.projection import projection
    from matinverse.visualizer import plot3D
    from matinverse.filtering import Conic3D
    from matinverse.optimizer import MMA,State
    from jax import numpy as jnp
    from functools import partial

    import jax

    L = 10
    size = [L,L,L]
    N = 20
    grid = [N,N,N]

    geo = Geometry3D(grid,size,periodic=[True,True,True]) #


    phi  = jnp.array([0,jnp.pi/2,0,jnp.pi/4,jnp.pi/2,0])
    theta = jnp.array([jnp.pi/2,jnp.pi/2,0,jnp.pi/2,jnp.pi/4,jnp.pi/4])
    directions = jnp.array([jnp.cos(phi)*jnp.sin(theta),jnp.sin(phi)*jnp.sin(theta),jnp.cos(theta)]).T
    A = jnp.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0.5,0.5,0,1,0,0],[0,0.5,0.5,0,1,0],[0.5,0,0.5,0,0,1]])
    Ainv = jnp.linalg.inv(A)


    bcs = BoundaryConditions(geo)
    bcs.periodic('x',lambda batch,space,t:directions[batch,0])
    bcs.periodic('y',lambda batch,space,t:directions[batch,1])
    bcs.periodic('z',lambda batch,space,t:directions[batch,2])


    kappa_bulk = jnp.eye(3)
    kd         = jnp.array([0.2,0.3,0.4,0,0.0,0.0])

    delta = 1e-12

    factor = 1/L

    R                  = L/10
    filtering          = Conic3D(geo,R)

    def transform(x,beta):

        x = filtering(x)

        return projection(x,beta=beta,resolution=N/L)


    @jax.jit
    def objective(rho,beta):

        rho = transform(rho,beta)

        rho = delta + rho*(1-delta)

        kappa_map = lambda batch,space,temp,t: kappa_bulk*rho[space]

        out = Fourier(geo,bcs,kappa_map,batch_size = len(directions),linear_solver='direct',collapse_direct='True')[0]
  
        out['projected_rho'] = rho

        kappa_effective = jnp.matmul(Ainv,out['kappa_effective'])*factor

        g = jnp.linalg.norm(kappa_effective-kd)

        return g,({'kappa':kappa_effective},out)


    betas = [8,16,jnp.inf]
    maxiter = [30,30,100]

    state = State()

    x = jax.random.uniform(jax.random.PRNGKey(1), N**3)


    for k,beta in enumerate(betas):
        print(beta)

        x= MMA(   partial(objective,beta=beta), \
              x0=x,\
              state = state,\
              nDOFs = N**3,\
              maxiter=maxiter[k])


    final_structure = state.aux[-1]['projected_rho'].reshape(grid)

    plot3D(final_structure,write=True)


.. only:: html

   .. raw:: html
      :file: _static/voxels.html



