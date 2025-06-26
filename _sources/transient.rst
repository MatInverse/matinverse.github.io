Transient Heat Conduction
=========================

In this example, we report simple transient heat conduction in a 2D domain, where we apply a difference of temperature across the ``x`` axis. Currently, time-stepping is performed using Runge-Kutta method (Tsitouras' 5/4 method).

.. code-block:: python

   import os
   from matinverse.visualizer import save_video
   from matinverse import Geometry2D,BoundaryConditions,Fourier
   from jax import numpy as jnp

   L = 1  #m
   size = [L,L]
   N = 30
   grid = [N,N]


   geo = Geometry2D(grid,size,domain=lambda p:jnp.linalg.norm(p) > L/4) 
   bcs = BoundaryConditions(geo)
   bcs.temperature(lambda p: jnp.isclose(p[0], -size[0]/2), lambda batch, x, t: 1)
   bcs.temperature(lambda p: jnp.isclose(p[0],  size[0]/2), lambda batch, x, t: 0)


   #Minimum timestep for stability: 
   DT_max = (L/N/2)**2

   N_timesteps = 3000

   out = Fourier(geo,bcs,mode='transient',DT=DT_max/10,NT=N_timesteps)[0]

   save_video(out['T'],geo,filename='heat.mp4',fps=100)


.. video:: _static/heat.mp4


Note that if you want a web-ready format, you should do the following conversion:

.. code-block:: bash

    ffmpeg -i video.mp4 -vcodec libx264 -pix_fmt yuv420p -movflags +faststart video_web.mp4


