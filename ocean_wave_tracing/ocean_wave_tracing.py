import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xa
import pyproj # type: ignore
import sys
import cmocean.cm as cm
from netCDF4 import Dataset
import json
from importlib.resources import files, as_file
from tqdm import tqdm 

from .util_solvers import Advection, WaveNumberEvolution, RungeKutta4
from .util_methods import make_xarray_dataArray, to_xarray_ds, check_velocity_field, check_bathymetry


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ocean_wave_tracing.log', level=logging.INFO)
logging.info('\nStarted')


class Wave_tracing():
    """ Class computing the path of ocean wave rays according to the geometrical
    optics approximation.
    """
    def __init__(self, U, V,  nx, ny, nt, T, dx, dy,
                 nb_wave_rays, domain_X0, domain_XN, domain_Y0, domain_YN,
                 temporal_evolution=False,
                 d=None,DEBUG=False,**kwargs):
        """
        Args:
            U (float): eastward velocity 2D field
            V (float): northward velocity 2D field
            nx (int): number of points in x-direction of velocity domain
            ny (int): number of points in y-direction of velocity domain
            nt (int): number of time steps for computation
            T (int): Seconds. Duration of wave tracing
            dx (int): Spatial resolution in x-direction. Units conforming to U
            dy (int): Spatial resolution in y-direction. Units conforming to V
            nb_wave_rays (int): Number of wave rays to track.
            domain_*0 (float): start value of domain area in X and Y direction
            domain_*N (float): end value of domain area in X and Y direction
            temporal_evolution (bool): flag if velocity field should change in time
            d (float): 2D bathymetry field
            **kwargs
        """
        self.g = 9.81
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.nb_wave_rays = nb_wave_rays
        self.C = -1
        assert nb_wave_rays > 0, "Number of wave rays must be larger than zero"

        self.domain_X0 = domain_X0 # left side
        self.domain_XN = domain_XN # right side
        self.domain_Y0 = domain_Y0 # bottom
        self.domain_YN = domain_YN # top
        self.T = T
        self.min_depth = 0
        
        self.temporal_evolution = temporal_evolution
        self.debug = DEBUG

        # Setting up X and Y domain
        assert(domain_X0 < domain_XN) # condition for fast search
        assert(domain_Y0 < domain_YN) # condition for fast search
        self.x = np.linspace(domain_X0, domain_XN, nx)
        self.y = np.linspace(domain_Y0, domain_YN, ny)

        # Check the bathymetry
        if d is not None:
            self.d = check_bathymetry(d=d,x=self.x,y=self.y)
        else:
            d_static = 1e5
            logging.info(f'Hardcoding bathymetry to {d_static}m since not given.')
            self.d = check_bathymetry(d=np.ones((ny,nx))*d_static,x=self.x,y=self.y)

        # Computing the horizontal gradients of the bathymetry
        self.dddx = self.d.differentiate(coord='x',edge_order=2)
        self.dddy = self.d.differentiate(coord='y',edge_order=2)

        # Cache numpy arrays for fast numerical operations
        self.d_numpy = self.d.values
        self.dddx_numpy = self.dddx.values
        self.dddy_numpy = self.dddy.values

        # Setting up the wave rays
        self.ray_x = np.zeros((nb_wave_rays,nt))
        self.ray_y = np.zeros((nb_wave_rays,nt))
        self.ray_kx = np.zeros((nb_wave_rays,nt))
        self.ray_ky = np.zeros((nb_wave_rays,nt))
        self.ray_k = np.zeros((nb_wave_rays,nt))
        self.ray_theta = np.ma.zeros((nb_wave_rays,nt))
        self.ray_cg = np.ma.zeros((nb_wave_rays,nt)) # intrinsic group velocity
        self.ray_U = np.ma.zeros((nb_wave_rays,nt)) # U component closest to ray
        self.ray_V = np.ma.zeros((nb_wave_rays,nt)) # V component closest to ray
        self.ray_dudx = np.ma.zeros((nb_wave_rays,nt)) # derivatives of the ambient current components
        self.ray_dvdy = np.ma.zeros((nb_wave_rays,nt)) 
        self.ray_dudy = np.ma.zeros((nb_wave_rays,nt)) 
        self.ray_dvdx = np.ma.zeros((nb_wave_rays,nt)) 
        self.ray_depth = np.zeros((nb_wave_rays,nt))

        # along-ray bathymetry gradient
        self.dsigma_dx = np.ma.zeros((nb_wave_rays,nt))
        self.dsigma_dy = np.ma.zeros((nb_wave_rays,nt))
        
        # along-ray phase velocity gradient
        self.d_cy = np.ma.zeros((nb_wave_rays,nt))
        self.d_cx = np.ma.zeros((nb_wave_rays,nt))
        
        # make xarray DataArray of velocity field
        self.U = check_velocity_field(U,temporal_evolution,x=self.x,y=self.y)
        self.V = check_velocity_field(V,temporal_evolution,x=self.x,y=self.y)

        # Cache numpy arrays for fast indexing
        self.U_numpy = self.U.values
        self.V_numpy = self.V.values

        # Time
        self.dt = T/nt
        self.nb_velocity_time_steps = len(self.U.time)
        if not temporal_evolution:
            self.velocity_idt = np.zeros(nt,dtype=int)
            #logging.info('Vel idt : {}'.format(self.velocity_idt))
        else:
            t_velocity_field = U.time.data
            self.T0 = t_velocity_field[0]
            t_wr = np.arange(self.T0, self.T0+np.timedelta64(T,'s'),np.timedelta64(int(((T/nt)*1e3)),'ms'))
            self.velocity_idt = self.find_nearest_fast(t_velocity_field, t_wr)

        self.kwargs = kwargs


    def check_CFL(self, cg, max_speed):
        """ Method for checking the Courant, Friedrichs, and Lewy
            condition for numerical intergration
        """
        dt = self.dt
        DX = np.abs(np.min([self.dx,self.dy])) 

        assert cg>=0, "Group velocity must be positive. Currently {}".format(cg)
        assert max_speed>=0,  "Maximum current speed must be positive. Currently {}".format(max_speed)

        u = cg+max_speed

        C = u*(dt/DX)

        if C<=1:
            logger.info('Courant number is {}'.format(np.round(C,2)))
        else:
            logger.warning('Courant number is {}'.format(np.round(C,2)))

        self.C = C

    def c_intrinsic(self,k,d,group_velocity=False):
        """ Computing the intrinsic wave phase and group velocity according
        to the general dispersion relation

        Args:
            k (float): wave number (numpy array)
            d (float): depth (numpy array)
            group_velocity (bool): returns group velocity (True) or phase
                velocity (False)

        Returns:
            instrinsic velocity (float): group or phase velocity depending on
                flag
        """
        g = self.g
        
        # Handle both scalar and array inputs
        k = np.atleast_1d(k)
        d = np.atleast_1d(d)
        kd = k * d
        dw_criteria = kd > 25
        
        if np.all(dw_criteria):
            # Deep water approximation
            c_in = np.sqrt(g / k)
            n = np.full_like(c_in, 0.5)
        else:
            # General case
            kd = k * d
            tanh_kd = np.tanh(kd)
            c_in = np.sqrt((g / k) * tanh_kd)
            sinh_2kd = np.sinh(2 * kd)
            n = 0.5 * (1 + (2 * kd) / sinh_2kd)
        
        if group_velocity:
            return c_in * n
        else:
            return c_in

    def sigma(self,k,d):
        """ Intrinsic frequency dispersion relation

        Args:
            k (float): Wave number
            d (float): depth

        Returns:
            sigma (float): intrinsic frequency
        """

        g=self.g
        sigma = np.sqrt(g*k*np.tanh(k*d))
        return sigma

    def dsigma_x(self,k,idxs,idys,ray_depths):
        """ Compute the gradient of sigma in the x-direction due to
        the bathymetry.
        """
        kd = k * ray_depths
        tanh_kd = np.tanh(kd)
        nabla_d_rays = self.dddx_numpy[idys, idxs]
        dsigma = 0.5 * k * np.sqrt((self.g * k) / tanh_kd) * (1 - tanh_kd**2) * nabla_d_rays
        return dsigma

    def dsigma_y(self,k,idxs,idys,ray_depths):
        """ Compute the gradient of sigma in the y-direction due to
        the bathymetry.
        """
        kd = k * ray_depths
        tanh_kd = np.tanh(kd)
        nabla_d_rays = self.dddy_numpy[idys, idxs]
        dsigma = 0.5 * k * np.sqrt((self.g * k) / tanh_kd) * (1 - tanh_kd**2) * nabla_d_rays
        return dsigma

    def grad_c_x(self,k,idxs,idys,ray_depths):
        """ Compute the phase speed gradient in x-direction 
        """
        kd = k * ray_depths
        tanh_kd = np.tanh(kd)
        nabla_d_rays = self.dddx_numpy[idys, idxs]
        nabla_c = 0.5 * np.sqrt((self.g * k) / tanh_kd) * (1 - tanh_kd**2) * nabla_d_rays
        return nabla_c
        
 
    def grad_c_y(self,k,idxs,idys,ray_depths):
        """ Compute the phase speed gradient in y-direction 
        """
        kd = k * ray_depths
        tanh_kd = np.tanh(kd)
        nabla_d_rays = self.dddy_numpy[idys, idxs]
        nabla_c = 0.5 * np.sqrt((self.g * k) / tanh_kd) * (1 - tanh_kd**2) * nabla_d_rays
        return nabla_c


    def wave(self,T,theta,d):
        """ Method computing wave number from initial wave period.
        Solving implicitly for the wave number k, with initial guess from the approximate
        wave number according to Eckart (1952)

        Args:
            T (float): Wave period
            theta (float): radians. Wave direction

        Returns:
            k0 (float): wave number
            ray_kx0 (float): wave number in x-direction
            ray_ky0 (float): wave number in y-direction
        """
        g = self.g
        sigma = (2 * np.pi) / T
        k0 = (sigma**2) / g
        
        # Initialize result
        k = np.zeros_like(d, dtype=np.float64)
        
        # DEEP WATER (kd > 3): Analytical solution
        k_deep = k0
        deep_mask = k_deep * d > 3.0
        k[deep_mask] = k_deep
        
        # SHALLOW WATER (kd < 0.5): Analytical solution
        k_shallow = sigma / np.sqrt(g * d)
        shallow_mask = k_shallow * d < 0.5
        k[shallow_mask] = k_shallow[shallow_mask]
        
        # INTERMEDIATE DEPTHS: Newton-Raphson
        intermediate_mask = ~(deep_mask | shallow_mask)
        
        if np.any(intermediate_mask):
            d_int = d[intermediate_mask]
            
            # Initial guess using Hunt's approximation (1979)
            y = sigma**2 * d_int / g
            k_int = k0 * np.sqrt(y + 1/(1 + 0.6522*y + 0.4622*y**2 + 0.0864*y**3 + 0.0675*y**4))
            
            # Newton-Raphson iterations
            for iteration in range(8):
                kd = k_int * d_int
                tanh_kd = np.tanh(kd)
                sech2_kd = 1 - tanh_kd**2
                
                sqrt_term = np.sqrt(g * k_int * tanh_kd)
                f = sqrt_term - sigma
                f_prime = (g * tanh_kd + g * k_int * d_int * sech2_kd) / (2 * sqrt_term)
                
                k_new = k_int - f / f_prime
                
                if np.max(np.abs(k_new - k_int)) < 1e-10:
                    k_int = k_new
                    break
                
                k_int = k_new
            
            k[intermediate_mask] = k_int
        
        # Directional components
        kx = k * np.cos(theta)
        ky = k * np.sin(theta)
        
        return k, kx, ky


    def set_initial_condition(self, wave_period, theta0,**kwargs):
        """ Setting inital conditions for the domain. Support domain side and
            initial x- and y-positions. However, side trumps initial position
            if both are given.

            args:
            wave_period (float): Wave period.
            theta0 (rad, float): Wave initial direction. In radians.
                         (0,.5*pi,pi,1.5*pi) correspond to going
                         (right, up, left, down).

            **kwargs
                incoming_wave_side (str): side for incoming wave direction
                                            [left, right, top, bottom]
                ipx (float, array of floats): initial position x
                ipy (float, array of floats): initial position y
                minimum_depth (float): depth for early stop
        """

        nb_wave_rays = self.nb_wave_rays

        valid_sides = ['left', 'right','top','bottom']

        if 'incoming_wave_side' in kwargs:
            i_w_side = kwargs['incoming_wave_side']

            if not i_w_side in valid_sides:
                logger.info('Invalid initial side. Left will be used.')
                i_w_side = 'left'

            if i_w_side == 'left':
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'right':
                xs = np.ones(nb_wave_rays)*self.domain_XN
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            elif i_w_side == 'top':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_YN

            elif i_w_side == 'bottom':
                xs = np.linspace(self.domain_X0, self.domain_XN, nb_wave_rays)
                ys = np.ones(nb_wave_rays)*self.domain_Y0

        else:
            logger.info('No initial side given. Try with discrete points')

            ipx = kwargs.get('ipx', None)
            ipy = kwargs.get('ipy', None)

            if ipx is None and ipy is None:
                logger.info('No initial position points given. Left will be used')
                xs = np.ones(nb_wave_rays)*self.domain_X0
                ys = np.linspace(self.domain_Y0, self.domain_YN, nb_wave_rays)

            else:
                # First check initial position x
                if ipx is not None and np.isfinite(ipx).all():
                    #Check if it is an array
                    if isinstance(ipx,np.ndarray):
                        assert nb_wave_rays == len(ipx), "Need same dimension on initial x-values"
                        xs=ipx.copy()
                    # if not, use it as single value
                    else:
                        xs = np.ones(nb_wave_rays)*ipx

                if ipy is not None and np.isfinite(ipy).all():
                    if isinstance(ipy,np.ndarray):
                        assert nb_wave_rays == len(ipy), "Need same dimension on initial y-values"
                        ys=ipy.copy()
                    else:
                        ys = np.ones(nb_wave_rays)*ipy


        # Set initial position
        self.ray_x[:,0] = xs
        self.ray_y[:,0] = ys

        #Theta0
        if type(theta0) is float or type(theta0) is int:
            theta0 = np.ones(nb_wave_rays)*theta0
        elif isinstance(theta0,np.ndarray):
            assert nb_wave_rays == len(theta0), "Initial values must have same dimension as number of wave rays"
        else:
            logger.error('Theta0 must be either float or numpy array. Terminating.')
            sys.exit()

        # Get depths using direct numpy indexing
        idxs = self.find_nearest_fast(self.x, xs)
        idys = self.find_nearest_fast(self.y, ys)
        depths = self.d_numpy[idys, idxs]

        # Vectorized wave computation
        k_all, kx_all, ky_all = self.wave(T=wave_period, theta=theta0, d=depths)
        
        self.ray_k[:, 0] = k_all
        self.ray_kx[:, 0] = kx_all
        self.ray_ky[:, 0] = ky_all

        # Vectorized group velocity computation
        self.ray_cg[:, 0] = self.c_intrinsic(k=k_all, d=depths, group_velocity=True)

        # set inital wave propagation direction
        self.ray_theta[:,0] = theta0

        #Check the CFL condition
        max_vel = np.sqrt(np.nanmax(self.U_numpy**2 + self.V_numpy**2))
        self.check_CFL(cg=np.nanmax(self.ray_cg[:, 0]), max_speed=max_vel)
        
        if 'min_depth' in kwargs:
            self.min_depth = kwargs['min_depth']


    def find_nearest_fast(self, array, values):
        """Fast nearest neighbor search for sorted arrays."""
        assert(np.all(np.diff(array) >=0))
        indices = np.searchsorted(array, values, side='left')
        indices = np.clip(indices, 1, len(array) - 1)
        
        # Check if left or right neighbor is closer
        left = indices - 1
        right = indices
        
        indices = np.where(
            np.abs(values - array[left]) < np.abs(values - array[right]),
            left,
            right
        )
        
        return indices    
    
    def solve(self, solver=RungeKutta4, early_stop=False):
        """ Solve the geometrical optics equations numerically by means of the
            method of characteristics. Optionally stops individual rays early 
            if they hit land or exit the domain.
        """

        if not callable(solver):
            raise TypeError('f is %s, not a solver' % type(solver))

        ray_k = self.ray_k
        ray_kx = self.ray_kx
        ray_ky = self.ray_ky
        ray_x = self.ray_x
        ray_y = self.ray_y
        ray_theta = self.ray_theta
        ray_cg = self.ray_cg

        # Use cached numpy arrays
        U = self.U_numpy
        V = self.V_numpy

        # Compute velocity gradients once and extract to numpy
        dudx = self.U.differentiate('x').values
        dudy = self.U.differentiate('y').values
        dvdx = self.V.differentiate('x').values
        dvdy = self.V.differentiate('y').values

        x = self.x
        y = self.y
        dt = self.dt
        nt = self.nt
        velocity_idt = self.velocity_idt

        # Track which rays are still active (only if early_stop is enabled)
        if early_stop:
            ray_active = np.ones(self.nb_wave_rays, dtype=bool)

        counter = 0
        t = np.linspace(0, self.T, nt)

        for n in tqdm(range(0, nt - 1), desc='Solving for {self.nb_wave_rays}'):
            
            # Get active rays if early stopping is enabled
            if early_stop:
                active_indices = np.where(ray_active)[0]
                if len(active_indices) == 0:
                    logger.info(f'All rays terminated by timestep {n}')
                    break
                idxs = self.find_nearest_fast(x, ray_x[active_indices, n])
                idys = self.find_nearest_fast(y, ray_y[active_indices, n])
                ray_depth = self.d_numpy[idys, idxs]
            else:
                active_indices = slice(None)  # All rays
                idxs = self.find_nearest_fast(x, ray_x[:, n])
                idys = self.find_nearest_fast(y, ray_y[:, n])
                ray_depth = self.d_numpy[idys, idxs]

            self.ray_depth[active_indices, n] = ray_depth

            # Direct numpy indexing
            vid = velocity_idt[n]
            self.ray_U[active_indices, n] = U[vid, idys, idxs]
            self.ray_V[active_indices, n] = V[vid, idys, idxs]

            self.ray_dudx[active_indices, n] = dudx[vid, idys, idxs]
            self.ray_dvdy[active_indices, n] = dvdy[vid, idys, idxs]
            self.ray_dudy[active_indices, n] = dudy[vid, idys, idxs]
            self.ray_dvdx[active_indices, n] = dvdx[vid, idys, idxs]

            k_active = ray_k[active_indices, n]
            self.d_cx[active_indices, n] = self.grad_c_x(k_active, idxs, idys, ray_depth)
            self.d_cy[active_indices, n] = self.grad_c_y(k_active, idxs, idys, ray_depth)

            #======================================================
            ### numerical integration of the wave ray equations ###
            #======================================================

            # Compute group velocity
            ray_cg[active_indices, n] = self.c_intrinsic(k_active, d=ray_depth, group_velocity=True)

            # ADVECTION
            f_adv = Advection(cg=ray_cg[active_indices, n], k=k_active, 
                            kx=ray_kx[active_indices, n], U=U[vid, idys, idxs])
            ray_x[active_indices, n + 1] = solver.advance(u=ray_x[active_indices, n], f=f_adv, k=n, t=t)

            f_adv = Advection(cg=ray_cg[active_indices, n], k=k_active, 
                            kx=ray_ky[active_indices, n], U=V[vid, idys, idxs])
            ray_y[active_indices, n + 1] = solver.advance(u=ray_y[active_indices, n], f=f_adv, k=n, t=t)

            # EVOLUTION IN WAVE NUMBER
            self.dsigma_dx[active_indices, n] = self.dsigma_x(k_active, idxs, idys, ray_depth)
            self.dsigma_dy[active_indices, n] = self.dsigma_y(k_active, idxs, idys, ray_depth)

            f_wave_nb = WaveNumberEvolution(d_sigma=self.dsigma_dx[active_indices, n], 
                                        kx=ray_kx[active_indices, n], ky=ray_ky[active_indices, n],
                                        dUkx=self.ray_dudx[active_indices, n], 
                                        dUky=self.ray_dvdx[active_indices, n])
            ray_kx[active_indices, n + 1] = solver.advance(u=ray_kx[active_indices, n], f=f_wave_nb, k=n, t=t)

            f_wave_nb = WaveNumberEvolution(d_sigma=self.dsigma_dy[active_indices, n], 
                                        kx=ray_kx[active_indices, n], ky=ray_ky[active_indices, n],
                                        dUkx=self.ray_dudy[active_indices, n], 
                                        dUky=self.ray_dvdy[active_indices, n])
            ray_ky[active_indices, n + 1] = solver.advance(u=ray_ky[active_indices, n], f=f_wave_nb, k=n, t=t)

            # Compute wave number k
            ray_k[active_indices, n + 1] = np.sqrt(ray_kx[active_indices, n + 1]**2 + 
                                                ray_ky[active_indices, n + 1]**2)

            # THETA
            ray_theta[active_indices, n + 1] = np.arctan2(ray_ky[active_indices, n + 1], 
                                                        ray_kx[active_indices, n + 1])

            # Keep angles between 0 and 2pi
            ray_theta[active_indices, n + 1] = np.mod(ray_theta[active_indices, n + 1], (2 * np.pi))

            # Early stopping check
            if early_stop:
                # Check domain boundaries
                out_of_bounds = (
                    (ray_x[active_indices, n + 1] < self.domain_X0) |
                    (ray_x[active_indices, n + 1] > self.domain_XN) |
                    (ray_y[active_indices, n + 1] < self.domain_Y0) |
                    (ray_y[active_indices, n + 1] > self.domain_YN)
                )
                
                # Check for hitting land (depth <= 0)
                next_idxs = self.find_nearest_fast(x, ray_x[active_indices, n + 1])
                next_idys = self.find_nearest_fast(y, ray_y[active_indices, n + 1])
                next_depths = self.d_numpy[next_idys, next_idxs]
                hit_land = next_depths <= self.min_depth
                
                # Deactivate rays
                rays_to_deactivate = out_of_bounds | hit_land
                
                if np.any(rays_to_deactivate):
                    deactivated_ray_ids = active_indices[rays_to_deactivate]
                    ray_active[deactivated_ray_ids] = False
                    logger.info(f'Timestep {n}: Deactivated {np.sum(rays_to_deactivate)} rays. '
                            f'{np.sum(ray_active)} rays still active.')
                    
                    # Set remaining values to NaN for deactivated rays
                    ray_x[deactivated_ray_ids, n + 1:] = np.nan
                    ray_y[deactivated_ray_ids, n + 1:] = np.nan
                    ray_k[deactivated_ray_ids, n + 1:] = np.nan
                    ray_kx[deactivated_ray_ids, n + 1:] = np.nan
                    ray_ky[deactivated_ray_ids, n + 1:] = np.nan
                    ray_theta[deactivated_ray_ids, n + 1:] = np.nan

            counter += 1

        ###
        # Fill last values
        ###
        if early_stop:
            active_indices = np.where(ray_active)[0]
            if len(active_indices) == 0:
                active_indices = np.array([], dtype=int)
        else:
            active_indices = slice(None)

        if early_stop and len(active_indices) > 0 or not early_stop:
            idxs = self.find_nearest_fast(x, ray_x[active_indices, n])
            idys = self.find_nearest_fast(y, ray_y[active_indices, n])
            
            self.ray_depth[active_indices, n + 1] = self.d_numpy[idys, idxs]

            self.ray_U[active_indices, n + 1] = U[velocity_idt[n + 1], idys, idxs]
            self.ray_V[active_indices, n + 1] = V[velocity_idt[n + 1], idys, idxs]

            self.ray_dudx[active_indices, n + 1] = dudx[velocity_idt[n + 1], idys, idxs]
            self.ray_dvdy[active_indices, n + 1] = dvdy[velocity_idt[n + 1], idys, idxs]
            self.ray_dudy[active_indices, n + 1] = dudy[velocity_idt[n + 1], idys, idxs]
            self.ray_dvdx[active_indices, n + 1] = dvdx[velocity_idt[n + 1], idys, idxs]

            k_final = ray_k[active_indices, n + 1]
            depth_final = self.ray_depth[active_indices, n + 1]
            self.dsigma_dx[active_indices, n + 1] = self.dsigma_x(k_final, idxs, idys, depth_final)
            self.dsigma_dy[active_indices, n + 1] = self.dsigma_y(k_final, idxs, idys, depth_final)

            self.d_cx[active_indices, n + 1] = self.grad_c_x(k_final, idxs, idys, depth_final)
            self.d_cy[active_indices, n + 1] = self.grad_c_y(k_final, idxs, idys, depth_final)

            ray_cg[active_indices, n + 1] = self.c_intrinsic(k_final, d=depth_final, group_velocity=True)

        self.dudy = dudy
        self.dudx = dudx
        self.dvdy = dvdy
        self.dvdx = dvdx
        self.ray_k = ray_k
        self.ray_kx = ray_kx
        self.ray_ky = ray_ky
        self.ray_x = ray_x
        self.ray_y = ray_y
        self.ray_theta = ray_theta
        self.ray_cg = ray_cg
        
        if early_stop:
            self.ray_active = ray_active
            logging.info(f'Final active rays: {np.sum(ray_active)}/{self.nb_wave_rays}')
        
        logging.info(f'Stopped at time idt: {velocity_idt[n]}')

    def to_ds(self,**kwargs):
        """Convert wave ray information to xarray object"""

        if 'proj4' in kwargs:
            lons,lats = self.to_latlon(kwargs['proj4'])
        else:
            lons = np.zeros((self.nb_wave_rays,self.nt))
            lats = lons.copy()

        variables = {'ray_k':self.ray_k,
                    'ray_kx':self.ray_kx,
                    'ray_ky':self.ray_ky,
                    'ray_x':self.ray_x,
                    'ray_y':self.ray_y,
                    'ray_U':self.ray_U,
                    'ray_V':self.ray_V,
                    'ray_theta':self.ray_theta,
                    'ray_cg':self.ray_cg,
                    'ray_depth':self.ray_depth,
                    'ray_lat': lats,
                    'ray_lon':lons,
                    'ray_dudx':self.ray_dudx,
                    'ray_dvdy':self.ray_dvdy,
                    'ray_dudy':self.ray_dudy,
                    'ray_dvdx':self.ray_dvdx
                    }

        source = files('ocean_wave_tracing').joinpath('ray_metadata.json')
        with as_file(source) as sfile:
            logging.info('Loading JSON file')
            data = json.loads(sfile.read_text())
            logging.info('Finished loading JSON file')

        # relative time
        t = np.linspace(0,self.T,self.nt)
        ray_id = np.arange(self.nb_wave_rays)

        vars = [make_xarray_dataArray(var=variables[vname], t=t,rays=ray_id,name=vname,attribs=data[vname]) for vname in list(variables.keys())]

        return to_xarray_ds(vars)

    def ray_density(self,x_increment, y_increment, plot=False, proj4=None):
        """ Method computing ray density within boxes. The density of wave rays
        can be used as proxy for wave energy density

        Args:
            x_increment (int): size of box in x direction. Length = x_increment*dx
            y_increment (int): size of box in y direction. Length = y_increment*dy

        Returns:
            xx (2d): x grid
            yy (2d): y grid
            hm (2d): heat map of wave ray density

        #>>> wt = Wave_tracing(U=1,V=1,nx=1,ny=1,nt=1,T=1,dx=1,dy=1,wave_period=1, theta0=1,nb_wave_rays=1,domain_X0=0,domain_XN=0,domain_Y0=0,domain_YN=1,incoming_wave_side='left')
        #>>> wt.solve()
        #>>> wt.ray_density(x_increment=20,y_increment=20)
        """
        # Create grid
        xs = self.x[::x_increment]
        ys = self.y[::y_increment]
        xx, yy = np.meshgrid(xs, ys)
        
        xs = xx[0]
        ys = yy[:, 0]
        
        n_y_bins = len(ys) - 1
        n_x_bins = len(xs) - 1
        hm = np.zeros((n_y_bins, n_x_bins))

        
        # Flatten ray coordinates
        ray_x_flat = self.ray_x.ravel()
        ray_y_flat = self.ray_y.ravel()
        
        # Digitize returns bin indices (1-indexed, 0 for out-of-bounds).
        x_indices = np.digitize(ray_x_flat, xs) - 1
        y_indices = np.digitize(ray_y_flat, ys) - 1
        
        # Filter valid indices (within bounds)
        valid_mask = (
            (x_indices >= 0) & (x_indices < n_x_bins) &
            (y_indices >= 0) & (y_indices < n_y_bins)
        )
        
        x_indices_valid = x_indices[valid_mask]
        y_indices_valid = y_indices[valid_mask]
        
        # Count occurrences in each bin using bincount with 2D indexing
        # Convert 2D indices to 1D
        flat_indices = y_indices_valid * n_x_bins + x_indices_valid
        counts = np.bincount(flat_indices, minlength=n_y_bins * n_x_bins)
        
        # Reshape back to 2D
        hm = counts.reshape(n_y_bins, n_x_bins)
        
        if plot:
            plt.figure(figsize=(10, 8))
            
            plt.pcolormesh(xs, ys, hm, cmap='viridis', shading='flat')
            plt.colorbar(label='Ray Density')
            
            # Plot individual rays
            for i in range(self.nb_wave_rays):
                plt.plot(self.ray_x[i, :], self.ray_y[i, :], '-r', alpha=0.3, linewidth=0.5)
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Wave Ray Density (Vectorized)')
            plt.xlim(xs[0], xs[-1])
            plt.ylim(ys[0], ys[-1])
            plt.show()
        
        if proj4 is not None:
            print(proj4)
            transformer = pyproj.Transformer.from_proj(proj4, 'epsg:4326', always_xy=True)
            lons, lats = transformer.transform(xx.flatten(), yy.flatten())
            lons, lats = lons.reshape(xx.shape), lats.reshape(yy.shape)
            return lons, lats, hm
        
        return xx, yy, hm
    
    def to_latlon(self, proj4):
        """ Method for reprojecting wave rays to latitude/longitude values

        Args:
            proj4 (str): proj4 string

        Returns:
            lons (2d): wave rays in longitude
            lats (2d): wave rays in latitude
        """
        transformer = pyproj.Transformer.from_proj(proj4, 'epsg:4326', always_xy=True)
        lons, lats = transformer.transform(self.ray_x.flatten(), self.ray_y.flatten())
        return lons.reshape(self.ray_x.shape), lats.reshape(self.ray_y.shape)
    
    def get_ray_curvature(self,decomposed=False):
        """ Compute the approximate analytical ray curvature after Halsne and Li (2025, in rev.)
        """

        # Tangent and normal vector
        ds_ray = self.to_ds()
        eo=2
        xprime=ds_ray.ray_x.differentiate(coord='time',edge_order=eo)
        yprime=ds_ray.ray_y.differentiate(coord='time',edge_order=eo)

        arclength=np.sqrt(xprime**2+yprime**2)
        nx, ny = -yprime/arclength, xprime/arclength

        # intrinsic phase velocity
        non_normalized_ray_curvature_depth = -((nx*self.d_cx) + (ny*self.d_cy)) # NOTE: The negative sign is taken into account here
        
        vorticity = ds_ray.ray_dvdx-ds_ray.ray_dudy #ray curvature currents

        group_velocity = ds_ray.ray_cg 

        # Compute the curvature
        ray_curvature_curr = vorticity/group_velocity
        ray_curvature_depth = non_normalized_ray_curvature_depth/group_velocity 
        ray_curvature_tot = ray_curvature_curr + ray_curvature_depth 

        if decomposed:
            return ray_curvature_tot, ray_curvature_depth, ray_curvature_curr
        else:
            return ray_curvature_tot

    def get_shoaling_coefficient(self):
        """ Compute the shoaling coefficient due to group velocity changes
        """
        ds_ray = self.to_ds()
        sc=np.sqrt(ds_ray.ray_cg[:,0]/ds_ray.ray_cg)
        sc.attrs['units']='-'
        sc.attrs['long_name']='Shoaling coefficient'
        return sc