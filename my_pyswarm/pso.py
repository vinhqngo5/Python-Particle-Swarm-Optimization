from functools import partial
import numpy as np

def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)

def _is_feasible_wrapper(func, x):
    return np.all(func(x)>=0)

def _cons_none_wrapper(x):
    return np.array([0])

def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])

def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))
    
def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
        swarmsize=100, omega=0.72985, phip=1.49618, phig=1.49618, maxiter=1e6, 
        minstep=0, minfunc=0, debug=True, processes=1,
        particle_output=False, topo='star', random_seed=0, evaluations=1e6):
    """
    Thực hiện thuật toán particle swarm optimization (PSO)
   
    Parameters (Đây là các parameter bắt buộc)
    ==========
    func : function
        Đây là function cần được tối ưu, sẽ được truyền vô = con trỏ hàm qua option.py
    lb : array
        ~lower bound: là chặn dưới của các biến đầu vào
    ub : array
        ~upper bound: là chặn trên của các biến đầu vào

    Các trọng số quán tính và trọng số gia tốc được tune sẵn khi truyền vào
   
    Optional
    ========
    swarmsize : int
        Số lượng khởi tạo của quần thể
    maxiter : int
        Số lượng thế hệ tối đa cho đến khi thuật toán dừng lại
    debug : boolean
        Có in ra quá trình chạy qua từng thế hệ hay không?
    Returns
    =======
    g : array
        Trả về mảng lưu trữ tọa độ của những kết quả tối ưu
    f : scalar
        Trả về kết quả tối ưu của hàm f ứng với g
    p : array
        Trả về tọa độ tốt nhất của từng particles
    pf: arrray
        Trả về kết quả tốt nhất của từng particles
   
    """
    # Cảnh báo khi không thỏa điều kiện
    assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Initialize objective function
    obj = partial(_obj_wrapper, func, args, kwargs)
    
    # Check for constraint function(s) #########################################
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = _cons_none_wrapper
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
    is_feasible = partial(_is_feasible_wrapper, cons)

    # Initialize the multiprocessing module if necessary
    if processes > 1:
        import multiprocessing
        mp_pool = multiprocessing.Pool(processes)
        
    # Initialize the particle swarm ############################################
    np.random.seed(random_seed)
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fx = np.zeros(S)  # current particle function values
    fs = np.zeros(S, dtype=bool)  # feasibility of each particle
    fp = np.ones(S)*np.inf  # best particle function values
    g = []  # best swarm position
    fg = np.inf  # best swarm position starting value
    count_evaluations = 0 # calculate times of calling function
    # Initialize the particle's position
    x = lb + x*(ub - lb)

    # Calculate objective and constraints for each particle
    if processes > 1:
        count_evaluations += S
        fx = np.array(mp_pool.map(obj, x))
        fs = np.array(mp_pool.map(is_feasible, x))
    else:
        for i in range(S):
            count_evaluations += 1
            fx[i] = obj(x[i, :])
            fs[i] = is_feasible(x[i, :])
       
    # Store particle's best position (if constraints are satisfied)
    i_update = np.logical_and((fx < fp), fs)
    p[i_update, :] = x[i_update, :].copy()
    fp[i_update] = fx[i_update]

    # Update swarm's best position
    i_min = np.argmin(fp)
    if fp[i_min] < fg:
        fg = fp[i_min]
        g = p[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        g = x[0, :].copy()
       
    # Initialize the particle's velocity
    v = vlow + np.random.rand(S, D)*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    draw_p_infos = []
    draw_fp_infos = []
    while (it <= maxiter and count_evaluations <= evaluations):
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))

        # Update the particles velocities
        if (topo == 'star'):
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
        else:
            for i in range(S):
                idx = i
                if (fp[(i + 1) % S] < fp[idx]):
                    idx = (i + 1) % S
                if (fp[(i - 1) % S] < fp[idx]):
                    idx = (i - 1) % S
                near_g = p[idx]
                v[i] = omega * v[i] + phip*rp[i]*(p[i] - x[i]) + phig*rg[i]*(near_g - x[i])
        # Update the particles' positions
        x = x + v
        # Correct for bound violations
        maskl = x < lb
        masku = x > ub
        x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku

        # Update objectives and constraints
        if processes > 1:
            count_evaluations += S
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                count_evaluations += 1
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
        # Get the draw_infos of per particles per iterations
        draw_p_infos.append(x)
        draw_fp_infos.append(fx)
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        # print('i_update = {}'.format(i_update))
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            if debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(it, p[i_min, :], fp[i_min]))

            p_min = p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((g - p_min)**2))

            if np.abs(fg - fp[i_min]) < minfunc:
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(minfunc))
                if particle_output:
                    return p_min, fp[i_min], p, fp, draw_p_infos, draw_fp_infos
                else:
                    return p_min, fp[i_min], draw_p_infos, draw_fp_infos
            elif stepsize < minstep:
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(minstep))
                if particle_output:
                    return p_min, fp[i_min], p, fp, draw_p_infos, draw_fp_infos
                else:
                    return p_min, fp[i_min], draw_p_infos, draw_fp_infos
            else:
                g = p_min.copy()
                fg = fp[i_min]

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1
    
    if (it > maxiter):
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    else:
        print('Stopping search: maximum Evaluations reached --> {:}'.format(evaluations))
    
    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    if particle_output:
        return g, fg, p, fp, draw_p_infos, draw_fp_infos
    else:
        return g, fg, draw_p_infos, draw_fp_infos
