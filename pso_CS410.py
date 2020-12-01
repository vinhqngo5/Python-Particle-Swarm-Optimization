import numpy as np
import os
from my_pyswarm import pso
import visualize
import create_gif
import option
from absl import app
from absl import flags
import time 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
FLAGS = flags.FLAGS
flags.DEFINE_string('func', 'Rastrigin', 'Test Function')
flags.DEFINE_string('topo', 'star', 'Test Function')
flags.DEFINE_string('gifpath', None, 'Folder_path of gif images')
flags.DEFINE_integer('num_dims', 2, 'Numbers of dimensions')
flags.DEFINE_integer('swarm_size', 100, 'Numbers of swarms')
flags.DEFINE_integer('n_gens', 50, 'Numbers of generations')
flags.DEFINE_integer('times', 1, 'Numbers of times executing the algorithm')
flags.DEFINE_integer('evals', 1000000, 'Numbers of times calling the cost function')
flags.DEFINE_boolean('gif', False, 'If draw gif or not')
flags.DEFINE_boolean('log', False, 'If output log or not')


def main(argv):
    if FLAGS.func == 'Rastrigin':
        print('Rastrigin function')
        myfunc = option.Rastrigin
        contour_func = visualize.Rastrigin_contour
        lb = [-5.12] * FLAGS.num_dims 
        ub = [5.12] * FLAGS.num_dims
    elif FLAGS.func == 'Rosenbrock':
        print('Rosenbrock function')
        myfunc = option.Rosenbrock
        contour_func = visualize.Rosenbrock_contour
        lb = [-1e4] * FLAGS.num_dims
        ub = [1e4] * FLAGS.num_dims
    elif FLAGS.func == 'Eggholder':
        print('Eggholder function')
        contour_func = visualize.Eggholder_contour
        myfunc = option.Eggholder
        lb = [-512, -512]
        ub = [512, 512]
    else:
        print('Ackley function')
        myfunc = option.Ackley
        contour_func = visualize.Ackley_contour
        lb = [-5, -5]
        ub = [5, 5]
    '''
    xopt: optimal x
    xopf: optimal f
    p: best x of per particle
    fp: best f of per particle corresponding to x
    draw_infos: storing infos of global optimum through iterations
    '''
    print('{} function with num_dims = {}, n_gens = {} and swarm_size = {}'.format(FLAGS.func, FLAGS.num_dims, FLAGS.n_gens, FLAGS.swarm_size))
    print('*' * 65)
    if (FLAGS.log):
        log_path = FLAGS.gifpath.replace('gif', 'log').replace('result', 'log') + '.txt'
        f = open(log_path, 'w+')
        f.close()
    
    for i in range(FLAGS.times):
        if (FLAGS.log):
            f = open(log_path, 'a')
            f.writelines('Running the algorithm time: {}\n'.format(i + 1))
        print('Run the function No.{}'.format(i + 1))
        start_time = time.time()
        xopt, fopt, p, fp, draw_p_infos, draw_fp_infos = pso(myfunc, lb, ub, particle_output=True, debug=False, 
                                            swarmsize=FLAGS.swarm_size, maxiter=FLAGS.n_gens, topo=FLAGS.topo, 
                                            evaluations=FLAGS.evals, random_seed=19520354+i)
        end_time = time.time()
        print('x = {}'.format(xopt))
        print('f = {}'.format(fopt))
        print('execution time: {} second'.format(end_time - start_time))
        print('*' * 65)
        if (FLAGS.log):
            f.writelines('optimal solution: {}\n'.format(xopt))
            f.writelines('optimal f: {}\n'.format(fopt))
            f.writelines('optimal found by each particle: \n')
            for i in range(FLAGS.swarm_size):
                f.writelines('\tparticle {}:\n'.format(i))
                f.writelines('\t\toptimal solution of this particle: {}\n'.format(p[i].tolist()))
                f.writelines('\t\toptimal f in this particle: {}\n'.format(fp[i]))
            f.writelines('*'*65 + '\n')
            
        if (FLAGS.gif and FLAGS.num_dims == 2):
            fig = plt.figure()
            ims = []
            for i in range(FLAGS.n_gens):
                a1 ,a2 = zip(*draw_p_infos[i].tolist())
                a3 = draw_fp_infos[i]
                des = os.path.join(FLAGS.gifpath, '{}.png'.format(i))
                num = i
                # print(des)
                plot = visualize.draw_contour(contour_func, lb, ub, a1=a1, a2=a2, a3=a3, des=des, num=num)
            create_gif.gif_init(FLAGS.gifpath)
            print('Creating gif successfully')
 

if __name__ == '__main__':
  app.run(main)