from absl import app
from absl import flags
import numpy as np
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t

FLAGS = flags.FLAGS
# flags.DEFINE_string('typetopo', None, 'type of topology')
flags.DEFINE_string('func', None, 'type of function')
flags.DEFINE_boolean('t_test', False, 'if using t-test')
flags.DEFINE_string('type', 'star', 'type of topo')

def find_mean(file_path):
    with open(file_path) as f:
        infos = f.readlines()
    good_infos = []
    for line in infos:
        if (line.startswith('optimal f:')):
            good_infos.append(line)
    for i in range(len(good_infos)):
        good_infos[i] = float(good_infos[i].split(':')[1].strip())
    np_array = np.array(good_infos)
    std = np.std(np_array)
    return sum(good_infos) / len(good_infos), std


def independent_ttest(data1, data2, alpha=0.05):
    mean1, mean2 = mean(data1), mean(data2)
    se1, se2 = sem(data1), sem(data2)
    sed = sqrt(se1**2.0 + se2**2.0)
    t_stat = (mean1 - mean2) / sed
    df = len(data1) + len(data2) - 2
    cv = t.ppf(1.0 - alpha, df)
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    return t_stat, df, cv, p



def main(argv):
    swam_sizes = [128, 256, 512, 1024, 2048]
    if (FLAGS.t_test == False):
        star, ring, star_std, ring_std = [], [], [], []
        for i in swam_sizes:
            file_path = "/content/pyswarm/log/" + FLAGS.func + "_10dims_star_" + str(i) + "_log.txt"
            mean, std = find_mean(file_path)
            star.append(mean)
            star_std.append(std)
            file_path = "/content/pyswarm/log/" + FLAGS.func + "_10dims_ring_" + str(i) + "_log.txt"
            mean, std = find_mean(file_path)
            ring.append(mean)
            ring_std.append(std)

        print ("{:<15} {:<20}                               {:<15}".format('Popsize N', 'Star Topology', 'Ring Topology'))
        for i in range(len(star)):
            print ("{:<15} {:4f}({:4f})               {:<4f}({:<4f})".format(swam_sizes[i], star[i], star_std[i], ring[i], ring_std[i]))
    else:
        rastrigin, rosenbrock = [], []
        for i in swam_sizes:
            file_path_1 = "/content/pyswarm/log/Rastrigin_10dims_" + FLAGS.type + "_" + str(i) + "_log.txt"
            file_path_2 = "/content/pyswarm/log/Rosenbrock_10dims_" + FLAGS.type + "_" + str(i) + "_log.txt"
            mean_1, std_1 = find_mean(file_path_1)
            mean_2, std_2 = find_mean(file_path_2)
            rastrigin.append(mean_1)
            rosenbrock.append(mean_2)
        t_stat, df, cv, p = independent_ttest(rastrigin, rosenbrock)
        print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
        # interpret via critical value
        if abs(t_stat) <= cv:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
        

        # interpret via p-value
        if p > 0.05:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')

        

if __name__ == '__main__':
  app.run(main)