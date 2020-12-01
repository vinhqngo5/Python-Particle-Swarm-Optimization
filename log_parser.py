from absl import app
from absl import flags
import numpy as np
FLAGS = flags.FLAGS
# flags.DEFINE_string('typetopo', None, 'type of topology')
flags.DEFINE_string('func', None, 'type of function')

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



def main(argv):
    swam_sizes = [128, 256, 512, 1024, 2048]
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
    

if __name__ == '__main__':
  app.run(main)