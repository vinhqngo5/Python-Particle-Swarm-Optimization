import imageio
import os
def gif_init(folder_path):
    folder_path = folder_path
    all_file = os.listdir(folder_path)
    file_paths = []
    for i in range(len(all_file)):
        file_path = os.path.join(folder_path, '{}.png'.format(i))
        file_paths.append(file_path)

    images = []
    for file_path in file_paths:
        images.append(imageio.imread(file_path))
        
    gif_name = os.path.basename(folder_path)
    imageio.mimsave('/content/pyswarm/result/{}.gif'.format(gif_name), images, duration=0.25)


