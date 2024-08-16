import imageio
images = []
filenames = []
for i in range(199):
    filenames.append("s_0_1_t_" + str(i) + ".png")
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie_1.gif', images)
