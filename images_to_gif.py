import imageio
images = []
filenames = []
for i in range(200):
    filenames.append("Plots/Diff_cent_" + str(i) + ".png")
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('Plots/Diff_movie.gif', images)
