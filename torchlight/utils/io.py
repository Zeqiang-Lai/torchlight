def save_ani(img_list, filename='a.gif', fps=60):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    def animation_generate(img):
        ims_i = []
        im = plt.imshow(img, cmap='gray')
        ims_i.append([im])
        return ims_i

    ims = []
    fig = plt.figure()
    for img in img_list:
        ims += animation_generate(img)
    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, fps=fps, writer='ffmpeg')
