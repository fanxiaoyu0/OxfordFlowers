plt.rcParams["savefig.bbox"] = 'tight'
def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        # row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()    
    
    
    image_dir='../data/train/0'
    image_name_list=os.listdir(image_dir)
    image_list=[]
    for image_name in image_name_list:
        image_list.append(read_image(image_dir+'/'+image_name))
    print(image_list)
    transform_list=[train_transform,validate_transform,test_transform]
    transformed_image_list=[]
    for transform in transform_list:
        transformed_image_list.append([image for image in image_list])
    plot(image_list)
    plt.savefig('../result/temp/transformed_image.png')
    fshdkj