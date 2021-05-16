from data_process import getLoaders, visualize
import numpy as np

def combine_masks(masks):
    # masks should be size (channels, w, h)
    output_mask = np.zeros(masks[0].shape, dtype=np.uint8)

    for i, mask in enumerate(masks):
        output_mask[mask == 1] = i + 1

    return output_mask

    # n = 1
def main(dataset,img_chan):
    for n in range(100,200):
        # image_vis = test_dataset_vis[n][0].astype('uint8')

        image, gt_mask = dataset[n]


        if img_chan == 1:
            image = image.squeeze()
            print("img_chan: 1")
        elif img_chan == 3:
            image = image.transpose(1, 2, 0)
            print("img_chan: 3")
        else:
            print("img channel incorrect! Try 1 or 3 channels!")
            exit()
        print(image.shape)

        print(gt_mask.shape)
        gt_mask = combine_masks(gt_mask)
        print(np.unique(gt_mask))
        # break

        # gt_mask = combine_masks(gt_mask)
        # print(np.unique(gt_mask))
        # print(gt_mask.shape)
        visualize(
            image=image,
            mask = gt_mask
        )
if __name__ == '__main__':
    DATA_DIR = "D:\\data\\CamVid"

    CLASSES = ["sky", 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    # your classes should be like this
    # here not need "back" just add yours classes name
    # CLASSES = ["your_class_name","your_class_name" .... ..]

    # your img channel
    # if img channel 1 not need preproces so make it False
    img_channel = 3
    prepocess_data = True
    _,_,m_dataset, _ = getLoaders(DATA_DIR, CLASSES, data_channel=img_channel,
                                                             preproc=prepocess_data)
    main(m_dataset,img_channel)