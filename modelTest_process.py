from data_process import getLoaders, visualize
import torch
import numpy as np



DATA_DIR = "d:/path/to/CamVid"

CLASSES = ["sky",'building', 'pole', 'road', 'pavement',
           'tree', 'signsymbol', 'fence', 'car',
           'pedestrian', 'bicyclist', 'unlabelled']

# here not need "back" just add yours classes name
# CLASSES = ["your_class_name","your_class_name" .... ..]

DEVICE = 'cuda'

best_model = torch.load('./best_model.pth')
# Change data_channel and preproc
_,_,test_dataset,test_dataset_vis = getLoaders(DATA_DIR,CLASSES,data_channel=3,preproc=True)

def combine_masks(masks):
  # masks should be size (channels, w, h)
  output_mask = np.zeros(masks[0].shape, dtype=np.uint8)

  for i, mask in enumerate(masks):
    output_mask[mask==1] = i + 1

  return output_mask
# n = np.random.choice(4)
for n in range(100):
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze(1).cpu().numpy().round())


    pr_mask = combine_masks(pr_mask.squeeze())
    if len(CLASSES) > 1:
        gt_mask = combine_masks(gt_mask)



    print(f"GT: {np.unique(gt_mask)}")
    print(f"PR:  {np.unique(pr_mask)}")

    visualize(
        image=image_vis.squeeze(),
        ground_truth_mask=gt_mask,
        # colHchang = colHchang,
        predicted_mask=pr_mask
    )