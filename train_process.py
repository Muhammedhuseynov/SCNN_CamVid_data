import segmentation_models_pytorch as smp
import torch
from model_process import m_SCNN
from data_process import getLoaders

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

def main(DATA_DIR,classes,img_channel=3,prepocess_data=True):


    model = m_SCNN(len(classes),img_channel=img_channel)


    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    DEVICE = "cuda"

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0

    train_loader,valid_loader,test_dataset,_ = getLoaders(DATA_DIR,classes,data_channel=img_channel,preproc=prepocess_data)

    epoch = 100
    for i in range(1, epoch+1):

        print('\nEpoch: {}/{}'.format(i,epoch))
        train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 50:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        if i == 80:
            optimizer.param_groups[0]['lr'] = 1e-8
            print('Decrease decoder learning rate to 1e-8!')


if __name__ == '__main__':
    CLASSES = ["sky",'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    # here not need "back" just add yours classes name
    # CLASSES = ["your_class_name","your_class_name" .... ..]

    DATA_DIR = "D:\\data\\CamVid"

    #here if your img chanel 3 then use preprocess_data = True
    #if your img chanel 1 then dont use preprocess_data = False
    main(DATA_DIR,classes=CLASSES,img_channel=3,prepocess_data=True)