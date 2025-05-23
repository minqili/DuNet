import os
import datetime
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import glob
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.DuNet import DUNET
from model.Memory_module import MemoryBank


if __name__ == '__main__':

    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%m")


    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

        loss0 = bce_loss(d0, labels_v)
        loss1 = bce_loss(d1, labels_v)
        loss2 = bce_loss(d2, labels_v)
        loss3 = bce_loss(d3, labels_v)
        loss4 = bce_loss(d4, labels_v)
        loss5 = bce_loss(d5, labels_v)
        loss6 = bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
            loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
            loss5.data.item(), loss6.data.item()))

        return loss0, loss


    # ------- 2. set the directory of training dataset --------

    model_name = 'DuNet'

    data_dir = os.path.join(os.getcwd(), 'test_data' + os.sep)

    tra_image_dir = os.path.join('SIRST-TE', 'images')
    tra_label_dir = os.path.join('SIRST-TE', 'masks')

    memory_bank = MemoryBank(normal_dataset=tra_label_dir)
    image_ext = '.png'
    label_ext = '.png'
    model_dir = os.path.join(os.getcwd(), 'saved_model', model_name + os.sep)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epoch_num = 300
    batch_size_train = 4
    batch_size_val = 2
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '//*' + image_ext)
    tra_lbl_name_list = []

    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + '//' + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train,
                                   shuffle=False, num_workers=0, drop_last=True)  # shuffle=True 乱序

    # #############Z
    log_dir = 'runs/logs/SIRST'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    losses = []
    lr_values = []

    # ------- 3. define model --------
    net = DUNET(3, 1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000

    for epoch in range(0, epoch_num):
        net.train()
        # memory_bank.update(feature_extractor=net)
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            processed_inputs = memory_bank.select(inputs)

            # # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(processed_inputs.cuda(),
                                              requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(processed_inputs,
                                              requires_grad=False), Variable(labels, requires_grad=False)
            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            writer.add_scalar('Training Loss', loss.data.item(), ite_num)
            writer.add_scalar('Training Target Loss', loss2.data.item(), ite_num)

            losses.append(loss.data.item())
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                    ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()
                ite_num4val = 0
            torch.cuda.empty_cache()
    writer.close()