import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy
from demo import get_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from faceformer import Faceformer
import tensorrt

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    print ("Create the save folder...")
    save_path = os.path.join(args.dataset,args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    print ("Start training...")
    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            loss = model(audio, template,  vertice, one_hot, criterion,teacher_forcing=False)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration ,np.mean(loss_log)))
        # validation
        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all,file_name in dev_loader:
            # to gpu
            audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                loss = model(audio, template,  vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    loss = model(audio, template,  vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                        
        current_loss = np.mean(valid_loss_log)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e+1,current_loss))    
    return model

@torch.no_grad()
def test(args, model, test_loader,epoch, criterion):
    
    print ("Create the result folder...")
    result_path = os.path.join(args.dataset,args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    print("Load the model...")
    model = get_model(args, os.path.join(save_path, '{}_model.pth'.format(epoch)))
    if os.path.exists(os.path.join(save_path, '{}_model.pth'.format(epoch))):
        print ("model loaded successfully")
    else:
        print ("model loaded failed")

    # to cuda
    model = model.to(torch.device("cuda"))
    model.eval()
   
    print ("Start testing...")
    test_subjects_list = [i for i in args.test_subjects.split(" ")]
    test_loss_log = []

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)

            #print("shapes:", prediction.shape, vertice.shape)
            loss = criterion(prediction, vertice[:,1:prediction.shape[1]+1,:])

            test_loss_log.append(loss.item())
            print ("test loss: ", loss.item())
            prediction = prediction.squeeze() # (seq_len, V*3)
            
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)

                #print("shapes:", prediction.shape, vertice.shape)
                loss = criterion(prediction, vertice[:,1:prediction.shape[1]+1,:])

                print ("test loss: ", loss.item())
                prediction = prediction.squeeze() # (seq_len, V*3)
                test_loss_log.append(loss.item())
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
    
    print ("mean test loss: ", np.mean(test_loss_log))
    np.save(os.path.join(result_path, "test_loss.npy"), np.mean(test_loss_log))    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    parser.add_argument("--onlyTestFlag", type=bool, default=False)
    parser.add_argument("--int8_quantization", type=str, default="", help='')
    parser.add_argument("--optimize_last_layer", type=bool, default=False, help='Dont calculate linear layer for all')
    parser.add_argument("--set_seed", type=bool, default=False, help='')
    parser.add_argument('--static_quantized_layers', nargs='*', help='<Layers to quantize', default=[])
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--calibration_wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')

    args = parser.parse_args()

    # print tesnorrt version
    print("tensorrt version: ", tensorrt.__version__)

    # print cuda version
    print("cuda version: ", torch.version.cuda)

    # print pytorch version
    print("pytorch version: ", torch.__version__)

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    
    #load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    if (args.onlyTestFlag == False):
        # Train the model
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
        model = trainer(args, dataset["train"], dataset["valid"],model, optimizer, criterion, epoch=args.max_epoch)
    else:
        print ("Only test the model...")

    test(args, model, dataset["test"], epoch=args.max_epoch, criterion = criterion)
    print ("Finito!")
    
if __name__=="__main__":
    main()