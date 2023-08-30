import numpy as np
import scipy.io.wavfile as wav
import librosa
import os,sys,shutil,argparse,copy,pickle
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
  QConfigMapping,
)
import bitsandbytes
from bitsandbytes.utils import replace_linear
from bitsandbytes.nn import Linear8bitLt
import torch.quantization.quantize_fx as quantize_fx
import copy
import cv2
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # egl
import pyrender
from psbody.mesh import Mesh
import trimesh
import random
from torch.profiler import profile, record_function, ProfilerActivity

# def compute_error(x, y):
#     Ps = torch.norm(x)
#     Pn = torch.norm(x-y)
#     return 20*torch.log10(Ps/Pn)


import torch.nn as nn
from accelerate import init_empty_weights

def replace_8bit_linear(model, module_to_not_convert, threshold=6.0):
    """ This function is used for the LLMint8 quantization. It replaces all the linear layers with the 8bit linear layers."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0 and not name in module_to_not_convert:
            replace_8bit_linear(module, threshold=threshold, module_to_not_convert=module_to_not_convert)
        if isinstance(module, nn.Linear) and not name in module_to_not_convert:
            model._modules[name] = bitsandbytes.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
    return model

@torch.no_grad()
def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if args.set_seed:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        print("Setting seed to 42...")
    #build model
    model = Faceformer(args)
    if args.int8_quantization == "llm_int8":
        model = replace_8bit_linear(model, module_to_not_convert=["transformer_decoder"])
        model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),  map_location=torch.device(args.device)))
        model.half()
        model = model.to(0)
    elif args.int8_quantization == "float16":
        model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),  map_location=torch.device(args.device)))
        model.half()

    else:
        model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),  map_location=torch.device(args.device)))
    model = model.to(torch.device(args.device))
    model.eval()
    
    # Generate old model
    old_model = Faceformer(args)
    old_model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name)),  map_location=torch.device(args.device)))
    old_model = old_model.to(torch.device(args.device))
    old_model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)


    print("Starting to predict...")
    #TODO consider using intel ipex...
    if args.int8_quantization == "static_int8":
        # normed_conv_model = model.audio_encoder.encoder.pos_conv_embed.conv
        # weight_g, weight_v = normed_conv_model.weight_g, normed_conv_model.weight_v        
        model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
        if not "linear_layers" in args.static_quantized_layers:
            model.vertice_map_r.qconfig = None
            model.vertice_map.qconfig = None
            
        #Quantizing vertice_map_r, vertice_map
        model.audio_encoder.qconfig = None
        model.obj_vector.qconfig = None
        model.audio_feature_map.qconfig = None
        model.PPE.qconfig = None
        
        # model.transformer_decoder.qconfig = None
        # Quantizing self_attn multihead_attn
        if "decoder_attention" in args.static_quantized_layers:
            model.transformer_decoder._modules["layers"][0].norm1.qconfig = None
            model.transformer_decoder._modules["layers"][0].norm2.qconfig = None
            model.transformer_decoder._modules["layers"][0].norm3.qconfig = None
            model.transformer_decoder._modules["layers"][0].dropout1.qconfig = None
            model.transformer_decoder._modules["layers"][0].dropout2.qconfig = None
            model.transformer_decoder._modules["layers"][0].dropout3.qconfig = None
            model.transformer_decoder._modules["layers"][0].dropout.qconfig = None
            model.transformer_decoder._modules["layers"][0].linear1.qconfig = None
            model.transformer_decoder._modules["layers"][0].linear2.qconfig = None
        else:
            model.transformer_decoder.qconfig = None

        # Not sure what modules needs to be fused, I just wrote here conv and relu
        #TODO!! use fuzed models
        # model_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'activation']])
        model = torch.ao.quantization.prepare(model)
        model.predict(audio_feature, template, one_hot, args.optimize_last_layer)
        model = torch.ao.quantization.convert(model)

        # Currently pytorch cant convert weight_norm(conv) to quantisized model - so Iam doing it my own and will open an issue for that
        # Maybe it is not the right solution
        # TODO somone asked this in https://discuss.pytorch.org/t/what-is-the-correct-way-to-qat-a-conv-layer-with-weight-norm/160562
        # Consider upgrade pytorch, I think they changed/fix it in this PR: https://github.com/pytorch/pytorch/pull/103001/files/ec49a529e43509179eee48dbaaac3638913b82a5#diff-b6cd5a3dc103e64e85a2a8cec5b3a9af430319a5c7603dcbb9b7d2642a930f6f
        # normed_conv_model = model.audio_encoder.encoder.pos_conv_embed.conv
        # normed_conv_model.register_parameter("weight" + '_g', weight_g)
        # normed_conv_model.register_parameter("weight" + '_v', weight_v)


    elif args.int8_quantization == "dynamic_int8":
        model = torch.ao.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8)  # the target dtype for quantized weights

    print("Model size before quantization: ")
    print_size_of_model(old_model)
    print("Model size after quantization: ")
    print_size_of_model(model)
    print(model)
    print("Starting to predict...")
    activities=[ProfilerActivity.CPU]
    sort_by = "cpu_time_total"
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_by = "cuda_time_total"
    
    with profile(activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/faceformer_{args.int8_quantization}')) as prof:

        if args.int8_quantization == "llm_int8" or args.int8_quantization == "float16":
            model.biased_mask = model.biased_mask.half()
            prediction = model.predict(audio_feature.half(), template.half(), one_hot.half(), args.optimize_last_layer)
        else:
            prediction = model.predict(audio_feature, template, one_hot, args.optimize_last_layer)

    print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by, row_limit=20))

    if args.calculate_mse:
        print("Calculating MSE...")
        non_quantized_prediction = old_model.predict(audio_feature, template, one_hot, args.optimize_last_layer)

        mse = F.mse_loss(prediction, non_quantized_prediction)
        print("MSE IS: ", mse)
    
    prediction = prediction.squeeze() # (seq_len, V*3)
    np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def transform_model_to_int8_fx(model, input_fp32):
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
    # a tuple of one or more example inputs are needed to trace the model
    example_inputs = (input_fp32)
    # prepare
    model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    # no calibration needed when we only have dynamic/weight_only quantization
    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0):
    if args.dataset == "BIWI":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif args.dataset == "vocaset":
        camera_params = {'c': np.array([400, 400]),
                         'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                         'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if args.background_black:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence(args):
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    predicted_vertices_path = os.path.join(args.result_path,test_name+".npy")
    if args.dataset == "BIWI":
        template_file = os.path.join(args.dataset, args.render_template_path, "BIWI.ply")
    elif args.dataset == "vocaset":
        template_file = os.path.join(args.dataset, args.render_template_path, "FLAME_sample.ply")
         
    print("rendering: ", test_name)
                 
    template = Mesh(filename=template_file)
    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        pred_img = render_mesh_helper(args,render_mesh, center)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name+"_"+args.subject+"_condition_"+args.condition

    video_fname = os.path.join(output_path, file_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="biwi")
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    parser.add_argument("--int8_quantization", type=str, default="", help='')
    parser.add_argument("--optimize_last_layer", type=bool, default=False, help='Dont calculate linear layer for all')
    parser.add_argument("--set_seed", type=bool, default=False, help='')
    parser.add_argument("--calculate_mse", type=bool, default=False, help='')
    parser.add_argument('--static_quantized_layers', nargs='*', help='<Layers to quantize', default=[])


    args = parser.parse_args()   

    test_model(args)
    render_sequence(args)

if __name__=="__main__":
    main()
