import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from PIL import Image
import cv2
import time

# Torch
import torch
from torch import nn
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# ONNX: pip install onnx, onnxruntime
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

# CUDA & TensorRT
#import pycuda.driver as cuda 
from cuda import cuda 
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT')
    
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')
        
    # Sample image
    parser.add_argument('--batch_size', type=int, help='data batch size',
        default=1)
    parser.add_argument('--img_size', help='input size',
        default=[3, 224, 224])
    parser.add_argument('--sample_folder_path', help='sample image folder path',
        default='./imagenet-mini/train')
    #parser.add_argument('--sample_image_path', help='sample image path',
        #default='./sample.jpg')

    # Model path
    parser.add_argument('--onnx_model_path',  help='onnx model path',
        default='./onnx_model.onnx')
    parser.add_argument('--tensorrt_engine_path',  help='tensorrt engine path',
        default='./tensorrt_engine.engine')

    # TensorRT engine params
    parser.add_argument('--dynamic_axes', help='dynamic batch input or output',
        default='True')
    parser.add_argument('--engine_precision', help='precision of TensorRT engine', choices=['FP32', 'FP16'], 
    	default='FP32')
    parser.add_argument('--min_engine_batch_size', type=int, help='set the max input data size of model for inference', 
    	default=1)
    parser.add_argument('--opt_engine_batch_size', type=int, help='set the most used input data size of model for inference', 
    	default=1)
    parser.add_argument('--max_engine_batch_size', type=int, help='set the min input data size of model for inference', 
    	default=8)
    parser.add_argument('--engine_workspace', type=int, help='workspace of engine', 
    	default=1024)
        
    args = string_to_bool(parser.parse_args())

    return args

def string_to_bool(args):

    if args.dynamic_axes.lower() in ('true'): args.dynamic_axes = True
    else: args.dynamic_axes = False

    return args

def get_transform(img_size):
    options = []
    options.append(transforms.Resize((img_size[1], img_size[2])))
    options.append(transforms.ToTensor())
    #options.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(options)
    return transform
    
'''
def load_image(img_path, size):
    img_raw = io.imread(img_path)
    img_raw = np.rollaxis(img_raw, 2, 0)
    img_resize = resize(img_raw / 255, size, anti_aliasing=True)
    img_resize = img_resize.astype(np.float32)
    return img_resize, img_raw
    '''

def load_image_folder(folder_path, img_size, batch_size):
    transforming = get_transform(img_size)
    dataset = datasets.ImageFolder(folder_path, transform=transforming)
    data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1)
    data_iter = iter(data_loader)
    torch_images, class_list = next(data_iter)
    print('class:', class_list)
    print('torch images size:', torch_images.size())
    save_image(torch_images[0], 'sample.png')
    
    return torch_images.cpu().numpy()

def build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, dynamic_axes, \
	img_size, batch_size, min_engine_batch_size, opt_engine_batch_size, max_engine_batch_size):
    
    # Builder
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)
    # Set FP16 
    if engine_precision == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
            	print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    
    # Input
    inputTensor = network.get_input(0) 
    # Dynamic batch (min, opt, max)
    print('inputTensor.name:', inputTensor.name)
    if dynamic_axes:
        profile.set_shape(inputTensor.name, (min_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(opt_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
        print('Set dynamic')
    else:
        profile.set_shape(inputTensor.name, (batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2]))
    config.add_optimization_profile(profile)
    #network.unmark_output(network.get_output(0))
    
    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)
        
def trt_inference(engine, context, data, batch_size, img_size, device):  
    
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    print('nInput:', nInput)
    print('nOutput:', nOutput)
    
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput,nInput+nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
        
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cuda.cuMemAlloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cuda.cuMemcpyHtoD(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes)
    
    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cuda.cuMemcpyDtoH(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes)
        
    for b in bufferD:
        cuda.cuMemFree(b)  
    
    return bufferH

def main():
    args = parse_args()

    # Sample images (folder)
    print(args.sample_folder_path)
    img_resize = load_image_folder(args.sample_folder_path, args.img_size, args.batch_size).astype(np.float32)
    '''
    # Sample (one image)
    print(args.sample_image_path)
    img_resize, img_raw = load_image(args.sample_image_path, args.img_size)
    '''
            
    print('inference image size:', img_resize.shape)
    
    # Build TensorRT engine
    build_engine(args.onnx_model_path, args.tensorrt_engine_path, args.engine_precision, args.dynamic_axes, \
    	args.img_size, args.batch_size, args.min_engine_batch_size, args.opt_engine_batch_size, args.max_engine_batch_size)
    
    # Read the engine from the file and deserialize
    with open(args.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
        engine = runtime.deserialize_cuda_engine(f.read())    
    context = engine.create_execution_context()
    
    # TensorRT inference
    context.set_binding_shape(0, (args.batch_size, args.img_size[0], args.img_size[1], args.img_size[2]))
    
    trt_start_time = time.time()
    trt_outputs = trt_inference(engine, context, img_resize, args.batch_size, args.img_size, args.device)
    trt_outputs = np.array(trt_outputs[1]).reshape(args.batch_size, -1)
    trt_end_time = time.time()
    
    # ONNX inference
    onnx_model = onnx.load(args.onnx_model_path)
    sess = rt.InferenceSession(args.onnx_model_path)
    
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [
        node.name for node in onnx_model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1
    
    sess_input = sess.get_inputs()[0].name
    sess_output = sess.get_outputs()[0].name
    
    onnx_start_time = time.time()
    onnx_result = sess.run([sess_output], {sess_input: img_resize})[0]
    onnx_end_time = time.time()
    
    # Pytorch inference
    resnet18 = models.resnet18(pretrained=True).to(args.device)
    resnet18.eval()
    
    img_resize_torch = torch.Tensor(img_resize).to(args.device)
    torch_start_time = time.time()
    pytorch_result = resnet18(img_resize_torch)
    torch_end_time = time.time()
    pytorch_result = pytorch_result.detach().cpu().numpy()
        
    ## Comparision output of TensorRT and output of onnx model

    # Time Efficiency & output
    print('--pytorch--')
    print(pytorch_result.shape) # (batch_size, 1000)
    print(pytorch_result[0][:10])
    print(np.argmax(pytorch_result, axis=1))
    print('Time:', torch_end_time - torch_start_time)

    print('--onnx--')
    print(onnx_result.shape)
    print(onnx_result[0][:10])
    print(np.argmax(onnx_result, axis=1))
    print('Time: ', onnx_end_time - onnx_start_time)
    
    print('--tensorrt--')
    print(trt_outputs.shape)
    print(trt_outputs[0][:10])
    print(np.argmax(trt_outputs, axis=1))
    print('Time: ', trt_end_time - trt_start_time)
    
if __name__ == '__main__': 
    main()
