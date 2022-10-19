

Description
=============

#### - TensorRT engine convertor of various TensorRT versions (refer to each branch)

#### - ONNX (Open Neural Network Exchange)
  - Standard format for expressing machine learning algorithms and models
  - More details about ONNX: https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange

#### - TensorRT
  - NVIDIA SDK for high-performance deep learning inference
  - Deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications
  - Explicit batch is required when you are dealing with Dynamic shapes, otherwise network will be created using implicit batch dimension.
  - More details about TensorRT: https://blog.naver.com/qbxlvnf11/222403199156

Contents
=============
#### - [Converting Pytorch to onnx](https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt/blob/TensorRT-21.08/convert_pytorch_to_onnx/convert_pytorch_to_onnx.py)
  - Details: https://blog.naver.com/qbxlvnf11/222342675767
  - Export & load onnx
  - Inference onnx
  - Compare output and time efficiency between onnx and pytorch
  - Setting batch size of input data: explicit batch or implicit batch

#### - [Converting onnx to TensorRT and test time efficiency](https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt/blob/TensorRT-21.08/convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py)
  - Build & load TensorRT engine
  - Setting batch size of input data: explicit batch or implicit batch
  - Key trtexec options
    - Precision of engine: FP32, FP16
    - optShapes: set the most used input data size of model for inference
    - minShapes: set the max input data size of model for inference
    - maxShapes: set the min input data size of model for inference  
  - Inference TensorRT engine
  - Compare output and time efficiency among tensorrt and onnx and pytorch

TensorRT Docker Environment
=============

#### - Download TensorRT Docker environment
```
docker pull qbxlvnf11docker/tensorrt_21.08
```

#### - Run TensorRT Docker environment
```
nvidia-docker run -it -p 9000:9000 -e GRANT_SUDO=yes --user root --name tensorrt_21.08_env -v {code_folder_path}:/workspace -w /workspace qbxlvnf11docker/tensorrt_21.08:latest bash
```

Examples of inferencing ResNet18 with TensorRT
=============

#### - Explicit batch
  - Converting Pytorch model to onnx
  ```
  python convert_pytorch_to_onnx/convert_pytorch_to_onnx.py --dynamic_axes True --output_path onnx_output_explicit.onnx --batch_size {batch_size}
  ```

  - Converting onnx to TensorRT and test time efficiency (FP32)
    - Setting three parameters (minShapes, optShapes, maxShapes) according to the inference environment
  ```
  python convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py --dynamic_axes True --onnx_model_path onnx_output_explicit.onnx --batch_size {batch_size} --tensorrt_engine_path FP32_explicit.engine --engine_precision FP32 
  ```  

  - Converting onnx to TensorRT and test time efficiency (FP16)
    - Setting three parameters (minShapes, optShapes, maxShapes) according to the inference environment
  ```
  python convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py --dynamic_axes True --onnx_model_path onnx_output_explicit.onnx --batch_size {batch_size} --tensorrt_engine_path FP16_explicit.engine --engine_precision FP16 
  ```  

#### - Implicit batch
  - Converting Pytorch model to onnx
  ```
  python convert_pytorch_to_onnx/convert_pytorch_to_onnx.py --dynamic_axes False --output_path onnx_output_implicit.onnx --batch_size {batch_size}
  ```
  
  - Converting onnx to TensorRT and test time efficiency (FP32)
  ```
  python convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py --dynamic_axes False --onnx_model_path onnx_output_implicit.onnx --batch_size {batch_size_of_implicit_batch_onnx_model} --tensorrt_engine_path FP32_implicit.engine --engine_precision FP32 
  ```  

  - Converting onnx to TensorRT and test time efficiency (FP16)
  ```
  python convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py --dynamic_axes False --onnx_model_path onnx_output_implicit.onnx --batch_size {batch_size_of_implicit_batch_onnx_model} --tensorrt_engine_path FP16_implicit.engine --engine_precision FP16 
  ```  

#### - Comparision of time efficiency and output
  - Explicit batch test of FP32 TensorRT engine
    - Batch size of inf data = 1
    - Batch size of optShapes = 1
    
    <img src="https://user-images.githubusercontent.com/52263269/196143388-9508444c-29ac-481a-abe4-3297e27dbdb7.png" width="35%"></img>

  - Explicit batch test of FP16 TensorRT engine
    - Batch size of inf data = 1
    - Batch size of optShapes = 1
    
    <img src="https://user-images.githubusercontent.com/52263269/196143922-f42d3e5b-431c-4f34-acfc-84debe0dba3e.png" width="35%"></img>


  - Explicit batch test of FP16 TensorRT engine
    - Batch size of inf data = 8
    - Batch size of optShapes = 1
    
    <img src="https://user-images.githubusercontent.com/52263269/196144388-bd83d533-bb6c-4a82-b021-86c341155fad.png" width="35%"></img>

  - Implicit batch test of FP32 TensorRT engine
    - Batch size of inf data = 1
    
    <img src="https://user-images.githubusercontent.com/52263269/196143184-16a70676-eaf0-4e38-89a0-96030939619d.png" width="35%"></img>

  - Implicit batch test of FP16 TensorRT engine
    - Batch size of inf data = 1
    
    <img src="https://user-images.githubusercontent.com/52263269/196143568-3794c96f-d530-455f-861e-b14d63c2c04e.png" width="35%"></img>

References
=============

#### - Converting Pytorch models to onnx

https://pytorch.org/docs/stable/onnx.html

#### - TensorRT

https://developer.nvidia.com/tensorrt

#### - TensorRT Release 21.08

https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-08.html#rel_21-08

#### - TensorRT8 code

https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/04-Parser/pyTorch-ONNX-TensorRT/main.py

#### - ImageNet 1000 samples

https://www.kaggle.com/ifigotin/imagenetmini-1000

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

