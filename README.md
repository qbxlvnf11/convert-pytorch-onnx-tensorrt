

Description
=============

#### - ONNX (Open Neural Network Exchange)
  - Standard format for expressing machine learning algorithms and models
  - More details about ONNX: https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange

#### - TensorRT
  - NVIDIA SDK for high-performance deep learning inference
  - Deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications
  - More details about TensorRT: https://blog.naver.com/qbxlvnf11/222403199156
  
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

Contents
=============
#### - [Converting Pytorch to onnx]()
  - Details: https://blog.naver.com/qbxlvnf11/222342675767
  - Export onnx
  - Load onnx
  - Inference onnx
  - Comparision output of onnx and output of Pytorch (same or not)
  - Dynamic axes (implicit batch)

#### - [Converting onnx to TensorRT]()
  - Build & save TensorRT engine
  - Key trtexec options
    - Precision of engine: TF32, FP32, FP16, ...
    - optShapes: set the most used input data size of model for inference
    - minShapes: set the max input data size of model for inference
    - maxShapes: set the min input data size of model for inference

#### - [Converting onnx to TensorRT]()
  - Load TensorRT engine
  - Inference TensorRT engine
  - Comparision output of TensorRT and output of onnx
  - Comparision of time efficiency
  
Examples of inferencing ResNet18 with TensorRT engine and evaluation of inference time (time efficiency)
=============

#### - Converting Pytorch model to onnx
```
python convert_pytorch_to_onnx/convert_pytorch_to_onnx.py --dynamic_axes true
```

#### - Converting onnx to TensorRT
  - Refer to 'Converting onnx to TensorRT' in Contents section
  - Build TensorRT engine with textexec command in convert_onnx_to_tensorrt.ipynb
  - Run jupyter notebook in TensorRT Docker environment
  ```
  jupyter notebook --ip='0.0.0.0' --port=9000 --allow-root
  ```

#### - TensorRT Engine inference
```
python tensorrt_engine_inference/tensorrt_engine_inference.py --tensorrt_engine_path {engine_name} --batch_size {test_batch_size}
```

#### - Comparision of time efficiency
  - Time efficiency (vary from case to case)
    - onnx: 0.010984
    - TensorRT (TF32): 0.000288
    - TensorRT (FP16): 0.000240
  
References
=============

#### - Converting Pytorch models to onnx

https://pytorch.org/docs/stable/onnx.html

#### - TensorRT

https://developer.nvidia.com/tensorrt

#### - TensorRT Release 21.08

https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-08.html#rel_21-08

#### - TF32

https://blogs.nvidia.co.kr/2020/05/22/tensorfloat-32-precision-format/

#### - ImageNet 1000 samples

https://www.kaggle.com/ifigotin/imagenetmini-1000

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

