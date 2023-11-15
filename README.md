# jetson-vision-YoloV8_TensorRT
Inference: https://github.com/triple-Mu/YOLOv8-TensorRT

Alur convert model:  PyTorch(trained yolov8 model) -> ONNX -> TensorRT Engine

Untuk convert model yolov8.pt ke onnx, rada susah pake jetson, jadi convert dulu jadi onnx pake windows pake modul ultralytics

## Instal ultralytics
GitHub ultralytics: https://github.com/ultralytics/ultralytics

Diperlukan modifikasi modul ultralytics, karena hasil export onnx dari ultralytics outputnya ngawuer

https://medium.com/@smallerNdeeper/yolov8-batch-inference-implementation-using-tensorrt-2-converting-to-batch-model-engine-e02dc203fc8b

Modif file head.py (ultralytics - nn - modules - head.py) jadi gini:
```
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        conf, label = cls.sigmoid().max(1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        dbox = dbox.transpose(1,2)
        return (dbox, conf, label) if self.export else (y, x)
```

## Instal YOLOv8-TensorRT
GitHub YOLOv8-TensorRT: https://github.com/triple-Mu/YOLOv8-TensorRT

Versi ONNX yang dipake: 1.6.0

Versi ONNXRuntime-gpu yang dipake: 1.6.0 (https://elinux.org/Jetson_Zoo#ONNX_Runtime)

Versi CUDA: 10.2.300

Versi TensorRT: 8.0.1.6

Versi OpenCV: 4.8.0 with CUDA

## Cara Convert Model
Bikin dulu model yang dah di train (pytorch model .pt)

Kemudian convert .pt ke .onnx pake ultralytics yang di windows (gatau kenapa gabisa di jetson langsung)
```
yolo export model=<NAMA MODEL>.pt format=onnx simplify=True half=True imgsz=480
```
panduan parameter export: https://docs.ultralytics.com/modes/export/#arguments

Setelah model dikonvert jadi .onnx, masukin ke jetson buat konvert .onnx ke .engine:
```
/usr/src/tensorrt/bin/trtexec --onnx=<NAMA MODEL>.onnx --saveEngine=<NAMA MODEL>.engine
```

selesai deh, buat ngecek modelnya bisa apa ga:
```
/usr/src/tensorrt/bin/trtexec --loadEngine=<NAMA MODEL>.engine
```

## Cara ngetest inference pake webcam
```
python3 camera.py
```
