//
//  ViewController.swift
//  Tiramisu
//
//  Created by James Kauten on 10/15/18.
//  Copyright © 2018 Kautenja. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import Metal
import MetalPerformanceShaders

/// A view controller to pass camera inputs through a vision model
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    /// a local reference to time to update the framerate
    var time = Date()
    
    var ready: Bool = true
    var original:UIImage?
    var inoutSizee:CGSize?
    private lazy var flashMode = AVCaptureDevice.FlashMode.off
    private lazy var devicePosition = AVCaptureDevice.Position.back
    private var input: AVCaptureDeviceInput?
    /// the view to preview raw RGB data from the camera
    @IBOutlet weak var preview: UIView!
    /// the view for showing the segmentation
    @IBOutlet weak var segmentation: UIImageView!
    /// a label to show the framerate of the model
    @IBOutlet weak var framerate: UILabel!
    
    /// the camera session for streaming data from the camera
    var captureSession: AVCaptureSession!
    /// the video preview layer
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    
    /// TODO:
    private var _device: MTLDevice?
    /// TODO:
    var device: MTLDevice! {
        get {
            // try to unwrap the private device instance
            if let device = _device {
                return device
            }
            _device = MTLCreateSystemDefaultDevice()
            return _device
        }
    }
    
    var _queue: MTLCommandQueue?
    
    var queue: MTLCommandQueue! {
        get {
            // try to unwrap the private queue instance
            if let queue = _queue {
                return queue
            }
            _queue = device.makeCommandQueue()
            return _queue
        }
    }

    /// the model for the view controller to apss camera data through
    private var _model: VNCoreMLModel?
    /// the model for the view controller to apss camera data through
    var model: VNCoreMLModel! {
        get {
            // try to unwrap the private model instance
            if let model = _model {
                return model
            }
            // try to create a new model and fail gracefully
            do {
                _model = try VNCoreMLModel(for: deeplab_dm05_float32().model)
            } catch let error {
                let message = "failed to load model: \(error.localizedDescription)"
                popup_alert(self, title: "Model Error", message: message)
            }
            return _model
        }
    }
    
    /// the request and handler for the model
    private var _request: VNCoreMLRequest?
    /// the request and handler for the model
    var request: VNCoreMLRequest! {
        get {
            // try to unwrap the private request instance
            if let request = _request {
                return request
            }
            // create the request
            _request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
                // handle an error from the inference engine
                if let error = error {
                    print("inference error: \(error.localizedDescription)")
                    return
                }
                // make sure the UI is ready for another frame
                guard self.ready else { return }
                // get the outputs from the model
                let outputs = finishedRequest.results as? [VNCoreMLFeatureValueObservation]
                // get the probabilities as the first output of the model
                guard let softmax = outputs?[0].featureValue.multiArrayValue else {
                    print("failed to extract output from model")
                    return
                }
                // get the dimensions of the probability tensor
                let channels = softmax.shape[0].intValue
                let height = softmax.shape[1].intValue
                let width = softmax.shape[2].intValue
                                
                // create an image for the softmax outputs
                let desc = MPSImageDescriptor(channelFormat: .float32,
                                              width: width,
                                              height: height,
                                              featureChannels: channels)
                let probs = MPSImage(device: self.device, imageDescriptor: desc)
                probs.writeBytes(softmax.dataPointer,
                                 dataLayout: .featureChannelsxHeightxWidth,
                                 imageIndex: 0)
                
                // create an output image for the Arg Max output
                let desc1 = MPSImageDescriptor(channelFormat: .float32,
                                               width: width,
                                               height: height,
                                               featureChannels: 1)
                let classes = MPSImage(device: self.device, imageDescriptor: desc1)

                // create a buffer and pass the inputs through the filter to the outputs
                let buffer = self.queue.makeCommandBuffer()
                let filter = MPSNNReduceFeatureChannelsArgumentMax(device: self.device)
                filter.encode(commandBuffer: buffer!, sourceImage: probs, destinationImage: classes)
                
                // add a callback to handle the buffer's completion and commit the buffer
                buffer?.addCompletedHandler({ (_buffer) in
                    let argmax = try! MLMultiArray(shape: [1, softmax.shape[1], softmax.shape[2]], dataType: .float32)
                    classes.readBytes(argmax.dataPointer,
                                      dataLayout: .featureChannelsxHeightxWidth,
                                      imageIndex: 0)
    
                    // unmap the discrete segmentation to RGB pixels
                    var image = codesToImage(argmax)
                    
                    
                    
                   image = image?.resize(size: self.inoutSizee)
                    
                         //self.imageView.image = self.overLay(original: self.mImage!, mask: segmentedMaskImage.resize(to: input.size)!)//can overlay here
                    
                    // update the image on the UI thread
                    DispatchQueue.main.async {
                        self.segmentation.image = self.overLay(original: self.original!, mask: image!)
                        let fps = -1 / self.time.timeIntervalSinceNow
                        self.time = Date()
                        self.framerate.text = "\(fps)"
                    }
                    self.ready = true
                })
                self.ready = false
                buffer?.commit()

            }
            // set the input image size to be a scaled version
            // of the image
            _request?.imageCropAndScaleOption = .scaleFill
            return _request
        }
    }
    
    /// Respond to a memory warning from the OS
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        popup_alert(self, title: "Memory Warning", message: "received memory warning")
    }
    
    /// Handle the view appearing
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // setup the AV session
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720
        
        // get a handle on the back camera
        //AVCaptureDevice.default(for: AVMediaType.video)
        guard let camera = cameraWithPosition(position: AVCaptureDevice.Position.front) else {
            //
            
            let message = "Unable to access the back camera!"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
        // create an input device from the back camera and handle
        // any errors (i.e., privacy request denied)
        do {
            AVCaptureDevice.Position.front
            // setup the camera input and video output
            let input = try AVCaptureDeviceInput(device: camera)
            //input.device.position = AVCaptureDevice.Position.front
            input.device.position
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            // add the inputs and ouptuts to the sessionr and start the preview
            if captureSession.canAddInput(input) && captureSession.canAddOutput(videoOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(videoOutput)
                setupCameraPreview()
            }
        }
        catch let error  {
            let message = "failed to intialize camera: \(error.localizedDescription)"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
    }

    
    func cameraWithPosition(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let discoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .unspecified)
        for device in discoverySession.devices {
            if device.position == position {
                return device
            }
        }
        
        return nil
    }
    
    /// Setup the live preview from the camera
    func setupCameraPreview() {
        // create a video preview layer for the view controller
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        // set the metadata of the video preview
        videoPreviewLayer.videoGravity = .resizeAspect
        
        videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        // add the preview layer as a sublayer of the preview view
        preview.layer.addSublayer(videoPreviewLayer)
        // start the capture session asyncrhonously
        DispatchQueue.global(qos: .userInitiated).async {
            // start the capture session in the background thread
            self.captureSession.startRunning()
            // set the frame of the video preview to the bounds of the
            // preview view
            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.preview.bounds
            }
        }
    }
    
    /// Handle a frame from the camera video stream
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
        // Applications generating frames, compressing or decompressing video, or using Core Image
        // can all make use of Core Video pixel buffers
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            let message = "failed to create pixel buffer from video input"
            popup_alert(self, title: "Inference Error", message: message)
            return
        }
        // execute the request
        do {
           // let ciimage = CIImage(cvPixelBuffer: pixelBuffer)
            let imageCg = CGImage.create(pixelBuffer: pixelBuffer)
            self.original = UIImage(cgImage: imageCg!)
            self.inoutSizee = original?.size
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        } catch let error {
            let message = "failed to perform inference: \(error.localizedDescription)"
            popup_alert(self, title: "Inference Error", message: message)
        }
    }
    
}
extension ViewController{
    
    func overLay(original: UIImage, mask: UIImage) -> UIImage{
        /*var size = CGSize(width: original.size.width, height: original.size.height)
         UIGraphicsBeginImageContext(size)
         
         let areaSize = CGRect(x: 0, y: 0, width: size.width, height: size.height)
         original.draw(in: areaSize)
         
         mask.draw(in: areaSize, blendMode: .normal, alpha: 0.8)
         
         var newImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()!
         UIGraphicsEndImageContext()
         
         let ms = UIImage(cgImage: CGImage.masking(mask.cgImage))
         
         return ms*/
        
        return maskImage(image: original, mask: mask)
        
        
    }
    
    func invert(originalImage: UIImage) -> UIImage{
        
        let image = CIImage(cgImage: originalImage.cgImage!)
        if let filter = CIFilter(name: "CIColorInvert") {
            filter.setDefaults()
            filter.setValue(image, forKey: kCIInputImageKey)
            
            let context = CIContext(options: nil)
            let imageRef = context.createCGImage(filter.outputImage!, from: image.extent)
            print("Inverted done")
            return UIImage(cgImage: imageRef!)
        }
        
        return originalImage
        
    }
    
    
    
    func maskImage(image:UIImage, mask:(UIImage))->UIImage{
     
        let str = CFAbsoluteTimeGetCurrent()
        
        // Inverter
        //let maskm = invert(originalImage: mask)
        
        
        
        guard
        let imageReference = image.cgImage,
            let maskReference = mask.cgImage else {
                fatalError("Reference found nil")
        }
      
       
        
        
        guard
        let imageMask = CGImage(maskWidth: maskReference.width,
                                height: maskReference.height,
                                bitsPerComponent: maskReference.bitsPerComponent,
                                bitsPerPixel: maskReference.bitsPerPixel,
                                bytesPerRow: maskReference.bytesPerRow,
                                provider: maskReference.dataProvider!, decode: nil, shouldInterpolate: true)
            else {
            fatalError("image Mask isuue")
        }
        let maskedReference = imageReference.masking(imageMask)
                
        _ = imageReference.masking(imageMask)
                
                let maskedImage = UIImage(cgImage:maskedReference!)
                let end = CFAbsoluteTimeGetCurrent()
                print("Masking Time \((end-str)*100))")
                return maskedImage
        
       
    }
    
    
}

extension UIImage {
    
    func resize(size: CGSize!) -> UIImage? {
        let rect = CGRect(x: 0.0, y: 0.0, width: size.width, height: size.height)
        UIGraphicsBeginImageContext(rect.size)
        self.draw(in:rect)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return img
    }
    
}


fileprivate class SmoothFilter : CIFilter {
    
    private let kernel: CIColorKernel
    var inputImage: CIImage?
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override init() {
        let kernelStr = """
            kernel vec4 myColor(__sample source) {
                float maskValue = smoothstep(0.3, 0.5, source.r);
                return vec4(maskValue,maskValue,maskValue,1.0);
            }
        """
        let kernels = CIColorKernel.makeKernels(source:kernelStr)!
        kernel = kernels[0] as! CIColorKernel
        super.init()
    }
    
    override var outputImage: CIImage? {
        guard let inputImage = inputImage else {return nil}
        let blurFilter = CIFilter.init(name: "CIGaussianBlur")!
        blurFilter.setDefaults()
        blurFilter.setValue(inputImage.extent.width / 90.0, forKey: kCIInputRadiusKey)
        blurFilter.setValue(inputImage, forKey: kCIInputImageKey)
        let bluredImage = blurFilter.value(forKey:kCIOutputImageKey) as! CIImage
        return kernel.apply(extent: bluredImage.extent, arguments: [bluredImage])
    }
}

