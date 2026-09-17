//
//  RoadDetectionService.swift
//  DriveSensAI
//

import AVFoundation
import Combine
import CoreML
import Foundation
import Vision

/// Rear-camera road object detection via the bundled `RoadObjectDetector` Core ML model + Vision.
/// Named separately from the Xcode-generated `RoadObjectDetector` model class to avoid a symbol clash.
/// Owns its own back-camera `CameraManager`. Only one camera session should run at a time app-wide.
@MainActor
final class RoadDetectionService: ObservableObject {
    @Published private(set) var detections: [RoadDetection] = []
    @Published private(set) var state: RoadMonitoringState = .clear
    @Published private(set) var cameraError: CameraError?
    @Published private(set) var isRunning = false
    @Published private(set) var isModelReady = false
    @Published private(set) var modelLoadMessage: String = "Looking for RoadObjectDetector…"

    /// Easy-to-tune minimum confidence for retained detections.
    nonisolated static let confidenceThreshold: Float = 0.35

    /// Roughly 5 inferences/sec.
    nonisolated static let minimumInferenceInterval: TimeInterval = 0.2

    /// COCO-style labels we care about for road monitoring (easy to extend later).
    nonisolated static let relevantLabels: Set<String> = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck"
    ]

    private let cameraManager = CameraManager(position: .back)

    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false
    private nonisolated(unsafe) var lastInferenceAt: CFAbsoluteTime = 0

    /// Loaded once and reused. Vision request is also reused.
    private nonisolated(unsafe) var visionModel: VNCoreMLModel?
    private nonisolated(unsafe) var detectionRequest: VNCoreMLRequest?

    /// Rear camera + portrait rotation (90°) without mirroring.
    /// Front driver path uses `.leftMirrored`; rear is not mirrored, so `.right` matches
    /// the upright portrait buffers produced with `videoRotationAngle = 90`.
    private nonisolated static let visionOrientation: CGImagePropertyOrientation = .right

    init() {
        loadModel()
    }

    func start() {
        cameraError = nil

        guard isModelReady else {
            state = .modelUnavailable
            print("[RoadDetector] Cannot start — model unavailable.")
            return
        }

        cameraManager.onFrame = { [weak self] pixelBuffer in
            self?.handleFrame(pixelBuffer)
        }

        cameraManager.requestAccessAndStart { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success:
                    self.isRunning = true
                    self.cameraError = nil
                    print("[RoadDetector] Rear camera started.")
                case .failure(let error):
                    self.isRunning = false
                    self.cameraError = error
                    self.detections = []
                    self.state = self.isModelReady ? .clear : .modelUnavailable
                    print("[RoadDetector] Camera start failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func stop() {
        cameraManager.onFrame = nil
        cameraManager.stop()
        isRunning = false
        detections = []
        if isModelReady {
            state = .clear
        }
        processingLock.lock()
        isProcessingFrame = false
        processingLock.unlock()
        print("[RoadDetector] Rear camera stopped.")
    }

    // MARK: - Model loading

    /// Loads the bundled `RoadObjectDetector` Core ML model once.
    /// Prefers the Xcode-generated wrapper, then compiled `.mlmodelc` in the app bundle.
    private func loadModel() {
        do {
            let mlModel = try Self.makeMLModel()
            let vnModel = try VNCoreMLModel(for: mlModel)
            let request = VNCoreMLRequest(model: vnModel)
            request.imageCropAndScaleOption = .scaleFill

            visionModel = vnModel
            detectionRequest = request
            isModelReady = true
            state = .clear
            modelLoadMessage = "RoadObjectDetector loaded"
            print("[RoadDetector] Model loaded successfully.")
        } catch {
            visionModel = nil
            detectionRequest = nil
            isModelReady = false
            state = .modelUnavailable
            modelLoadMessage = error.localizedDescription
            print("[RoadDetector] Model load failed: \(error.localizedDescription)")
            Self.logBundleModelCandidates()
        }
    }

    nonisolated private static func makeMLModel() throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Xcode compiles DriveSensAI/ML/RoadObjectDetector.mlpackage → RoadObjectDetector.mlmodelc in the app bundle.
        // Load by URL only (avoid the generated MainActor wrapper from a nonisolated context).
        let resourceNames = ["RoadObjectDetector", "yolov8n", "YOLOv8n"]
        for name in resourceNames {
            if let modelURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: name, withExtension: "mlmodelc", subdirectory: "ML") {
                print("[RoadDetector] Loading compiled model at \(modelURL.path)")
                return try MLModel(contentsOf: modelURL, configuration: config)
            }
        }

        for name in resourceNames {
            if let packageURL = Bundle.main.url(forResource: name, withExtension: "mlpackage")
                ?? Bundle.main.url(forResource: name, withExtension: "mlpackage", subdirectory: "ML") {
                print("[RoadDetector] Compiling mlpackage at \(packageURL.path)")
                let compiledURL = try MLModel.compileModel(at: packageURL)
                return try MLModel(contentsOf: compiledURL, configuration: config)
            }

            if let mlmodelURL = Bundle.main.url(forResource: name, withExtension: "mlmodel")
                ?? Bundle.main.url(forResource: name, withExtension: "mlmodel", subdirectory: "ML") {
                print("[RoadDetector] Compiling mlmodel at \(mlmodelURL.path)")
                let compiledURL = try MLModel.compileModel(at: mlmodelURL)
                return try MLModel(contentsOf: compiledURL, configuration: config)
            }
        }

        if let all = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) {
            if let match = all.first(where: {
                let n = $0.deletingPathExtension().lastPathComponent.lowercased()
                return n.contains("road") || n.contains("yolo") || n.contains("object")
            }) {
                print("[RoadDetector] Loading discovered model at \(match.path)")
                return try MLModel(contentsOf: match, configuration: config)
            }
        }

        throw RoadDetectorError.modelNotFound
    }

    nonisolated private static func logBundleModelCandidates() {
        let mlmodelc = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? []
        let mlpackage = Bundle.main.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil) ?? []
        let mlmodel = Bundle.main.urls(forResourcesWithExtension: "mlmodel", subdirectory: nil) ?? []
        print("[RoadDetector] Bundle mlmodelc: \(mlmodelc.map(\.lastPathComponent))")
        print("[RoadDetector] Bundle mlpackage: \(mlpackage.map(\.lastPathComponent))")
        print("[RoadDetector] Bundle mlmodel: \(mlmodel.map(\.lastPathComponent))")
    }

    // MARK: - Frame intake

    private nonisolated func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        let now = CFAbsoluteTimeGetCurrent()

        processingLock.lock()
        if isProcessingFrame || (now - lastInferenceAt) < Self.minimumInferenceInterval {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        lastInferenceAt = now
        processingLock.unlock()

        analyze(pixelBuffer: pixelBuffer)
    }

    private nonisolated func analyze(pixelBuffer: CVPixelBuffer) {
        defer {
            processingLock.lock()
            isProcessingFrame = false
            processingLock.unlock()
        }

        guard let request = detectionRequest else {
            Task { @MainActor in
                self.state = .modelUnavailable
                self.detections = []
            }
            return
        }

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: Self.visionOrientation,
            options: [:]
        )

        do {
            try handler.perform([request])
            let observations = (request.results as? [VNRecognizedObjectObservation]) ?? []
            let filtered = Self.parseDetections(from: observations)

            Task { @MainActor in
                self.publish(detections: filtered)
            }
        } catch {
            print("[RoadDetector] Vision inference failed: \(error.localizedDescription)")
            Task { @MainActor in
                if self.detections.isEmpty {
                    self.state = .clear
                }
            }
        }
    }

    nonisolated private static func parseDetections(
        from observations: [VNRecognizedObjectObservation]
    ) -> [RoadDetection] {
        var results: [RoadDetection] = []

        for observation in observations {
            guard let top = observation.labels.first else { continue }
            let label = top.identifier.lowercased()
            let confidence = top.confidence

            guard relevantLabels.contains(label) else { continue }
            guard confidence >= confidenceThreshold else { continue }

            results.append(
                RoadDetection(
                    label: label,
                    confidence: confidence,
                    boundingBox: observation.boundingBox
                )
            )
        }

        results.sort { $0.confidence > $1.confidence }
        return results
    }

    private func publish(detections: [RoadDetection]) {
        self.detections = detections

        if !isModelReady {
            state = .modelUnavailable
            return
        }

        if detections.isEmpty {
            state = .clear
        } else {
            state = .objectsDetected
            for detection in detections.prefix(5) {
                print(String(format: "[RoadDetector] %@ %.2f", detection.label, detection.confidence))
            }
            print("[RoadDetector] \(detections.count) relevant objects")
        }
    }
}

private enum RoadDetectorError: LocalizedError {
    case modelNotFound

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "RoadObjectDetector model was not found in the app bundle. Add DriveSensAI/ML/RoadObjectDetector.mlpackage to the target and rebuild."
        }
    }
}
