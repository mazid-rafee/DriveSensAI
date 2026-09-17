//
//  DriveView.swift
//  DriveSensAI
//

import Combine
import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @StateObject private var roadDetector = RoadDetectionService()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    var body: some View {
        VStack(spacing: 20) {
            Text("DriveSensAI")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(driverStatusTitle)
                .font(.system(size: 36, weight: .bold, design: .rounded))
                .multilineTextAlignment(.center)
                .foregroundStyle(driverStatusColor)
                .frame(maxWidth: .infinity)
                .padding(.top, 12)

            Text(multiCamOwner.statusLine)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(multiCamOwner.isActive ? .green : .secondary)

            if let error = multiCamOwner.errorMessage {
                Text(error)
                    .font(.subheadline)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Divider()

            roadSection

            Spacer()
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .onAppear {
            startMultiCamIfNeeded()
        }
        .onDisappear {
            stopMultiCam()
        }
    }

    // MARK: - Road UI

    @ViewBuilder
    private var roadSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("ROAD")
                .font(.title3.weight(.bold))

            if !roadDetector.isModelReady || roadDetector.state == .modelUnavailable {
                Text("ROAD MODEL UNAVAILABLE")
                    .font(.body.weight(.medium))
                    .foregroundStyle(.orange)
            } else if roadDetector.detections.isEmpty {
                Text("ROAD CLEAR")
                    .font(.body.weight(.medium))
                    .foregroundStyle(.secondary)
            } else {
                Text("Objects: \(roadDetector.detections.count)")
                    .font(.body.weight(.semibold))

                ForEach(roadDetector.detections.prefix(3)) { detection in
                    HStack {
                        Text(detection.label.uppercased())
                            .font(.body.weight(.semibold))
                        Spacer()
                        Text("\(Int((detection.confidence * 100).rounded()))%")
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var driverStatusTitle: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "ATTENTIVE"
        case .lookingAway:
            return "LOOKING AWAY"
        case .noFace:
            return "NO FACE"
        }
    }

    private var driverStatusColor: Color {
        switch driverMonitor.attentionState {
        case .attentive:
            return .green
        case .lookingAway:
            return .orange
        case .noFace:
            return .red
        }
    }

    // MARK: - MultiCam lifecycle

    private func startMultiCamIfNeeded() {
        guard !multiCamOwner.hasStarted else { return }
        multiCamOwner.hasStarted = true
        multiCamOwner.statusLine = "MULTICAM STARTING…"
        multiCamOwner.errorMessage = nil

        // External processing only — do NOT call legacy CameraManager start APIs.
        driverMonitor.beginExternalFrameProcessing()
        roadDetector.beginExternalFrameProcessing()

        let manager = multiCamOwner.manager
        manager.onFrontFrame = { [weak driverMonitor] pixelBuffer in
            driverMonitor?.processExternalFrame(pixelBuffer)
        }
        manager.onRearFrame = { [weak roadDetector] pixelBuffer in
            roadDetector?.processExternalFrame(pixelBuffer)
        }

        manager.requestAccessAndStart { result in
            switch result {
            case .success:
                multiCamOwner.isActive = true
                multiCamOwner.statusLine = "Front + Rear: ACTIVE"
                multiCamOwner.errorMessage = nil
            case .failure(let error):
                multiCamOwner.isActive = false
                multiCamOwner.statusLine = "MULTICAM UNAVAILABLE"
                multiCamOwner.errorMessage = error.localizedDescription
                driverMonitor.endExternalFrameProcessing()
                roadDetector.endExternalFrameProcessing()
                // Allow a later onAppear retry after a failed start.
                multiCamOwner.hasStarted = false
            }
        }
    }

    private func stopMultiCam() {
        guard multiCamOwner.hasStarted || multiCamOwner.isActive else { return }

        multiCamOwner.manager.onFrontFrame = nil
        multiCamOwner.manager.onRearFrame = nil
        driverMonitor.endExternalFrameProcessing()
        roadDetector.endExternalFrameProcessing()

        multiCamOwner.manager.stop {
            multiCamOwner.isActive = false
            multiCamOwner.statusLine = "MULTICAM STOPPED"
            multiCamOwner.hasStarted = false
        }
    }
}

/// Holds the shared MultiCamManager and publishes DriveView status lines.
@MainActor
final class MultiCamSessionOwner: ObservableObject {
    let manager = MultiCamManager()

    @Published var isActive = false
    @Published var statusLine = "MULTICAM IDLE"
    @Published var errorMessage: String?
    var hasStarted = false
}

#Preview {
    DriveView()
}
