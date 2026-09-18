//
//  DriveView.swift
//  DriveSensAI
//

import Combine
import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @StateObject private var roadDetector = RoadDetectionService()
    @StateObject private var roadRiskAnalyzer = RoadRiskAnalyzer()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 16) {
                topBar

                // Future MapKit swap point — keep this region intact.
                NavigationPlaceholderView()
                    .frame(maxHeight: .infinity)
                    .layoutPriority(1)

                if let banner = warningBanner {
                    WarningBannerView(title: banner.title, style: banner.style)
                        .transition(.opacity)
                }

                speedInstrument

                bottomStatusPanel

                onDeviceFooter
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.2), value: driverMonitor.attentionState)
        .animation(.easeInOut(duration: 0.2), value: roadRiskAnalyzer.state)
        .onAppear {
            startMultiCamIfNeeded()
        }
        .onDisappear {
            stopMultiCam()
        }
        .onReceive(roadDetector.$detections) { detections in
            roadRiskAnalyzer.update(
                detections: detections,
                timestamp: ProcessInfo.processInfo.systemUptime
            )
        }
        .onChange(of: roadDetector.isModelReady) { _, ready in
            if !ready {
                roadRiskAnalyzer.reset()
            }
        }
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(alignment: .center) {
            Text("DriveSensAI")
                .font(.headline.weight(.semibold))
                .foregroundStyle(.primary)

            Spacer()

            liveIndicator
        }
    }

    private var liveIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(multiCamOwner.isActive ? Color.green : Color.orange)
                .frame(width: 8, height: 8)

            Text(multiCamOwner.isActive ? "LIVE" : (multiCamOwner.errorMessage == nil ? "STARTING" : "CAMERA ERROR"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(multiCamOwner.isActive ? Color.green : Color.orange)
        }
        .accessibilityLabel(multiCamOwner.isActive ? "Live" : "Camera error")
    }

    // MARK: - Speed (placeholder — no Core Location yet)

    private var speedInstrument: some View {
        VStack(spacing: 4) {
            Text("--")
                .font(.system(size: 72, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.primary)

            Text("MPH")
                .font(.subheadline.weight(.semibold))
                .tracking(1.2)
                .foregroundStyle(.secondary)

            Text("SPEED LIMIT --")
                .font(.caption.weight(.medium))
                .foregroundStyle(.tertiary)
                .padding(.top, 6)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Speed unavailable. Speed limit unavailable.")
    }

    // MARK: - Bottom status

    private var bottomStatusPanel: some View {
        HStack(spacing: 12) {
            StatusItemView(
                title: "DRIVER",
                value: driverDisplayText,
                tone: driverTone
            )

            StatusItemView(
                title: "ROAD",
                value: roadDisplayText,
                tone: roadTone
            )
        }
    }

    private var onDeviceFooter: some View {
        HStack(spacing: 6) {
            Image(systemName: "lock.fill")
                .font(.caption2)
            Text("On-device AI")
                .font(.caption.weight(.medium))
        }
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity)
        .padding(.top, 4)
    }

    // MARK: - Display mapping (UI only)

    private var driverDisplayText: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "Attentive"
        case .lookingAway:
            return "Watch road"
        case .noFace:
            return "Driver not detected"
        }
    }

    private var driverTone: StatusItemView.Tone {
        switch driverMonitor.attentionState {
        case .attentive:
            return .normal
        case .lookingAway:
            return .urgent
        case .noFace:
            return .caution
        }
    }

    private var roadDisplayText: String {
        if !roadDetector.isModelReady || roadDetector.state == .modelUnavailable {
            return "Road monitoring unavailable"
        }

        // Experimental forward closing-risk estimation (not validated FCW).
        switch roadRiskAnalyzer.state {
        case .clear:
            return "Clear"
        case .monitoring:
            return "Vehicle ahead"
        case .caution:
            return "Closing vehicle"
        case .high:
            return "Rapid closing"
        }
    }

    private var roadTone: StatusItemView.Tone {
        if !roadDetector.isModelReady || roadDetector.state == .modelUnavailable {
            return .caution
        }
        switch roadRiskAnalyzer.state {
        case .clear, .monitoring:
            return .normal
        case .caution:
            return .caution
        case .high:
            return .urgent
        }
    }

    /// Priority: HIGH road risk > looking away > no face.
    private var warningBanner: (title: String, style: WarningBannerView.Style)? {
        if roadDetector.isModelReady,
           roadRiskAnalyzer.state == .high {
            return ("VEHICLE CLOSING", .critical)
        }

        switch driverMonitor.attentionState {
        case .lookingAway:
            return ("WATCH THE ROAD", .urgent)
        case .noFace:
            return ("Driver not detected", .caution)
        case .attentive:
            return nil
        }
    }

    // MARK: - MultiCam lifecycle (unchanged behavior)

    private func startMultiCamIfNeeded() {
        guard !multiCamOwner.hasStarted else { return }
        multiCamOwner.hasStarted = true
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
                multiCamOwner.errorMessage = nil
            case .failure(let error):
                multiCamOwner.isActive = false
                multiCamOwner.errorMessage = error.localizedDescription
                driverMonitor.endExternalFrameProcessing()
                roadDetector.endExternalFrameProcessing()
                roadRiskAnalyzer.reset()
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
        roadRiskAnalyzer.reset()

        multiCamOwner.manager.stop {
            multiCamOwner.isActive = false
            multiCamOwner.hasStarted = false
        }
    }
}

/// Holds the shared MultiCamManager and publishes DriveView status.
@MainActor
final class MultiCamSessionOwner: ObservableObject {
    let manager = MultiCamManager()

    @Published var isActive = false
    @Published var errorMessage: String?
    var hasStarted = false
}

#Preview {
    DriveView()
}
