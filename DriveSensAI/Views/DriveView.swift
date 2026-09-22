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
    @StateObject private var speedMonitor = SpeedMonitor()
    @StateObject private var alertManager = ADASAlertManager()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    /// Bridged from Google Navigation overspeed callbacks (unavailable outside guidance).
    @State private var speedingState: SpeedingState = .unavailable

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 8) {
                topBar

                // Map fills remaining height; telemetry stays compact so the map stays dominant.
                NavigationView(speedingState: $speedingState)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                lowerTelemetry
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.2), value: driverMonitor.attentionState)
        .animation(.easeInOut(duration: 0.2), value: roadRiskAnalyzer.state)
        .animation(.easeInOut(duration: 0.2), value: speedingState)
        .animation(.easeInOut(duration: 0.2), value: warningBanner?.title)
        .onAppear {
            startMultiCamIfNeeded()
            speedMonitor.start()
            alertManager.start()
            syncADASAlerts()
        }
        .onDisappear {
            alertManager.stop()
            stopMultiCam()
            speedMonitor.stop()
        }
        .onReceive(roadDetector.$detections) { detections in
            roadRiskAnalyzer.update(
                detections: detections,
                timestamp: ProcessInfo.processInfo.systemUptime
            )
            syncADASAlerts()
        }
        .onChange(of: roadDetector.isModelReady) { _, ready in
            if !ready {
                roadRiskAnalyzer.reset()
            }
            syncADASAlerts()
        }
        .onChange(of: driverMonitor.attentionState) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: roadRiskAnalyzer.state) { _, _ in
            syncADASAlerts()
        }
    }

    /// Feeds centralized alert manager from explicit state changes only.
    private func syncADASAlerts() {
        alertManager.update(
            driverAttention: driverMonitor.attentionState,
            roadRisk: roadRiskAnalyzer.state
        )
    }

    /// Compact strip under the map: optional warning → centered speed → ADAS chips → footer.
    private var lowerTelemetry: some View {
        VStack(spacing: 6) {
            if let banner = warningBanner {
                WarningBannerView(title: banner.title, style: banner.style)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            speedInstrumentCluster

            adasStatusRow

            onDeviceFooter
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

    // MARK: - Centered speed instrument cluster (real GPS only)
    // Posted numeric speed limit is not exposed by Navigation SDK → Google’s native
    // map indicator remains the only speed-limit UI when guidance is active.

    private var speedInstrumentCluster: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(speedDisplayText)
                .font(.system(size: 36, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(speedForeground)

            Text("MPH")
                .font(.system(size: 13, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(speedAccessibilityLabel)
    }

    private var speedDisplayText: String {
        guard speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH else {
            return "--"
        }
        return "\(Int(mph.rounded()))"
    }

    private var speedForeground: Color {
        switch speedingState {
        case .major:
            return .red
        case .minor:
            return .orange
        case .unavailable, .normal:
            return .primary
        }
    }

    private var speedAccessibilityLabel: String {
        if speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH {
            return "\(Int(mph.rounded())) miles per hour"
        }
        return "Speed unavailable"
    }

    // MARK: - Expandable ADAS status row

    /// Currently shows DRIVER + ROAD. Additional chips (lane, side, …) can append here later.
    private var adasStatusRow: some View {
        HStack(spacing: 6) {
            ForEach(adasItems) { item in
                ADASStatusItem(
                    icon: item.icon,
                    title: item.title,
                    status: item.status,
                    tone: item.tone
                )
            }
        }
    }

    private struct ADASItemModel: Identifiable {
        let id: String
        let icon: String
        let title: String
        let status: String
        let tone: ADASStatusItem.Tone
    }

    private var adasItems: [ADASItemModel] {
        [
            ADASItemModel(
                id: "driver",
                icon: "person.fill",
                title: "DRIVER",
                status: driverDisplayText,
                tone: driverTone
            ),
            ADASItemModel(
                id: "road",
                icon: "car.fill",
                title: "ROAD",
                status: roadDisplayText,
                tone: roadTone
            )
        ]
    }

    private var onDeviceFooter: some View {
        HStack(spacing: 4) {
            Image(systemName: "lock.fill")
                .font(.caption2)
            Text("On-device AI")
                .font(.caption2.weight(.medium))
        }
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity)
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

    private var driverTone: ADASStatusItem.Tone {
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

    private var roadTone: ADASStatusItem.Tone {
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

    /// Priority: HIGH road risk > looking away > major speeding > no face.
    private var warningBanner: (title: String, style: WarningBannerView.Style)? {
        if roadDetector.isModelReady,
           roadRiskAnalyzer.state == .high {
            return ("VEHICLE CLOSING", .critical)
        }

        if driverMonitor.attentionState == .lookingAway {
            return ("WATCH THE ROAD", .urgent)
        }

        if speedingState == .major {
            return ("SLOW DOWN", .urgent)
        }

        if driverMonitor.attentionState == .noFace {
            return ("Driver not detected", .caution)
        }

        return nil
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
