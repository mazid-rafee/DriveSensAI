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
    @StateObject private var laneDetector = LaneDetectionService()
    @StateObject private var alertManager = ADASAlertManager()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    /// Bridged from Google Navigation overspeed callbacks (unavailable outside guidance).
    @State private var speedingState: SpeedingState = .unavailable

    #if DEBUG
    @State private var showLaneDebugOverlay = false
    #endif

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
        .animation(.easeInOut(duration: 0.2), value: laneDetector.result.state)
        .animation(.easeInOut(duration: 0.2), value: speedMonitor.postedSpeedLimitMPH)
        .animation(.easeInOut(duration: 0.2), value: speedMonitor.speedMPH)
        .animation(.easeInOut(duration: 0.2), value: warningBanner?.title)
        .onAppear {
            startMultiCamIfNeeded()
            speedMonitor.start()
            alertManager.start()
            syncLaneSpeedGate()
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
        .onChange(of: laneDetector.result.state) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: speedMonitor.speedMPH) { _, _ in
            syncLaneSpeedGate()
        }
        .onChange(of: speedMonitor.hasReliableSpeed) { _, _ in
            syncLaneSpeedGate()
        }
    }

    /// Feeds centralized alert manager from explicit state changes only.
    private func syncADASAlerts() {
        alertManager.update(
            driverAttention: driverMonitor.attentionState,
            roadRisk: roadRiskAnalyzer.state,
            laneAssist: laneDetector.result.state
        )
    }

    private func syncLaneSpeedGate() {
        laneDetector.updateSpeed(
            mph: speedMonitor.speedMPH,
            reliable: speedMonitor.hasReliableSpeed
        )
    }

    /// Compact strip under the map: optional warning → speed | lane | limit → ADAS chips → footer.
    private var lowerTelemetry: some View {
        VStack(spacing: 6) {
            if let banner = warningBanner {
                WarningBannerView(title: banner.title, style: banner.style)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            instrumentClusterRow

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

    // MARK: - Instrument row: speed | lane | limit (compact, centered)

    private let laneRoadHeight: CGFloat = 46

    private var instrumentClusterRow: some View {
        HStack(alignment: .center, spacing: 16) {
            speedColumn
            laneAssistColumn
                .frame(width: 64)
            limitColumn
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(speedAccessibilityLabel)
    }

    private var speedColumn: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(speedDisplayText)
                .font(.system(size: 28, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(speedForeground)
            Text("MPH")
                .font(.system(size: 10, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(.secondary)
        }
    }

    private var limitColumn: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(speedLimitDisplayText)
                .font(.system(size: 28, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.primary.opacity(0.92))
            Text("LIMIT")
                .font(.system(size: 10, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var laneAssistColumn: some View {
        #if DEBUG
        LaneAssistView(
            result: laneDetector.result,
            roadHeight: laneRoadHeight,
            debugSnapshot: laneDetector.debugSnapshot,
            showDebugOverlay: showLaneDebugOverlay
        )
        .onLongPressGesture(minimumDuration: 0.6) {
            showLaneDebugOverlay.toggle()
        }
        #else
        LaneAssistView(result: laneDetector.result, roadHeight: laneRoadHeight)
        #endif
    }

    private var speedDisplayText: String {
        guard speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH else {
            return "--"
        }
        return "\(Int(mph.rounded()))"
    }

    private var speedLimitDisplayText: String {
        guard let limit = speedMonitor.postedSpeedLimitMPH, limit > 0 else {
            return "--"
        }
        return "\(limit)"
    }

    /// Current speed only: deeper red as MPH exceeds the posted Google limit.
    private var speedForeground: Color {
        guard speedMonitor.hasReliableSpeed,
              let mph = speedMonitor.speedMPH,
              let limit = speedMonitor.postedSpeedLimitMPH,
              limit > 0 else {
            return .primary
        }
        let over = mph - Double(limit)
        guard over > 0 else { return .primary }

        // 0 mph over → soft red; ≥20 mph over → deep crimson.
        let t = min(1.0, over / 20.0)
        return Color(
            red: 0.92 - 0.22 * t,
            green: 0.28 - 0.24 * t,
            blue: 0.22 - 0.16 * t
        )
    }

    private var speedAccessibilityLabel: String {
        let speedPart: String
        if speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH {
            speedPart = "\(Int(mph.rounded())) miles per hour"
        } else {
            speedPart = "Speed unavailable"
        }
        if let limit = speedMonitor.postedSpeedLimitMPH, limit > 0 {
            return "\(speedPart), limit \(limit)"
        }
        return "\(speedPart), limit unavailable"
    }

    // MARK: - Expandable ADAS status row

    /// DRIVER + ROAD status chips.
    private var adasStatusRow: some View {
        HStack(spacing: 6) {
            ADASStatusItem(
                icon: "person.fill",
                title: "DRIVER",
                status: driverDisplayText,
                tone: driverTone
            )
            ADASStatusItem(
                icon: "car.fill",
                title: "ROAD",
                status: roadDisplayText,
                tone: roadTone
            )
        }
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

    /// Priority: HIGH road risk > lane drift (speed-gated) > looking away > major speeding > no face.
    private var warningBanner: (title: String, style: WarningBannerView.Style)? {
        if roadDetector.isModelReady,
           roadRiskAnalyzer.state == .high {
            return ("VEHICLE CLOSING", .critical)
        }

        switch laneDetector.result.state {
        case .driftingLeft:
            return ("LANE DRIFT LEFT", .urgent)
        case .driftingRight:
            return ("LANE DRIFT RIGHT", .urgent)
        case .unavailable, .tracking:
            break
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
        laneDetector.beginExternalFrameProcessing()

        let manager = multiCamOwner.manager
        manager.onFrontFrame = { [weak driverMonitor] pixelBuffer in
            driverMonitor?.processExternalFrame(pixelBuffer)
        }
        manager.onRearFrame = { [weak roadDetector, weak laneDetector] pixelBuffer in
            roadDetector?.processExternalFrame(pixelBuffer)
            laneDetector?.processExternalFrame(pixelBuffer)
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
                laneDetector.endExternalFrameProcessing()
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
        laneDetector.endExternalFrameProcessing()
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
