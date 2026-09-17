//
//  DriveView.swift
//  DriveSensAI
//

import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @State private var showRoadDebug = false
    @State private var isHandingOffToRoad = false

    var body: some View {
        VStack(spacing: 24) {
            Text("DriveSensAI")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(statusTitle)
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .multilineTextAlignment(.center)
                .foregroundStyle(statusColor)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 48)

            if let error = driverMonitor.cameraError {
                Text(error.localizedDescription)
                    .font(.subheadline)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            } else if !driverMonitor.isRunning && !isHandingOffToRoad && !showRoadDebug {
                Text("Starting camera…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // Temporary Milestone 2 entry point — present only after front session has stopped.
            Button("Test Road Detection") {
                guard !isHandingOffToRoad, !showRoadDebug else { return }
                isHandingOffToRoad = true
                driverMonitor.stop {
                    showRoadDebug = true
                    isHandingOffToRoad = false
                }
            }
            .buttonStyle(.bordered)
            .disabled(isHandingOffToRoad || showRoadDebug)
            .padding(.bottom, 8)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .onAppear {
            // Restore front camera when returning from road debug (or first launch).
            if !showRoadDebug && !isHandingOffToRoad {
                driverMonitor.start()
            }
        }
        .onDisappear {
            // Leaving DriveView entirely (e.g. app background / teardown).
            if !showRoadDebug {
                driverMonitor.stop()
            }
        }
        .fullScreenCover(isPresented: $showRoadDebug, onDismiss: {
            // Rear session is already stopped by RoadDebugView before dismiss.
            driverMonitor.start()
        }) {
            RoadDebugView()
        }
    }

    private var statusTitle: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "ATTENTIVE"
        case .lookingAway:
            return "LOOKING AWAY"
        case .noFace:
            return "NO FACE"
        }
    }

    private var statusColor: Color {
        switch driverMonitor.attentionState {
        case .attentive:
            return .green
        case .lookingAway:
            return .orange
        case .noFace:
            return .red
        }
    }
}

#Preview {
    DriveView()
}
