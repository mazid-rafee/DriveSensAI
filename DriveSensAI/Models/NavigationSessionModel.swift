//
//  NavigationSessionModel.swift
//  DriveSensAI
//

import Combine
import CoreLocation
import Foundation
import UIKit

/// Owns navigation mode, selected places, preview route state, and GO / End actions.
@MainActor
final class NavigationSessionModel: ObservableObject {
    @Published private(set) var mode: NavigationMode = .chooseLocation
    @Published var sourceText = ""
    @Published var destinationText = ""
    @Published private(set) var selectedSource: SelectedPlace?
    @Published private(set) var selectedDestination: SelectedPlace?

    @Published private(set) var routeState: RouteLoadingState = .idle
    @Published private(set) var previewRoute: ComputedRoute?

    @Published private(set) var isStartingNavigation = false
    @Published var statusMessage: String?
    @Published private(set) var navigationStartRequestID: UUID?
    @Published private(set) var navigationEndRequestID: UUID?

    private let routesService = GoogleRoutesService()
    private var routeTask: Task<Void, Never>?
    private var routeGeneration = 0
    private var startupTimeoutTask: Task<Void, Never>?

    var hasValidSource: Bool {
        guard let selectedSource else { return false }
        return CLLocationCoordinate2DIsValid(selectedSource.coordinate)
    }

    var hasValidDestination: Bool {
        guard let selectedDestination else { return false }
        return CLLocationCoordinate2DIsValid(selectedDestination.coordinate)
    }

    var hasValidPair: Bool {
        hasValidSource && hasValidDestination
    }

    var isGOVisible: Bool {
        // Hide during active guidance and while startup is in progress.
        mode != .navigation && !isStartingNavigation
    }

    var isGOEnabled: Bool {
        mode == .showRoute && routeState.isReady && !isStartingNavigation
    }

    // MARK: - Selection updates

    func applySelection(_ place: SelectedPlace, to field: DirectionsSearchField) {
        endActiveGuidanceIfNeeded()
        switch field {
        case .source:
            selectedSource = place
            sourceText = place.primaryDisplayName
        case .destination:
            selectedDestination = place
            destinationText = place.primaryDisplayName
        }
        reconcileModeAfterSelectionChange()
    }

    func clearField(_ field: DirectionsSearchField) {
        endActiveGuidanceIfNeeded()
        switch field {
        case .source:
            selectedSource = nil
            sourceText = ""
        case .destination:
            selectedDestination = nil
            destinationText = ""
        }
        reconcileModeAfterSelectionChange()
    }

    func handleTypedText(_ text: String, for field: DirectionsSearchField) {
        switch field {
        case .source:
            sourceText = text
            if let selectedSource, text != selectedSource.primaryDisplayName {
                endActiveGuidanceIfNeeded()
                self.selectedSource = nil
                reconcileModeAfterSelectionChange()
            }
        case .destination:
            destinationText = text
            if let selectedDestination, text != selectedDestination.primaryDisplayName {
                endActiveGuidanceIfNeeded()
                self.selectedDestination = nil
                reconcileModeAfterSelectionChange()
            }
        }
    }

    func swapLocations() {
        endActiveGuidanceIfNeeded()
        let previousSource = selectedSource
        let previousDestination = selectedDestination
        let previousSourceText = sourceText
        let previousDestinationText = destinationText

        selectedSource = previousDestination
        selectedDestination = previousSource
        sourceText = previousDestinationText
        destinationText = previousSourceText
        reconcileModeAfterSelectionChange()
    }

    // MARK: - Mode / route

    private func reconcileModeAfterSelectionChange() {
        if mode == .navigation {
            return
        }

        if hasValidPair {
            mode = .showRoute
            statusMessage = nil
            requestPreviewRoute()
        } else {
            mode = .chooseLocation
            clearPreviewRoute()
        }
    }

    private func requestPreviewRoute() {
        guard let source = selectedSource, let destination = selectedDestination else {
            clearPreviewRoute()
            return
        }

        routeTask?.cancel()
        routeGeneration += 1
        let generation = routeGeneration
        routeState = .loading
        previewRoute = nil
        statusMessage = nil

        let origin = source.coordinate
        let dest = destination.coordinate
        let sourceID = source.placeID
        let destinationID = destination.placeID

        routeTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.routesService.computeDrivingRoute(
                    from: origin,
                    to: dest
                )
                guard !Task.isCancelled, generation == self.routeGeneration else { return }
                let route = ComputedRoute(
                    encodedPolyline: result.encodedPolyline,
                    durationText: result.durationText,
                    distanceMeters: result.distanceMeters,
                    sourcePlaceID: sourceID,
                    destinationPlaceID: destinationID
                )
                self.previewRoute = route
                self.routeState = .ready
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled, generation == self.routeGeneration else { return }
                self.previewRoute = nil
                self.routeState = .failed(error.localizedDescription)
                self.statusMessage = error.localizedDescription
            }
        }
    }

    func retryPreviewRoute() {
        guard hasValidPair, mode == .showRoute else { return }
        requestPreviewRoute()
    }

    private func clearPreviewRoute() {
        routeTask?.cancel()
        routeTask = nil
        routeGeneration += 1
        previewRoute = nil
        routeState = .idle
    }

    // MARK: - GO / navigation lifecycle

    func requestStartNavigation() {
        guard mode != .navigation else { return }

        guard hasValidPair else {
            statusMessage = "Choose a starting point and destination first."
            return
        }

        guard routeState.isReady else {
            if routeState.isLoading {
                statusMessage = "Wait for the route to finish loading."
            } else if case .failed = routeState {
                statusMessage = "Fix the route error before starting navigation."
            } else {
                statusMessage = "Choose a starting point and destination first."
            }
            return
        }

        guard !isStartingNavigation else { return }

        isStartingNavigation = true
        statusMessage = nil
        navigationStartRequestID = UUID()
        beginStartupTimeout()
    }

    private func beginStartupTimeout() {
        startupTimeoutTask?.cancel()
        startupTimeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 25_000_000_000)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                guard let self, self.isStartingNavigation, self.mode != .navigation else { return }
                self.handleNavigationStartupFailed(
                    "Navigation is taking too long. Please try again."
                )
            }
        }
    }

    private func cancelStartupTimeout() {
        startupTimeoutTask?.cancel()
        startupTimeoutTask = nil
    }

    func handleTermsRejected() {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .showRoute
        statusMessage = "Accept Google Navigation terms to start guidance."
    }

    func handleNavigationStartupFailed(_ message: String) {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .showRoute
        statusMessage = message
    }

    func handleNavigationStartupSucceeded() {
        print("[NAV 4] Model changing to navigation mode")
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        mode = .navigation
        statusMessage = nil
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func requestEndNavigation() {
        guard mode == .navigation || isStartingNavigation else { return }
        navigationEndRequestID = UUID()
    }

    func handleNavigationEnded() {
        cancelStartupTimeout()
        isStartingNavigation = false
        navigationStartRequestID = nil
        navigationEndRequestID = nil
        UIApplication.shared.isIdleTimerDisabled = false
        mode = .showRoute
        if hasValidPair {
            if previewRoute == nil || !routeState.isReady {
                requestPreviewRoute()
            }
        } else {
            mode = .chooseLocation
            clearPreviewRoute()
        }
    }

    func handleArrivedAtDestination() {
        requestEndNavigation()
    }

    private func endActiveGuidanceIfNeeded() {
        guard mode == .navigation || isStartingNavigation else { return }
        requestEndNavigation()
    }
}
