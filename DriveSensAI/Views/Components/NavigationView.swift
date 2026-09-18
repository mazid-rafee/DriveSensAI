//
//  NavigationView.swift
//  DriveSensAI
//

import SwiftUI
import CoreLocation
import GoogleMaps
import GoogleNavigation
import UIKit

/// Navigation region backed by Google Maps / Navigation SDK.
struct NavigationView: View {
    @Environment(\.colorScheme) private var colorScheme

    @StateObject private var session = NavigationSessionModel()
    @StateObject private var placesService = PlacesAutocompleteService()
    @StateObject private var userLocation = UserLocationProvider()

    @State private var isFetchingPlaceDetails = false
    @State private var pendingCurrentLocationSelection = false

    @FocusState private var focusedField: DirectionsSearchField?

    private var showDropdown: Bool {
        focusedField != nil && session.mode != .navigation
    }

    private var showCurrentLocationOption: Bool {
        focusedField == .source
    }

    var body: some View {
        ZStack(alignment: .top) {
            GoogleMapView(
                source: session.selectedSource,
                destination: session.selectedDestination,
                previewRoute: session.previewRoute,
                mode: session.mode,
                colorScheme: colorScheme,
                navigationStartRequestID: session.navigationStartRequestID,
                navigationEndRequestID: session.navigationEndRequestID,
                debugSimulateAlongRoute: debugSimulateFlag,
                onTermsRejected: { session.handleTermsRejected() },
                onNavigationFailed: { session.handleNavigationStartupFailed($0) },
                onNavigationStarted: { session.handleNavigationStartupSucceeded() },
                onNavigationEnded: { session.handleNavigationEnded() },
                onArrived: { session.handleArrivedAtDestination() }
            )
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                    .allowsHitTesting(false)
            }
            .accessibilityLabel("Navigation map")
            .zIndex(0)

            if showDropdown {
                Color.clear
                    .contentShape(Rectangle())
                    .onTapGesture { dismissSearchUI() }
                    .zIndex(1)
            }

            if session.mode != .navigation {
                searchOverlay
                    .zIndex(2)
            }

            bottomControls
                .zIndex(3)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onChange(of: focusedField) { _, newValue in
            guard session.mode != .navigation else { return }
            guard let newValue else {
                placesService.clearResults()
                return
            }
            placesService.resetSession()
            let query = newValue == .source ? session.sourceText : session.destinationText
            if newValue == .source, query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                placesService.clearResults()
            } else {
                placesService.scheduleSearch(query: query, biasCoordinate: userLocation.coordinate)
            }
        }
        .onChange(of: userLocation.coordinate?.latitude) { _, _ in
            guard pendingCurrentLocationSelection else { return }
            if userLocation.hasValidCoordinate {
                pendingCurrentLocationSelection = false
                applyCurrentLocationSelection()
            }
        }
        .onAppear {
            userLocation.refreshCachedLocation()
        }
    }

    private var debugSimulateFlag: Bool {
        #if DEBUG
        session.debugSimulateAlongRoute
        #else
        false
        #endif
    }

    // MARK: - Overlays

    private var searchOverlay: some View {
        VStack(spacing: 8) {
            DirectionsSearchCard(
                sourceText: Binding(
                    get: { session.sourceText },
                    set: { session.sourceText = $0 }
                ),
                destinationText: Binding(
                    get: { session.destinationText },
                    set: { session.destinationText = $0 }
                ),
                focusedField: $focusedField,
                onSourceTextChange: { text in
                    handleSearchTextChange(text, for: .source)
                },
                onDestinationTextChange: { text in
                    handleSearchTextChange(text, for: .destination)
                },
                onClearSource: {
                    session.clearField(.source)
                    placesService.clearResults()
                    focusedField = .source
                },
                onClearDestination: {
                    session.clearField(.destination)
                    placesService.clearResults()
                    focusedField = .destination
                },
                onSwap: {
                    session.swapLocations()
                    placesService.clearResults()
                }
            )

            if showDropdown {
                PlacesAutocompleteDropdown(
                    field: focusedField ?? .destination,
                    predictions: placesService.predictions,
                    isLoading: placesService.isLoading || isFetchingPlaceDetails,
                    errorMessage: placesService.errorMessage,
                    showCurrentLocationOption: showCurrentLocationOption,
                    currentLocationAvailable: userLocation.hasValidCoordinate,
                    locationDeniedMessage: userLocation.locationDeniedMessage,
                    onSelectPrediction: { item in
                        Task { await selectPrediction(item) }
                    },
                    onSelectCurrentLocation: selectCurrentLocation
                )
            }

            if session.mode == .showRoute {
                routeStatusBanner
            }

            if session.isStartingNavigation && session.mode != .navigation {
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text("Starting navigation…")
                        .font(.subheadline.weight(.medium))
                    Spacer(minLength: 0)
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(Color(.systemBackground))
                        .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
                }
            }

            if let status = session.statusMessage, session.mode != .navigation, !session.isStartingNavigation {
                Text(status)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(Color(.systemBackground).opacity(0.95))
                    }
            }

            #if DEBUG
            if session.mode == .showRoute {
                Toggle("DEBUG: Simulate along route", isOn: $session.debugSimulateAlongRoute)
                    .font(.caption2)
                    .padding(.horizontal, 4)
                    .tint(.orange)
            }
            #endif
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
    }

    @ViewBuilder
    private var routeStatusBanner: some View {
        HStack(spacing: 10) {
            if session.routeState.isLoading {
                ProgressView()
                    .controlSize(.small)
                Text("Finding route…")
                    .font(.subheadline.weight(.medium))
            } else if let route = session.previewRoute, session.routeState.isReady {
                Image(systemName: "car.fill")
                    .foregroundStyle(.blue)
                Text(route.summaryText)
                    .font(.subheadline.weight(.semibold))
            } else if let failure = session.routeState.failureMessage {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(failure)
                    .font(.footnote)
                    .lineLimit(2)
                Spacer(minLength: 0)
                Button("Retry") { session.retryPreviewRoute() }
                    .font(.footnote.weight(.semibold))
            }
            Spacer(minLength: 0)
        }
        .foregroundStyle(.primary)
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
        }
    }

    private var bottomControls: some View {
        VStack {
            Spacer()
            HStack {
                if session.mode == .navigation {
                    endNavigationButton
                } else if session.isGOVisible {
                    goButton
                }
                Spacer()
            }
            .padding(.leading, 12)
            .padding(.bottom, 12)
            .safeAreaPadding(.bottom, 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(true)
    }

    private var goButton: some View {
        Button {
            if session.isGOEnabled {
                focusedField = nil
                placesService.clearResults()
                session.requestStartNavigation()
            } else if session.mode == .chooseLocation {
                session.statusMessage = "Choose a starting point and destination first."
            } else {
                session.requestStartNavigation()
            }
        } label: {
            Text("GO")
                .font(.system(size: 20, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 56, height: 56)
                .background(Color.green, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
                .opacity(session.isGOEnabled ? 1.0 : 0.45)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Start navigation")
        .accessibilityHint(
            session.isGOEnabled
                ? "Starts turn-by-turn guidance"
                : "Choose a starting point and destination first."
        )
    }

    private var endNavigationButton: some View {
        Button {
            session.requestEndNavigation()
        } label: {
            Text("End")
                .font(.headline.weight(.bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 18)
                .padding(.vertical, 12)
                .background(Color.red, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .shadow(color: .black.opacity(0.25), radius: 6, y: 3)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("End navigation")
    }

    // MARK: - Places selection

    private func handleSearchTextChange(_ text: String, for field: DirectionsSearchField) {
        session.handleTypedText(text, for: field)

        // Only query Places while this field is actively focused. Avoids restarting
        // autocomplete when selection code programmatically updates the text.
        guard focusedField == field else { return }

        placesService.scheduleSearch(
            query: text,
            biasCoordinate: userLocation.coordinate
        )
    }

    private func selectPrediction(_ item: PlacePredictionItem) async {
        guard let field = focusedField else { return }
        isFetchingPlaceDetails = true
        defer { isFetchingPlaceDetails = false }

        do {
            let place = try await placesService.fetchSelectedPlace(placeID: item.placeID)
            session.applySelection(place, to: field)
            dismissSearchUI()
        } catch {
            placesService.presentError(error.localizedDescription)
        }
    }

    private func selectCurrentLocation() {
        userLocation.clearDeniedMessage()
        switch userLocation.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            if userLocation.hasValidCoordinate {
                applyCurrentLocationSelection()
            } else {
                pendingCurrentLocationSelection = true
                userLocation.requestWhenInUseIfNeeded()
            }
        case .notDetermined:
            pendingCurrentLocationSelection = true
            userLocation.requestWhenInUseIfNeeded()
        case .denied, .restricted:
            userLocation.requestWhenInUseIfNeeded()
        @unknown default:
            break
        }
    }

    private func applyCurrentLocationSelection() {
        guard let coordinate = userLocation.coordinate,
              CLLocationCoordinate2DIsValid(coordinate) else { return }
        session.applySelection(.currentLocation(coordinate: coordinate), to: .source)
        dismissSearchUI()
    }

    private func dismissSearchUI() {
        focusedField = nil
        placesService.clearResults()
    }
}

// MARK: - Google Map + Navigation SDK

private struct GoogleMapView: UIViewRepresentable {
    var source: SelectedPlace?
    var destination: SelectedPlace?
    var previewRoute: ComputedRoute?
    var mode: NavigationMode
    var colorScheme: ColorScheme
    var navigationStartRequestID: UUID?
    var navigationEndRequestID: UUID?
    var debugSimulateAlongRoute: Bool

    var onTermsRejected: () -> Void
    var onNavigationFailed: (String) -> Void
    var onNavigationStarted: () -> Void
    var onNavigationEnded: () -> Void
    var onArrived: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> GMSMapView {
        let options = GMSMapViewOptions()
        if let coordinate = context.coordinator.lastKnownCoordinate {
            options.camera = GMSCameraPosition.camera(withTarget: coordinate, zoom: 16)
        } else {
            options.camera = GMSCameraPosition.camera(
                withLatitude: 25.7617,
                longitude: -80.1918,
                zoom: 13
            )
        }

        let mapView = GMSMapView(options: options)
        mapView.settings.myLocationButton = true
        mapView.settings.compassButton = true
        mapView.settings.zoomGestures = true
        mapView.settings.scrollGestures = true
        mapView.settings.rotateGestures = true
        mapView.settings.tiltGestures = true
        // Bottom inset matches GO button row so the location button shares the same baseline.
        mapView.padding = UIEdgeInsets(top: 150, left: 72, bottom: 12, right: 12)

        context.coordinator.attach(to: mapView)
        context.coordinator.syncCallbacks(from: self)
        context.coordinator.applyInterfaceStyle(colorScheme, navigationEnabled: false)
        context.coordinator.updateMarkers(source: source, destination: destination, mode: mode, previewRoute: previewRoute)
        context.coordinator.updatePreviewPolyline(previewRoute, mode: mode)
        return mapView
    }

    func updateUIView(_ mapView: GMSMapView, context: Context) {
        let coordinator = context.coordinator
        coordinator.syncCallbacks(from: self)
        coordinator.source = source
        coordinator.destination = destination
        coordinator.previewRoute = previewRoute
        coordinator.mode = mode
        coordinator.colorScheme = colorScheme
        coordinator.debugSimulateAlongRoute = debugSimulateAlongRoute
        coordinator.applyInterfaceStyle(colorScheme, navigationEnabled: mapView.isNavigationEnabled)
        coordinator.updateMarkers(source: source, destination: destination, mode: mode, previewRoute: previewRoute)
        coordinator.updatePreviewPolyline(previewRoute, mode: mode)
        coordinator.handleNavigationStartIfNeeded(navigationStartRequestID)
        coordinator.handleNavigationEndIfNeeded(navigationEndRequestID)
        coordinator.applyModePresentation(mode)
    }

    final class Coordinator: NSObject, CLLocationManagerDelegate, GMSNavigatorListener {
        var source: SelectedPlace?
        var destination: SelectedPlace?
        var previewRoute: ComputedRoute?
        var mode: NavigationMode = .chooseLocation
        var colorScheme: ColorScheme = .dark
        var debugSimulateAlongRoute = false

        var onTermsRejected: () -> Void = {}
        var onNavigationFailed: (String) -> Void = { _ in }
        var onNavigationStarted: () -> Void = {}
        var onNavigationEnded: () -> Void = {}
        var onArrived: () -> Void = {}

        private let locationManager = CLLocationManager()
        private weak var mapView: GMSMapView?
        private var didApplyLiveFix = false
        private var sourceMarker: GMSMarker?
        private var destinationMarker: GMSMarker?
        private var previewPolyline: GMSPolyline?
        private var lastFittedPairKey: String?
        private var lastPolylineKey: String?
        private var lastSourcePlaceID: String?
        private var lastDestinationPlaceID: String?
        private var handledStartID: UUID?
        private var handledEndID: UUID?
        private var isStarting = false
        private var didRegisterNavigatorListener = false

        var lastKnownCoordinate: CLLocationCoordinate2D? {
            locationManager.location?.coordinate
        }

        override init() {
            super.init()
            locationManager.delegate = self
            locationManager.desiredAccuracy = kCLLocationAccuracyBest
        }

        func syncCallbacks(from parent: GoogleMapView) {
            onTermsRejected = parent.onTermsRejected
            onNavigationFailed = parent.onNavigationFailed
            onNavigationStarted = parent.onNavigationStarted
            onNavigationEnded = parent.onNavigationEnded
            onArrived = parent.onArrived
        }

        func attach(to mapView: GMSMapView) {
            self.mapView = mapView
            requestLocationAccessIfNeeded()
        }

        func applyInterfaceStyle(_ colorScheme: ColorScheme, navigationEnabled: Bool) {
            guard let mapView else { return }
            // Follow the SwiftUI / app color scheme instead of Maps' light-mode default.
            mapView.overrideUserInterfaceStyle = (colorScheme == .dark) ? .dark : .light
            if navigationEnabled {
                mapView.lightingMode = (colorScheme == .dark) ? .lowLight : .normal
            }
        }

        func applyModePresentation(_ mode: NavigationMode) {
            guard let mapView else { return }
            if mode == .navigation {
                mapView.padding = UIEdgeInsets(top: 24, left: 12, bottom: 96, right: 12)
                previewPolyline?.map = nil
            } else {
                // Keep location button on the same horizontal line as the GO button.
                mapView.padding = UIEdgeInsets(top: 150, left: 72, bottom: 12, right: 12)
                if let previewPolyline {
                    previewPolyline.map = mapView
                }
            }
        }

        func updateMarkers(
            source: SelectedPlace?,
            destination: SelectedPlace?,
            mode: NavigationMode,
            previewRoute: ComputedRoute?
        ) {
            guard let mapView else { return }

            let sourceChanged = lastSourcePlaceID != source?.placeID
            let destinationChanged = lastDestinationPlaceID != destination?.placeID
            lastSourcePlaceID = source?.placeID
            lastDestinationPlaceID = destination?.placeID

            if let source {
                let marker = sourceMarker ?? GMSMarker()
                marker.position = source.coordinate
                marker.title = source.primaryDisplayName
                marker.snippet = source.formattedAddress
                marker.icon = GMSMarker.markerImage(with: .systemBlue)
                marker.map = mode == .navigation ? nil : mapView
                sourceMarker = marker
            } else {
                sourceMarker?.map = nil
                sourceMarker = nil
            }

            if let destination {
                let marker = destinationMarker ?? GMSMarker()
                marker.position = destination.coordinate
                marker.title = destination.primaryDisplayName
                marker.snippet = destination.formattedAddress
                marker.icon = GMSMarker.markerImage(with: .systemRed)
                marker.map = mode == .navigation ? nil : mapView
                destinationMarker = marker
            } else {
                destinationMarker?.map = nil
                destinationMarker = nil
            }

            if sourceChanged || destinationChanged {
                lastFittedPairKey = nil
            }

            if mode != .navigation, source != nil, destination != nil {
                fitCameraToRouteOrMarkers(
                    source: source,
                    destination: destination,
                    route: previewRoute,
                    force: sourceChanged || destinationChanged || previewRoute != nil
                )
            }
        }

        func updatePreviewPolyline(_ route: ComputedRoute?, mode: NavigationMode) {
            guard let mapView else { return }

            guard let route else {
                previewPolyline?.map = nil
                previewPolyline = nil
                lastPolylineKey = nil
                return
            }

            let key = "\(route.sourcePlaceID)|\(route.destinationPlaceID)|\(route.encodedPolyline)"
            let isNewRoute = key != lastPolylineKey
            if !isNewRoute, previewPolyline != nil {
                if mode != .navigation {
                    previewPolyline?.map = mapView
                }
                return
            }
            lastPolylineKey = key

            previewPolyline?.map = nil
            guard let path = GMSPath(fromEncodedPath: route.encodedPolyline), path.count() > 0 else {
                fitCameraToRouteOrMarkers(
                    source: source,
                    destination: destination,
                    route: nil,
                    force: true
                )
                return
            }

            let polyline = GMSPolyline(path: path)
            polyline.strokeWidth = 6
            polyline.strokeColor = UIColor.systemBlue
            polyline.geodesic = true
            polyline.zIndex = 50
            polyline.map = mode == .navigation ? nil : mapView
            previewPolyline = polyline

            lastFittedPairKey = nil
            fitCameraToRouteOrMarkers(
                source: source,
                destination: destination,
                route: route,
                force: true
            )
        }

        private func fitCameraToRouteOrMarkers(
            source: SelectedPlace?,
            destination: SelectedPlace?,
            route: ComputedRoute?,
            force: Bool
        ) {
            guard mode != .navigation, let mapView else { return }

            let edgePadding = UIEdgeInsets(top: 170, left: 56, bottom: 88, right: 56)

            if let route,
               let path = GMSPath(fromEncodedPath: route.encodedPolyline),
               path.count() > 1 {
                let pairKey = "route:\(route.sourcePlaceID)|\(route.destinationPlaceID)|\(route.encodedPolyline)"
                if !force, pairKey == lastFittedPairKey { return }
                lastFittedPairKey = pairKey

                let bounds = GMSCoordinateBounds(path: path)
                DispatchQueue.main.async { [weak mapView] in
                    guard let mapView else { return }
                    mapView.animate(with: GMSCameraUpdate.fit(bounds, with: edgePadding))
                }
                return
            }

            guard let source, let destination else {
                if source == nil || destination == nil {
                    lastFittedPairKey = nil
                }
                return
            }

            let pairKey = "markers:\(source.placeID)|\(destination.placeID)"
            if !force, pairKey == lastFittedPairKey { return }
            lastFittedPairKey = pairKey

            var bounds = GMSCoordinateBounds(
                coordinate: source.coordinate,
                coordinate: destination.coordinate
            )
            bounds = bounds.includingCoordinate(source.coordinate)
            bounds = bounds.includingCoordinate(destination.coordinate)

            DispatchQueue.main.async { [weak mapView] in
                guard let mapView else { return }
                mapView.animate(with: GMSCameraUpdate.fit(bounds, with: edgePadding))
            }
        }

        // MARK: Navigation start / end

        func handleNavigationStartIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledStartID, !isStarting else { return }
            handledStartID = requestID
            beginNavigationFlow()
        }

        func handleNavigationEndIfNeeded(_ requestID: UUID?) {
            guard let requestID, requestID != handledEndID else { return }
            handledEndID = requestID
            stopNavigation()
            notifyEnded()
        }

        private func beginNavigationFlow() {
            guard let mapView else {
                notifyFailed("Map is not ready.")
                return
            }
            isStarting = true

            let options = GMSNavigationTermsAndConditionsOptions(companyName: "DriveSensAI")
            GMSNavigationServices.showTermsAndConditionsDialogIfNeeded(with: options) { [weak self] accepted in
                Task { @MainActor in
                    guard let self else { return }
                    guard accepted else {
                        self.isStarting = false
                        self.notifyTermsRejected()
                        return
                    }
                    self.enableNavigatorAndSetDestinations(on: mapView)
                }
            }
        }

        private func enableNavigatorAndSetDestinations(on mapView: GMSMapView) {
            mapView.isNavigationEnabled = true
            mapView.travelMode = .driving
            mapView.settings.isRecenterButtonEnabled = true
            mapView.settings.compassButton = true
            applyInterfaceStyle(colorScheme, navigationEnabled: true)

            guard let navigator = mapView.navigator else {
                isStarting = false
                mapView.isNavigationEnabled = false
                notifyFailed("Couldn't create a navigator.")
                return
            }

            if !didRegisterNavigatorListener {
                navigator.add(self)
                didRegisterNavigatorListener = true
            }

            let waypoints: [GMSNavigationWaypoint]
            do {
                waypoints = try Self.makeWaypoints(
                    source: source,
                    destination: destination
                )
            } catch {
                isStarting = false
                mapView.isNavigationEnabled = false
                notifyFailed(error.localizedDescription)
                return
            }

            navigator.setDestinations(waypoints) { [weak self] routeStatus in
                Task { @MainActor in
                    self?.handleRouteStatus(routeStatus, mapView: mapView, navigator: navigator)
                }
            }
        }

        private func handleRouteStatus(
            _ routeStatus: GMSRouteStatus,
            mapView: GMSMapView,
            navigator: GMSNavigator
        ) {
            defer { isStarting = false }

            guard routeStatus == .OK else {
                mapView.isNavigationEnabled = false
                notifyFailed(Self.message(for: routeStatus))
                return
            }

            navigator.isGuidanceActive = true
            navigator.sendsBackgroundNotifications = true
            mapView.cameraMode = .following
            applyInterfaceStyle(colorScheme, navigationEnabled: true)
            previewPolyline?.map = nil

            #if DEBUG
            if debugSimulateAlongRoute {
                mapView.locationSimulator?.simulateLocationsAlongExistingRoute()
            }
            #endif

            notifyStarted()
        }

        private func stopNavigation() {
            guard let mapView else { return }
            #if DEBUG
            mapView.locationSimulator?.stopSimulation()
            #endif
            if let navigator = mapView.navigator {
                navigator.isGuidanceActive = false
                navigator.sendsBackgroundNotifications = false
                navigator.clearDestinations()
                if didRegisterNavigatorListener {
                    _ = navigator.remove(self)
                    didRegisterNavigatorListener = false
                }
            }
            mapView.isNavigationEnabled = false
            UIApplication.shared.isIdleTimerDisabled = false
            lastFittedPairKey = nil
            isStarting = false
            updatePreviewPolyline(previewRoute, mode: .showRoute)
            updateMarkers(
                source: source,
                destination: destination,
                mode: .showRoute,
                previewRoute: previewRoute
            )
        }

        private func notifyStarted() {
            let callback = onNavigationStarted
            Task { @MainActor in callback() }
        }

        private func notifyFailed(_ message: String) {
            let callback = onNavigationFailed
            Task { @MainActor in callback(message) }
        }

        private func notifyTermsRejected() {
            let callback = onTermsRejected
            Task { @MainActor in callback() }
        }

        private func notifyEnded() {
            let callback = onNavigationEnded
            Task { @MainActor in callback() }
        }

        private func notifyArrived() {
            let callback = onArrived
            Task { @MainActor in callback() }
        }

        static func makeWaypoints(
            source: SelectedPlace?,
            destination: SelectedPlace?
        ) throws -> [GMSNavigationWaypoint] {
            guard let destination else {
                throw NavigationStartupError.message("Choose a destination first.")
            }
            let destinationWaypoint = try waypoint(for: destination)

            guard let source else {
                return [destinationWaypoint]
            }

            if source.isCurrentLocation {
                return [destinationWaypoint]
            }

            let sourceWaypoint = try waypoint(for: source)
            return [sourceWaypoint, destinationWaypoint]
        }

        private static func waypoint(for place: SelectedPlace) throws -> GMSNavigationWaypoint {
            if !place.isCurrentLocation,
               !place.placeID.isEmpty,
               let byPlaceID = GMSNavigationWaypoint(
                placeID: place.placeID,
                title: place.primaryDisplayName
               ) {
                return byPlaceID
            }

            guard CLLocationCoordinate2DIsValid(place.coordinate),
                  let byCoordinate = GMSNavigationWaypoint(
                    location: place.coordinate,
                    title: place.primaryDisplayName
                  ) else {
                throw NavigationStartupError.message("Invalid waypoint for \(place.primaryDisplayName).")
            }
            return byCoordinate
        }

        private static func message(for status: GMSRouteStatus) -> String {
            switch status {
            case .OK:
                return "Route ready."
            case .noRouteFound:
                return "No navigation route found."
            case .networkError:
                return "Network error while starting navigation."
            case .quotaExceeded:
                return "Navigation quota exceeded."
            case .apiKeyNotAuthorized:
                return "API key is not authorized for Navigation SDK."
            case .canceled:
                return "Navigation route request was canceled."
            case .locationUnavailable:
                return "Current location is unavailable."
            case .waypointError:
                return "One of the waypoints is invalid."
            case .duplicateWaypointsError:
                return "Duplicate waypoints were provided."
            case .noWaypointsError:
                return "No waypoints were provided."
            case .travelModeUnsupported:
                return "Driving mode is unsupported for this route."
            case .internalError:
                return "Navigation internal error."
            @unknown default:
                return "Couldn't start navigation (status \(status.rawValue))."
            }
        }

        // MARK: Location

        private func requestLocationAccessIfNeeded() {
            switch locationManager.authorizationStatus {
            case .notDetermined:
                locationManager.requestWhenInUseAuthorization()
            case .authorizedAlways, .authorizedWhenInUse:
                beginTracking()
            case .denied, .restricted:
                mapView?.isMyLocationEnabled = false
            @unknown default:
                break
            }
        }

        private func beginTracking() {
            mapView?.isMyLocationEnabled = true
            if let location = locationManager.location {
                moveCamera(to: location, animated: false)
            }
            locationManager.startUpdatingLocation()
        }

        private func moveCamera(to location: CLLocation, animated: Bool) {
            guard mode != .navigation,
                  sourceMarker == nil,
                  destinationMarker == nil,
                  previewPolyline == nil else { return }

            let camera = GMSCameraPosition.camera(withTarget: location.coordinate, zoom: 16)
            if animated {
                mapView?.animate(to: camera)
            } else {
                mapView?.camera = camera
            }
        }

        func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
            switch manager.authorizationStatus {
            case .authorizedAlways, .authorizedWhenInUse:
                beginTracking()
            case .denied, .restricted:
                mapView?.isMyLocationEnabled = false
                manager.stopUpdatingLocation()
            default:
                break
            }
        }

        func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
            guard !didApplyLiveFix, let location = locations.last else { return }
            didApplyLiveFix = true
            moveCamera(to: location, animated: true)
            manager.stopUpdatingLocation()
        }

        func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {}

        // MARK: GMSNavigatorListener

        func navigator(_ navigator: GMSNavigator, didArriveAt waypoint: GMSNavigationWaypoint) {
            notifyArrived()
        }

        deinit {
            if let navigator = mapView?.navigator, didRegisterNavigatorListener {
                _ = navigator.remove(self)
            }
        }
    }
}

private enum NavigationStartupError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self {
        case .message(let text):
            return text
        }
    }
}

#Preview {
    NavigationView()
        .frame(height: 480)
        .padding()
        .preferredColorScheme(.dark)
        .background(Color.black)
}
