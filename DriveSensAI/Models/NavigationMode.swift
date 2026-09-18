//
//  NavigationMode.swift
//  DriveSensAI
//

import Foundation

/// Conceptual modes: CHOOSE LOCATION → SHOW ROUTE → NAVIGATION.
enum NavigationMode: Equatable {
    case chooseLocation
    case showRoute
    case navigation
}

enum RouteLoadingState: Equatable {
    case idle
    case loading
    case ready
    case failed(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var failureMessage: String? {
        if case .failed(let message) = self { return message }
        return nil
    }
}

struct ComputedRoute: Equatable {
    var encodedPolyline: String
    var durationText: String
    var distanceMeters: Int
    var sourcePlaceID: String
    var destinationPlaceID: String

    var distanceMilesText: String {
        let miles = Double(distanceMeters) / 1609.344
        return String(format: "%.1f mi", miles)
    }

    var summaryText: String {
        "\(durationText) · \(distanceMilesText)"
    }
}
