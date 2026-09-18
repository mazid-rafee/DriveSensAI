//
//  GoogleRoutesService.swift
//  DriveSensAI
//

import CoreLocation
import Foundation

enum GoogleRoutesError: LocalizedError {
    case missingAPIKey
    case invalidCoordinates
    case emptyRoutes
    case httpStatus(Int, String)
    case decoding(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "Routes API key is not configured."
        case .invalidCoordinates:
            return "Invalid start or destination coordinates."
        case .emptyRoutes:
            return "No driving route was returned."
        case .httpStatus(_, let message):
            return message
        case .decoding(let message):
            return message
        case .cancelled:
            return "Route request was cancelled."
        }
    }
}

/// Google Routes API v2 Compute Routes client.
actor GoogleRoutesService {
    private let session: URLSession
    private let endpoint = URL(string: "https://routes.googleapis.com/directions/v2:computeRoutes")!

    init(session: URLSession = .shared) {
        self.session = session
    }

    func computeDrivingRoute(
        from origin: CLLocationCoordinate2D,
        to destination: CLLocationCoordinate2D
    ) async throws -> (encodedPolyline: String, durationText: String, distanceMeters: Int) {
        guard CLLocationCoordinate2DIsValid(origin),
              CLLocationCoordinate2DIsValid(destination) else {
            throw GoogleRoutesError.invalidCoordinates
        }

        let apiKey = RoutesAPIConfiguration.apiKey
        guard !apiKey.isEmpty else {
            throw GoogleRoutesError.missingAPIKey
        }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "X-Goog-Api-Key")
        request.setValue(
            "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
            forHTTPHeaderField: "X-Goog-FieldMask"
        )
        if let bundleID = Bundle.main.bundleIdentifier {
            request.setValue(bundleID, forHTTPHeaderField: "X-Ios-Bundle-Identifier")
        }

        let body = RoutesComputeRequest(
            origin: .init(location: .init(latLng: .init(
                latitude: origin.latitude,
                longitude: origin.longitude
            ))),
            destination: .init(location: .init(latLng: .init(
                latitude: destination.latitude,
                longitude: destination.longitude
            ))),
            travelMode: "DRIVE",
            routingPreference: "TRAFFIC_AWARE",
            computeAlternativeRoutes: false,
            polylineQuality: "HIGH_QUALITY",
            languageCode: "en-US",
            units: "IMPERIAL"
        )
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw GoogleRoutesError.decoding("Invalid Routes API response.")
        }

        guard (200..<300).contains(http.statusCode) else {
            let message = Self.parseErrorMessage(from: data)
                ?? "Routes request failed (\(http.statusCode))."
            throw GoogleRoutesError.httpStatus(http.statusCode, message)
        }

        do {
            let decoded = try JSONDecoder().decode(RoutesComputeResponse.self, from: data)
            guard let route = decoded.routes?.first,
                  let encoded = route.polyline?.encodedPolyline,
                  !encoded.isEmpty,
                  let distance = route.distanceMeters else {
                throw GoogleRoutesError.emptyRoutes
            }
            let durationText = Self.displayDuration(from: route.duration)
            return (encoded, durationText, distance)
        } catch let error as GoogleRoutesError {
            throw error
        } catch {
            throw GoogleRoutesError.decoding("Couldn't decode the route response.")
        }
    }

    private static func parseErrorMessage(from data: Data) -> String? {
        guard let payload = try? JSONDecoder().decode(RoutesAPIErrorResponse.self, from: data) else {
            return nil
        }
        return payload.error?.message
    }

    /// Converts Routes API duration strings like "1234s" into a short display value.
    private static func displayDuration(from raw: String?) -> String {
        guard let raw else { return "--" }
        let digits = raw.trimmingCharacters(in: CharacterSet.decimalDigits.inverted)
        guard let seconds = Int(digits) else { return raw }
        let hours = seconds / 3600
        let minutes = (seconds % 3600 + 59) / 60
        if hours > 0 {
            return "\(hours) hr \(minutes) min"
        }
        return "\(max(minutes, 1)) min"
    }
}

enum RoutesAPIConfiguration {
    private static let placeholder = "REPLACE_WITH_ROUTES_API_KEY"

    /// Reads the Routes API key from Info.plist (`RoutesAPIKey`), populated via Secrets.xcconfig.
    static var apiKey: String {
        let candidates = [
            Bundle.main.object(forInfoDictionaryKey: "RoutesAPIKey") as? String,
            Bundle.main.object(forInfoDictionaryKey: "ROUTES_API_KEY") as? String
        ]
        for raw in candidates {
            let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !trimmed.isEmpty, trimmed != placeholder else { continue }
            return trimmed
        }
        return ""
    }
}
