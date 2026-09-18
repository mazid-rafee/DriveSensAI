//
//  GoogleRoutesModels.swift
//  DriveSensAI
//

import Foundation

struct RoutesComputeRequest: Encodable {
    struct LatLng: Encodable {
        var latitude: Double
        var longitude: Double
    }

    struct Location: Encodable {
        var latLng: LatLng
    }

    struct Waypoint: Encodable {
        var location: Location
    }

    var origin: Waypoint
    var destination: Waypoint
    var travelMode: String
    var routingPreference: String
    var computeAlternativeRoutes: Bool
    var polylineQuality: String
    var languageCode: String
    var units: String
}

struct RoutesComputeResponse: Decodable {
    struct Polyline: Decodable {
        var encodedPolyline: String?
    }

    struct Route: Decodable {
        var duration: String?
        var distanceMeters: Int?
        var polyline: Polyline?
    }

    var routes: [Route]?
}

struct RoutesAPIErrorResponse: Decodable {
    struct Status: Decodable {
        var code: Int?
        var message: String?
        var status: String?
    }

    var error: Status?
}
