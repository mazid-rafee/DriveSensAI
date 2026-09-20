//
//  MockRouteRiskScorer.swift
//  DriveSensAI
//

import Foundation
import UIKit

/// Relative safety ranking among currently returned mock-scored routes.
/// These tiers are comparative only — not absolute safety claims.
enum MockRouteSafetyTier: String, Equatable, Sendable {
    case safest
    case medium
    case unsafest
}

/// Centralized temporary styling for mock safety-colored route polylines.
/// Reuse later when Python model scores replace mock values.
enum MockRouteSafetyStyle {
    static let safestColor = UIColor(red: 52 / 255, green: 168 / 255, blue: 83 / 255, alpha: 1)   // #34A853
    static let mediumColor = UIColor(red: 251 / 255, green: 188 / 255, blue: 4 / 255, alpha: 1)    // #FBBC04
    static let unsafestColor = UIColor(red: 234 / 255, green: 67 / 255, blue: 53 / 255, alpha: 1)  // #EA4335

    static let selectedOpacity: CGFloat = 1.0
    static let unselectedOpacity: CGFloat = 0.35

    static let selectedStrokeWidth: CGFloat = 7
    static let unselectedStrokeWidth: CGFloat = 5

    static let selectedZIndex: Int32 = 60
    static let unselectedZIndex: Int32 = 40

    static func baseColor(for tier: MockRouteSafetyTier) -> UIColor {
        switch tier {
        case .safest: return safestColor
        case .medium: return mediumColor
        case .unsafest: return unsafestColor
        }
    }

    static func strokeColor(tier: MockRouteSafetyTier, isSelected: Bool) -> UIColor {
        let opacity = isSelected ? selectedOpacity : unselectedOpacity
        return baseColor(for: tier).withAlphaComponent(opacity)
    }

    static func strokeWidth(isSelected: Bool) -> CGFloat {
        isSelected ? selectedStrokeWidth : unselectedStrokeWidth
    }

    static func zIndex(isSelected: Bool) -> Int32 {
        isSelected ? selectedZIndex : unselectedZIndex
    }
}

/// Temporary deterministic crime-risk scores for preview routes.
///
/// TODO: Replace `MockRouteRiskScorer` scores with real Python model crime-risk outputs.
/// Lower score = safer; higher score = more dangerous.
enum MockRouteRiskScorer {
    /// Fixed sequence by Google response order (`route_0` … `route_3`).
    private static let scoresByResponseIndex: [Double] = [0.55, 0.20, 0.85, 0.40]

    /// Deterministic mock score in `0.0...1.0` for a Google response index.
    static func mockCrimeScore(forResponseIndex index: Int) -> Double {
        precondition(index >= 0)
        if index < scoresByResponseIndex.count {
            return scoresByResponseIndex[index]
        }
        // Safe deterministic fallback if Google ever returns more than four routes.
        let cycle = scoresByResponseIndex[index % scoresByResponseIndex.count]
        let bump = Double(index / scoresByResponseIndex.count) * 0.01
        return min(1.0, cycle + bump)
    }

    /// Assigns mock scores and relative safety tiers. Does not change route order for extraction.
    static func applyingScoresAndTiers(to routes: [ComputedRoute]) -> [ComputedRoute] {
        guard !routes.isEmpty else { return [] }

        var scored = routes.map { route -> ComputedRoute in
            var copy = route
            copy.mockCrimeScore = mockCrimeScore(forResponseIndex: route.responseIndex)
            copy.safetyTier = nil
            return copy
        }

        let rankedIDs = scored
            .sorted(by: saferThan)
            .map(\.id)

        let tiersByID = tiers(forRankedIDs: rankedIDs)
        for index in scored.indices {
            scored[index].safetyTier = tiersByID[scored[index].id]
        }
        return scored
    }

    /// Ranked safest-first (ascending mock score; ties → lower `responseIndex`).
    static func rankedSafestFirst(_ routes: [ComputedRoute]) -> [ComputedRoute] {
        routes.sorted(by: saferThan)
    }

    /// Route ID with the lowest mock crime score (ties → Google response order).
    static func safestRouteID(in routes: [ComputedRoute]) -> String? {
        rankedSafestFirst(routes).first?.id
    }

    /// Ascending score; equal scores break ties by original Google response order.
    static func saferThan(_ lhs: ComputedRoute, _ rhs: ComputedRoute) -> Bool {
        let leftScore = lhs.mockCrimeScore ?? .greatestFiniteMagnitude
        let rightScore = rhs.mockCrimeScore ?? .greatestFiniteMagnitude
        if leftScore != rightScore {
            return leftScore < rightScore
        }
        return lhs.responseIndex < rhs.responseIndex
    }

    static func tiers(forRankedIDs rankedIDs: [String]) -> [String: MockRouteSafetyTier] {
        let count = rankedIDs.count
        guard count > 0 else { return [:] }

        var result: [String: MockRouteSafetyTier] = [:]
        switch count {
        case 1:
            result[rankedIDs[0]] = .safest
        case 2:
            result[rankedIDs[0]] = .safest
            result[rankedIDs[1]] = .unsafest
        case 3:
            result[rankedIDs[0]] = .safest
            result[rankedIDs[1]] = .medium
            result[rankedIDs[2]] = .unsafest
        default:
            // Four or more: lowest safest, highest unsafest, everything between medium.
            result[rankedIDs[0]] = .safest
            result[rankedIDs[count - 1]] = .unsafest
            for id in rankedIDs.dropFirst().dropLast() {
                result[id] = .medium
            }
        }
        return result
    }

    #if DEBUG
    static func logScores(_ routes: [ComputedRoute], selectedRouteID: String?) {
        let ordered = routes.sorted { $0.responseIndex < $1.responseIndex }
        for route in ordered {
            let score = route.mockCrimeScore.map { String(format: "%.2f", $0) } ?? "nil"
            let tier = route.safetyTier?.rawValue ?? "nil"
            let selected = route.id == selectedRouteID
            print("[MOCK_ROUTE_SCORE] \(route.id) score=\(score) tier=\(tier) selected=\(selected)")
        }
    }
    #endif
}
