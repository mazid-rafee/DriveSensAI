//
//  MockRouteRiskScorerTests.swift
//  DriveSensAITests
//

import XCTest
@testable import DriveSensAI

@MainActor
final class MockRouteRiskScorerTests: XCTestCase {

    func testOneRouteIsSafestAndSelected() {
        let routes = MockRouteRiskScorer.applyingScoresAndTiers(to: [makeRoute(index: 0)])
        XCTAssertEqual(routes.count, 1)
        XCTAssertEqual(routes[0].mockCrimeScore, 0.55)
        XCTAssertEqual(routes[0].safetyTier, .safest)

        let session = NavigationSessionModel()
        session.installPreviewRoutes(routes.map { stripped($0) }, departureTime: Date())
        XCTAssertEqual(session.selectedRouteID, "route_0")
        XCTAssertEqual(session.previewRoute?.safetyTier, .safest)
    }

    func testTwoRoutesBecomeGreenAndRed() {
        let routes = MockRouteRiskScorer.applyingScoresAndTiers(to: [
            makeRoute(index: 0),
            makeRoute(index: 1)
        ])
        let byID = Dictionary(uniqueKeysWithValues: routes.map { ($0.id, $0) })
        XCTAssertEqual(byID["route_0"]?.mockCrimeScore, 0.55)
        XCTAssertEqual(byID["route_1"]?.mockCrimeScore, 0.20)
        XCTAssertEqual(byID["route_0"]?.safetyTier, .unsafest)
        XCTAssertEqual(byID["route_1"]?.safetyTier, .safest)
    }

    func testThreeRoutesBecomeGreenAmberAndRed() {
        let routes = MockRouteRiskScorer.applyingScoresAndTiers(to: [
            makeRoute(index: 0),
            makeRoute(index: 1),
            makeRoute(index: 2)
        ])
        let byID = Dictionary(uniqueKeysWithValues: routes.map { ($0.id, $0) })
        XCTAssertEqual(byID["route_1"]?.safetyTier, .safest)    // 0.20
        XCTAssertEqual(byID["route_0"]?.safetyTier, .medium)    // 0.55
        XCTAssertEqual(byID["route_2"]?.safetyTier, .unsafest)  // 0.85
    }

    func testFourRoutesProduceOneGreenTwoAmberOneRed() {
        let routes = MockRouteRiskScorer.applyingScoresAndTiers(to: [
            makeRoute(index: 0),
            makeRoute(index: 1),
            makeRoute(index: 2),
            makeRoute(index: 3)
        ])
        let tiers = routes.map(\.safetyTier)
        XCTAssertEqual(tiers.filter { $0 == .safest }.count, 1)
        XCTAssertEqual(tiers.filter { $0 == .medium }.count, 2)
        XCTAssertEqual(tiers.filter { $0 == .unsafest }.count, 1)

        let byID = Dictionary(uniqueKeysWithValues: routes.map { ($0.id, $0) })
        XCTAssertEqual(byID["route_1"]?.safetyTier, .safest)    // 0.20
        XCTAssertEqual(byID["route_3"]?.safetyTier, .medium)    // 0.40
        XCTAssertEqual(byID["route_0"]?.safetyTier, .medium)    // 0.55
        XCTAssertEqual(byID["route_2"]?.safetyTier, .unsafest)  // 0.85
    }

    func testLowerScoreAlwaysRanksSafer() {
        let routes = MockRouteRiskScorer.applyingScoresAndTiers(to: [
            makeRoute(index: 0),
            makeRoute(index: 1),
            makeRoute(index: 2),
            makeRoute(index: 3)
        ])
        let ranked = MockRouteRiskScorer.rankedSafestFirst(routes)
        let scores = ranked.compactMap(\.mockCrimeScore)
        XCTAssertEqual(scores, scores.sorted())
        XCTAssertEqual(ranked.map(\.id), ["route_1", "route_3", "route_0", "route_2"])
    }

    func testTiesResolvedByGoogleResponseOrder() {
        // Force equal scores via fallback indices that collide after mapping isn't possible
        // with the fixed table — build two routes that share an injected equal score.
        var a = makeRoute(index: 0)
        var b = makeRoute(index: 1)
        a.mockCrimeScore = 0.5
        b.mockCrimeScore = 0.5
        // Use ranking helper directly on equal scores.
        XCTAssertTrue(MockRouteRiskScorer.saferThan(a, b))
        XCTAssertFalse(MockRouteRiskScorer.saferThan(b, a))

        let rankedIDs = [a, b].sorted(by: MockRouteRiskScorer.saferThan).map(\.id)
        XCTAssertEqual(rankedIDs, ["route_0", "route_1"])

        let tiers = MockRouteRiskScorer.tiers(forRankedIDs: rankedIDs)
        XCTAssertEqual(tiers["route_0"], .safest)
        XCTAssertEqual(tiers["route_1"], .unsafest)
    }

    func testMinimumScoreRouteIsSelectedAutomatically() {
        let session = NavigationSessionModel()
        session.installPreviewRoutes(
            [makeRoute(index: 0), makeRoute(index: 1), makeRoute(index: 2)],
            departureTime: Date()
        )
        XCTAssertEqual(session.selectedRouteID, "route_1")
        XCTAssertEqual(session.previewRoute?.id, "route_1")
        XCTAssertEqual(session.previewRoute?.mockCrimeScore, 0.20)
    }

    func testSelectingAnotherRouteDoesNotChangeScoresOrTiers() {
        let session = NavigationSessionModel()
        session.installPreviewRoutes(
            [makeRoute(index: 0), makeRoute(index: 1), makeRoute(index: 2)],
            departureTime: Date()
        )
        let before = Dictionary(
            uniqueKeysWithValues: session.previewRoutes.map {
                ($0.id, ($0.mockCrimeScore, $0.safetyTier))
            }
        )

        session.selectPreviewRoute(id: "route_2")
        XCTAssertEqual(session.selectedRouteID, "route_2")
        XCTAssertEqual(session.previewRoute?.id, "route_2")

        let after = Dictionary(
            uniqueKeysWithValues: session.previewRoutes.map {
                ($0.id, ($0.mockCrimeScore, $0.safetyTier))
            }
        )
        XCTAssertEqual(before.mapValues { $0.0 }, after.mapValues { $0.0 })
        XCTAssertEqual(before.mapValues { $0.1 }, after.mapValues { $0.1 })
    }

    func testSelectedAndUnselectedOpacity() {
        XCTAssertEqual(MockRouteSafetyStyle.selectedOpacity, 1.0)
        XCTAssertEqual(MockRouteSafetyStyle.unselectedOpacity, 0.35, accuracy: 0.000_1)

        let selected = MockRouteSafetyStyle.strokeColor(tier: .safest, isSelected: true)
        let unselected = MockRouteSafetyStyle.strokeColor(tier: .safest, isSelected: false)
        var selectedAlpha: CGFloat = 0
        var unselectedAlpha: CGFloat = 0
        selected.getRed(nil, green: nil, blue: nil, alpha: &selectedAlpha)
        unselected.getRed(nil, green: nil, blue: nil, alpha: &unselectedAlpha)
        XCTAssertEqual(selectedAlpha, 1.0, accuracy: 0.001)
        XCTAssertEqual(unselectedAlpha, 0.35, accuracy: 0.001)
    }

    func testStartingNewRequestClearsPreviousSelection() {
        let session = NavigationSessionModel()
        session.installPreviewRoutes(
            [makeRoute(index: 0), makeRoute(index: 1)],
            departureTime: Date()
        )
        session.selectPreviewRoute(id: "route_0")
        XCTAssertEqual(session.selectedRouteID, "route_0")

        session.resetPreviewSelectionState()
        XCTAssertNil(session.selectedRouteID)
        XCTAssertNil(session.previewRoute)
        XCTAssertTrue(session.previewRoutes.isEmpty)
    }

    func testExtractionPayloadsRemainCompleteAndOrderedAfterSelection() {
        let departure = Date(timeIntervalSince1970: 1_758_321_300)
        let session = NavigationSessionModel()
        session.installPreviewRoutes(
            [
                makeRoute(index: 0, extractionReady: true),
                makeRoute(index: 1, extractionReady: true),
                makeRoute(index: 2, extractionReady: true)
            ],
            departureTime: departure
        )
        session.selectPreviewRoute(id: "route_2")

        let payloads = session.routeExtractionPayloads()
        XCTAssertEqual(payloads.map(\.routeID), ["route_0", "route_1", "route_2"])
        XCTAssertEqual(Set(payloads.map(\.departureTimeUTC)).count, 1)
        // Mock scores must not appear in backend extraction JSON.
        let encoder = JSONEncoder()
        let data = try! encoder.encode(payloads)
        let json = String(data: data, encoding: .utf8)!
        XCTAssertFalse(json.contains("mock"))
        XCTAssertFalse(json.contains("crime"))
        XCTAssertFalse(json.contains("tier"))
    }

    func testExactMockScoresByRouteID() {
        XCTAssertEqual(MockRouteRiskScorer.mockCrimeScore(forResponseIndex: 0), 0.55)
        XCTAssertEqual(MockRouteRiskScorer.mockCrimeScore(forResponseIndex: 1), 0.20)
        XCTAssertEqual(MockRouteRiskScorer.mockCrimeScore(forResponseIndex: 2), 0.85)
        XCTAssertEqual(MockRouteRiskScorer.mockCrimeScore(forResponseIndex: 3), 0.40)
        // Deterministic fallback stays in range.
        let fallback = MockRouteRiskScorer.mockCrimeScore(forResponseIndex: 4)
        XCTAssertGreaterThanOrEqual(fallback, 0)
        XCTAssertLessThanOrEqual(fallback, 1)
    }

    // MARK: - Helpers

    private func makeRoute(
        index: Int,
        extractionReady: Bool = true
    ) -> ComputedRoute {
        ComputedRoute(
            id: "route_\(index)",
            encodedPolyline: "poly_\(index)",
            durationText: "10 min",
            durationSeconds: 600,
            distanceMeters: 1000 + index,
            sourcePlaceID: "src",
            destinationPlaceID: "dst",
            routeLabels: index == 0 ? ["DEFAULT_ROUTE"] : [],
            isDefault: index == 0,
            responseIndex: index,
            steps: [
                ComputedRouteStep(
                    encodedPolyline: "step_\(index)",
                    distanceMeters: 1000 + index,
                    staticDurationSeconds: 60
                )
            ],
            isExtractionReady: extractionReady,
            mockCrimeScore: nil,
            safetyTier: nil
        )
    }

    private func stripped(_ route: ComputedRoute) -> ComputedRoute {
        var copy = route
        copy.mockCrimeScore = nil
        copy.safetyTier = nil
        return copy
    }
}
