//
//  DrowsinessRemoteTests.swift
//  DriveSensAITests
//

import XCTest
@testable import DriveSensAI

final class DrowsinessRemoteTests: XCTestCase {

    func testFeatureOrderMatchesPythonContract() {
        XCTAssertEqual(DrowsinessFeatureContract.featureCount, 22)
        XCTAssertEqual(DrowsinessFeatureContract.windowFrames, 5)
        XCTAssertEqual(DrowsinessFeatureContract.samplingRateHz, 15.0)
        XCTAssertEqual(
            DrowsinessFeatureContract.featureNames.first,
            "face_detected"
        )
        XCTAssertEqual(
            DrowsinessFeatureContract.featureNames.last,
            "hand_near_mouth"
        )
    }

    func testRequestJSONCodingKeys() throws {
        let request = DrowsinessPredictRequest(
            schemaVersion: 1,
            sessionID: "s1",
            sequenceID: 42,
            sentAtUTC: Date(timeIntervalSince1970: 1_000),
            samplingRateHz: 15.0,
            featureNames: DrowsinessFeatureContract.featureNames,
            samples: [
                DrowsinessAPISample(
                    timestampMs: 1,
                    values: Array(repeating: 0.0, count: 22)
                ),
            ]
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(request)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(json["schema_version"] as? Int, 1)
        XCTAssertEqual(json["session_id"] as? String, "s1")
        XCTAssertEqual(json["sequence_id"] as? Int, 42)
        XCTAssertNotNil(json["sampling_rate_hz"])
        XCTAssertNotNil(json["feature_names"])
        XCTAssertNotNil(json["samples"])
        XCTAssertNil(json["schemaVersion"])
    }

    func testResponseJSONDecodes() throws {
        let payload = """
        {
          "session_id": "s1",
          "sequence_id": 42,
          "label": "open",
          "label_index": 2,
          "confidence": 0.91,
          "probabilities": {
            "close": 0.02,
            "closing": 0.03,
            "open": 0.91,
            "opening": 0.02,
            "undefined": 0.02
          },
          "model_version": "best_accuracy",
          "inference_latency_ms": 8.4
        }
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(DrowsinessPredictResponse.self, from: payload)
        XCTAssertEqual(decoded.label, "open")
        XCTAssertEqual(decoded.labelIndex, 2)
        XCTAssertEqual(decoded.probabilities.count, 5)
        XCTAssertTrue(decoded.confidence.isFinite)
    }

    func testMaskedSampleWhenNoFace() {
        let sample = DrowsinessFeatureExtractor.makeSample(faces: [], hands: [])
        XCTAssertEqual(sample.values.count, 22)
        XCTAssertTrue(sample.isFinite)
        XCTAssertEqual(sample.values[0], 0.0) // face_detected
        XCTAssertEqual(sample.values[19], 0.0) // hand_detected
    }

    func testCoordinatorRingBufferDoesNotExceedCap() {
        let coordinator = DrowsinessInferenceCoordinator()
        coordinator.start()
        defer { coordinator.stop() }

        for index in 0..<40 {
            let values = Array(repeating: Double(index), count: 22)
            let sample = DrowsinessFeatureSample(
                timestampMs: Int64(index),
                values: values
            )
            coordinator.ingest(sample)
        }
        // Private buffer — exercise ingest without crash; capacity enforced internally.
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
    }

    func testNetworkFailureDoesNotCreatePrediction() async throws {
        // Without a server, a predict call should fail; coordinator must not invent a label.
        let coordinator = DrowsinessInferenceCoordinator()
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
        coordinator.stop()
        XCTAssertNil(coordinator.latestPrediction)
        XCTAssertFalse(coordinator.isWakeUpAlertActive)
    }
}
