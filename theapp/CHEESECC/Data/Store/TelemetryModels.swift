import Foundation
import SwiftData

@Model
final class DriveSession {
    var startedAt: Date
    var carId: String
    var carName: String
    @Relationship(deleteRule: .cascade, inverse: \SteerSample.session)
    var samples: [SteerSample]

    init(carId: String, carName: String, startedAt: Date = Date()) {
        self.startedAt = startedAt
        self.carId = carId
        self.carName = carName
        self.samples = []
    }
}

@Model
final class SteerSample {
    var time: Date
    var latitude: Double
    var longitude: Double
    var speedMps: Double
    var headingDeg: Double
    var curvature: Double
    var steerWheelDeg: Double
    var throttle: Double
    var brake: Double
    var lookaheadS: Double
    var source: String
    var carId: String
    var session: DriveSession?

    init(
        time: Date,
        latitude: Double,
        longitude: Double,
        speedMps: Double,
        headingDeg: Double,
        curvature: Double,
        steerWheelDeg: Double,
        throttle: Double,
        brake: Double,
        lookaheadS: Double,
        source: String,
        carId: String
    ) {
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.speedMps = speedMps
        self.headingDeg = headingDeg
        self.curvature = curvature
        self.steerWheelDeg = steerWheelDeg
        self.throttle = throttle
        self.brake = brake
        self.lookaheadS = lookaheadS
        self.source = source
        self.carId = carId
        self.session = nil
    }
}
