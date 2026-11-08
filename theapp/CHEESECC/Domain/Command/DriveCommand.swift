import Foundation

struct DriveCommand: Identifiable, Codable, Equatable {
    var id: UInt64
    var lookaheadS: Double
    var latitude: Double
    var longitude: Double
    var fromLatitude: Double
    var fromLongitude: Double
    var bearingDeg: Double
    var speedMps: Double
    var targetSpeedMps: Double
    var steerWheelDeg: Double
    var throttle: Double
    var brake: Double
    var curvature: Double
    var carId: String
    var source: String

    var line: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(self),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }
}
