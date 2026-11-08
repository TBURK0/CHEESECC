import Foundation

struct CarModel: Codable, Identifiable, Hashable {
    let id: String
    let make: String
    let model: String
    let year: Int
    let bodyStyle: String
    let lengthMM: Double
    let widthMM: Double
    let heightMM: Double
    let wheelbaseMM: Double
    let frontTrackMM: Double
    let rearTrackMM: Double
    let curbWeightKG: Double
    let dragCoefficient: Double
    let frontalAreaM2: Double
    let engineType: String
    let displacementL: Double?
    let horsepower: Double
    let torqueNm: Double
    let drivetrain: String
    let steeringRatio: Double
    let maxSteerWheelDeg: Double
    let tireRadiusM: Double

    var displayName: String { "\(year) \(make) \(model)" }

    var lengthM: Double { lengthMM / 1000 }
    var widthM: Double { widthMM / 1000 }
    var heightM: Double { heightMM / 1000 }
    var wheelbaseM: Double { wheelbaseMM / 1000 }

    var horsepowerPerTonne: Double {
        horsepower / (curbWeightKG / 1000)
    }

    /// Rear-axle radius at full lock. This is tighter than a published curb-to-curb circle.
    var rearAxleTurnRadiusM: Double {
        let roadWheel = (maxSteerWheelDeg / steeringRatio) * .pi / 180
        let tangent = tan(roadWheel)
        guard tangent > 0.05 else { return .infinity }
        return wheelbaseM / tangent
    }
}
