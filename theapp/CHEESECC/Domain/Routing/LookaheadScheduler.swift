import Foundation

enum LookaheadScheduler {
    /// Fast driving shortens the horizon toward 0.25s. A stop stretches it to 0.80s
    /// so the next push stays a readable step instead of GPS noise.
    static func horizon(speedMps: Double) -> Double {
        let fraction = min(1, max(0, speedMps) / 25)
        return 0.80 - fraction * 0.55
    }

    static func distance(speedMps: Double) -> Double {
        max(8, max(0, speedMps) * horizon(speedMps: speedMps))
    }
}
