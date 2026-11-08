import Foundation

struct OnlineRegressor: Equatable {
    private(set) var count = 0
    private var sumX = 0.0
    private var sumY = 0.0
    private var sumXX = 0.0
    private var sumXY = 0.0

    mutating func observe(x: Double, y: Double) {
        guard x.isFinite, y.isFinite else { return }
        count += 1
        sumX += x
        sumY += y
        sumXX += x * x
        sumXY += x * y
    }

    /// Least-squares slope of y on x. Nil until there is enough spread to mean something.
    var slope: Double? {
        guard count >= 8 else { return nil }
        let n = Double(count)
        let denom = n * sumXX - sumX * sumX
        guard abs(denom) > 1e-9 else { return nil }
        return (n * sumXY - sumX * sumY) / denom
    }

    static func fit(curvature: [Double], steer: [Double]) -> OnlineRegressor {
        var model = OnlineRegressor()
        for (x, y) in zip(curvature, steer) where abs(x) > 0.0008 {
            model.observe(x: x, y: y)
        }
        return model
    }
}
