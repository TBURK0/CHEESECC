import Foundation

struct RoutePath {
    let coordinates: [LatLon]
    let cumulative: [Double]

    init(coordinates: [LatLon]) {
        self.coordinates = coordinates
        var totals: [Double] = []
        totals.reserveCapacity(coordinates.count)
        var total = 0.0
        for index in coordinates.indices {
            if index == 0 {
                totals.append(0)
            } else {
                total += GeoMath.distanceM(coordinates[index - 1], coordinates[index])
                totals.append(total)
            }
        }
        cumulative = totals
    }

    var totalM: Double { cumulative.last ?? 0 }

    func point(atDistance distance: Double) -> (coordinate: LatLon, bearingDeg: Double) {
        guard let first = coordinates.first else {
            return (DemoOrigin.cincinnati, 0)
        }
        guard coordinates.count >= 2 else {
            return (first, 0)
        }
        let clamped = min(max(0, distance), totalM)
        var index = 1
        while index < cumulative.count - 1 && cumulative[index] < clamped {
            index += 1
        }
        let start = cumulative[index - 1]
        let end = cumulative[index]
        let span = max(end - start, 0.001)
        let t = min(1, max(0, (clamped - start) / span))
        let a = coordinates[index - 1]
        let b = coordinates[index]
        let coordinate = LatLon(
            latitude: a.latitude + (b.latitude - a.latitude) * t,
            longitude: a.longitude + (b.longitude - a.longitude) * t
        )
        return (coordinate, GeoMath.bearingDeg(from: a, to: b))
    }

    func closest(to point: LatLon) -> (along: Double, offsetM: Double) {
        guard coordinates.count >= 2 else { return (0, 0) }
        var bestAlong = 0.0
        var bestOffset = Double.greatestFiniteMagnitude
        for index in 1..<coordinates.count {
            let a = coordinates[index - 1]
            let b = coordinates[index]
            let segment = GeoMath.distanceM(a, b)
            let projection = GeoMath.project(point, onto: a, b)
            let along = cumulative[index - 1] + projection.t * segment
            if projection.offsetM < bestOffset {
                bestOffset = projection.offsetM
                bestAlong = along
            }
        }
        return (bestAlong, bestOffset)
    }

    func curvature(atDistance distance: Double, window: Double = 14) -> Double {
        let clamped = min(max(0, distance), totalM)
        let back = point(atDistance: max(0, clamped - window)).coordinate
        let mid = point(atDistance: clamped).coordinate
        let forward = point(atDistance: min(totalM, clamped + window)).coordinate
        return GeoMath.signedCurvature(back, mid, forward)
    }
}
