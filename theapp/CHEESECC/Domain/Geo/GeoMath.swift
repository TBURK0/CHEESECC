import Foundation

struct LatLon: Hashable, Codable {
    var latitude: Double
    var longitude: Double
}

enum DemoOrigin {
    /// University of Cincinnati. Used when the phone has no GPS fix yet.
    static let cincinnati = LatLon(latitude: 39.1329, longitude: -84.5150)
}

enum GeoMath {
    private static let earthRadiusM = 6_371_000.0

    static func distanceM(_ a: LatLon, _ b: LatLon) -> Double {
        let lat1 = a.latitude * .pi / 180
        let lat2 = b.latitude * .pi / 180
        let dLat = (b.latitude - a.latitude) * .pi / 180
        let dLon = (b.longitude - a.longitude) * .pi / 180
        let h = sin(dLat / 2) * sin(dLat / 2)
            + cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2)
        return 2 * earthRadiusM * asin(min(1, sqrt(max(0, h))))
    }

    /// Degrees clockwise from north, in [0, 360).
    static func bearingDeg(from a: LatLon, to b: LatLon) -> Double {
        let lat1 = a.latitude * .pi / 180
        let lat2 = b.latitude * .pi / 180
        let dLon = (b.longitude - a.longitude) * .pi / 180
        let y = sin(dLon) * cos(lat2)
        let x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
        let degrees = atan2(y, x) * 180 / .pi
        return (degrees + 360).truncatingRemainder(dividingBy: 360)
    }

    /// Signed smallest delta, in (-180, 180].
    static func wrapDegrees(_ value: Double) -> Double {
        var degrees = value.truncatingRemainder(dividingBy: 360)
        if degrees > 180 { degrees -= 360 }
        if degrees <= -180 { degrees += 360 }
        return degrees
    }

    static func toMeters(_ point: LatLon, origin: LatLon) -> (x: Double, y: Double) {
        let lat0 = origin.latitude * .pi / 180
        let x = (point.longitude - origin.longitude) * .pi / 180 * cos(lat0) * earthRadiusM
        let y = (point.latitude - origin.latitude) * .pi / 180 * earthRadiusM
        return (x, y)
    }

    /// Projects `point` onto segment AB. `t` is 0 at A and 1 at B. Offset is meters off the segment.
    static func project(_ point: LatLon, onto a: LatLon, _ b: LatLon) -> (t: Double, offsetM: Double) {
        let ab = toMeters(b, origin: a)
        let ap = toMeters(point, origin: a)
        let abLen2 = ab.x * ab.x + ab.y * ab.y
        let t: Double
        if abLen2 < 0.01 {
            t = 0
        } else {
            t = min(1, max(0, (ap.x * ab.x + ap.y * ab.y) / abLen2))
        }
        let dx = ap.x - ab.x * t
        let dy = ap.y - ab.y * t
        return (t, hypot(dx, dy))
    }

    /// Signed Menger curvature in 1/m. Positive curves to the right of travel A→B→C
    /// when bearings increase clockwise, which matches a positive steer angle.
    static func signedCurvature(_ a: LatLon, _ b: LatLon, _ c: LatLon) -> Double {
        let (x1, y1) = toMeters(a, origin: b)
        let x2 = 0.0
        let y2 = 0.0
        let (x3, y3) = toMeters(c, origin: b)
        let shoelace = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        let d12 = hypot(x2 - x1, y2 - y1)
        let d23 = hypot(x3 - x2, y3 - y2)
        let d13 = hypot(x3 - x1, y3 - y1)
        let denom = d12 * d23 * d13
        guard denom > 1 else { return 0 }
        // Negate so a right turn (clockwise) is positive, matching steer-right.
        return (-2 * shoelace) / denom
    }
}
