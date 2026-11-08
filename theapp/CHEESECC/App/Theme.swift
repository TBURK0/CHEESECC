import SwiftUI

enum Theme {
    static let amber = Color(red: 0.95, green: 0.70, blue: 0.22)
    static let good = Color(red: 0.42, green: 0.86, blue: 0.62)
    static let ink = Color(red: 0.07, green: 0.08, blue: 0.09)
}

enum Units {
    static func mph(_ metersPerSecond: Double) -> String {
        String(format: "%.0f mph", max(0, metersPerSecond) * 2.2369362921)
    }

    static func meters(_ meters: Double) -> String {
        String(format: "%.0f m", meters)
    }
}
