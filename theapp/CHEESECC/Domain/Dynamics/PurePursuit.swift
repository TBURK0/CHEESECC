import Foundation

enum PurePursuit {
    /// Curvature (1/m) that aims the bicycle at a point `lookaheadM` ahead.
    /// Heading error is radians, positive when the target bears to the right.
    static func curvature(lookaheadM: Double, headingErrorRad: Double) -> Double {
        let lookahead = max(1, lookaheadM)
        let alpha = min(.pi / 2, max(-.pi / 2, headingErrorRad))
        return (2 * sin(alpha)) / lookahead
    }
}
