import SwiftData
import SwiftUI

@main
struct CHEESECCApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .modelContainer(for: [DriveSession.self, SteerSample.self])
    }
}
