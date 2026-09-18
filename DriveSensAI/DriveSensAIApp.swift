import SwiftUI

@main
struct DriveSensAIApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self)
    private var appDelegate

    var body: some Scene {
        WindowGroup {
            DriveView()
        }
    }
}
