# MAKEUC 26'

CHEESECC is the iPhone side of the retrofit demo. Open `theapp/CHEESECC.xcodeproj` in Xcode 16 or newer.

The phone does three things:

- Apple Maps route from the phone's GPS, then Go. The route is broken into short pushes: where the car should be next. The horizon is about 0.80s when stopped and about 0.25s at highway speed.
- A garage of real cars. Wheelbase, track, mass, engine, steering ratio, and tire size change the wheel angle and the speed target. Add a car in `theapp/CHEESECC/Data/Catalog/cars.json`.
- An on-device log of those pushes, tied to the map point and the selected car. Learn charts handwheel angle against curvature. Drag the wheel during a run and those samples are marked `driver`. Export writes JSON.

The phone follows a live GPS fix, or an Xcode simulated location or GPX track the same way. Command frames stay in the on-screen bus and in SwiftData. This build does not actuate a car.

Bundle id is `com.makeuc.cheesecc`. Change it to the id registered on your Apple team before upload.

## TestFlight

Internal testers are the hackathon path. External review is too slow.

1. In App Store Connect, create the app with the same bundle id.
2. Create an App Store Connect API key. On the GitHub repo, add secrets `APPLE_TEAM_ID`, `ASC_KEY_ID`, `ASC_ISSUER_ID`, and `ASC_PRIVATE_KEY` (the full `.p8` file).
3. Run the **TestFlight** workflow. It archives `theapp/CHEESECC.xcodeproj` on a Mac runner and uploads.
4. Add internal testers. Processing usually takes a few tens of minutes.

This Windows checkout cannot compile or sign the iOS app. Until that workflow has secrets, run the Xcode simulator with a simulated location or GPX track, or install from Xcode onto a phone.
