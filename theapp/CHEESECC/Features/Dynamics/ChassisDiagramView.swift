import SwiftUI

struct ChassisDiagramView: View {
    let car: CarModel
    let steerWheelDeg: Double

    var body: some View {
        Canvas { context, size in
            let length = CGFloat(max(car.lengthM, 1))
            let width = CGFloat(max(car.widthM, 0.6))
            let scale = min((size.width - 24) / width, (size.height - 12) / length)
            let bodyLength = length * scale
            let bodyWidth = width * scale
            let top = (size.height - bodyLength) / 2
            let left = (size.width - bodyWidth) / 2
            let body = CGRect(x: left, y: top, width: bodyWidth, height: bodyLength)
            let bodyPath = Path(roundedRect: body, cornerRadius: bodyWidth * 0.16)
            context.fill(bodyPath, with: .color(Color.white.opacity(0.08)))
            context.stroke(bodyPath, with: .color(Theme.amber), lineWidth: 1.6)

            let wheelbase = min(CGFloat(car.wheelbaseM), length * 0.92)
            let overhang = bodyLength - wheelbase * scale
            let frontOverhang = max(4, overhang * 0.42)
            let rearAxleY = body.maxY - (overhang - frontOverhang)
            let frontAxleY = rearAxleY - wheelbase * scale
            let centerX = size.width / 2
            let frontHalf = min(bodyWidth * 0.42, CGFloat(car.frontTrackMM / 1000) * scale / 2)
            let rearHalf = min(bodyWidth * 0.42, CGFloat(car.rearTrackMM / 1000) * scale / 2)
            let roadAngle = CGFloat(steerWheelDeg / max(car.steeringRatio, 1)) * .pi / 180

            strokeAxle(context: context, y: frontAxleY, centerX: centerX, half: frontHalf)
            strokeAxle(context: context, y: rearAxleY, centerX: centerX, half: rearHalf)

            let wheelLength = max(8, CGFloat(0.66) * scale)
            let wheelWidth = max(4, CGFloat(0.22) * scale)
            drawWheel(context, x: centerX - frontHalf, y: frontAxleY, w: wheelWidth, h: wheelLength, angle: roadAngle)
            drawWheel(context, x: centerX + frontHalf, y: frontAxleY, w: wheelWidth, h: wheelLength, angle: roadAngle)
            drawWheel(context, x: centerX - rearHalf, y: rearAxleY, w: wheelWidth, h: wheelLength, angle: 0)
            drawWheel(context, x: centerX + rearHalf, y: rearAxleY, w: wheelWidth, h: wheelLength, angle: 0)

            var nose = Path()
            nose.move(to: CGPoint(x: centerX - bodyWidth * 0.16, y: top + bodyLength * 0.14))
            nose.addLine(to: CGPoint(x: centerX, y: top + 5))
            nose.addLine(to: CGPoint(x: centerX + bodyWidth * 0.16, y: top + bodyLength * 0.14))
            context.stroke(nose, with: .color(Theme.amber), lineWidth: 1.5)
        }
        .accessibilityLabel("Top-down \(car.displayName), steering \(Int(steerWheelDeg)) degrees")
    }

    private func strokeAxle(context: GraphicsContext, y: CGFloat, centerX: CGFloat, half: CGFloat) {
        var path = Path()
        path.move(to: CGPoint(x: centerX - half, y: y))
        path.addLine(to: CGPoint(x: centerX + half, y: y))
        context.stroke(path, with: .color(Color.white.opacity(0.35)), lineWidth: 1)
    }

    private func drawWheel(
        _ context: GraphicsContext,
        x: CGFloat,
        y: CGFloat,
        w: CGFloat,
        h: CGFloat,
        angle: CGFloat
    ) {
        let corners = [
            CGPoint(x: -w / 2, y: -h / 2),
            CGPoint(x: w / 2, y: -h / 2),
            CGPoint(x: w / 2, y: h / 2),
            CGPoint(x: -w / 2, y: h / 2)
        ]
        let sine = sin(angle)
        let cosine = cos(angle)
        let points = corners.map { corner in
            CGPoint(
                x: x + corner.x * cosine - corner.y * sine,
                y: y + corner.x * sine + corner.y * cosine
            )
        }
        var path = Path()
        path.move(to: points[0])
        for point in points.dropFirst() {
            path.addLine(to: point)
        }
        path.closeSubpath()
        context.fill(path, with: .color(Theme.amber))
    }
}
