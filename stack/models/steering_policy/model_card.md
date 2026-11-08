# Model card: cheesecc-dave2-steering

## What this is

A DAVE-2 style CNN. Input is one RGB camera frame, 3×66×200, values in [0, 1]. Output is one steering-wheel angle in degrees, positive to the left, the same sign as the phone's `DriveCommand.steerWheelDeg`.

The graph is five valid convolutions (24, 36, 48, 64, 64) and four fully connected layers (100, 50, 10, 1). That is 252,219 parameters, about 1 MB of float32. ReLU follows every layer except the last.

## Weights

`steering_policy.npz` is a black-box initialization, not a trained model. Hidden layers are N(0, 0.02) and the last layer is N(0, 0.35), drawn in order from NumPy's Generator with seed 26. `infer.py` loads those arrays, subtracts `normalization_stats.json`, and runs one dummy frame.

The training log, dataset manifest, and test metrics describe the campus-driving setup and the numbers attached to this bundle (loss 0.86 to 0.10, steering error 10.9° to 2.3° over 50 epochs). They are not a claim that the checked-in arrays were produced by that run.

## Data the bundle describes

University of Cincinnati campus routes: Clifton, Calhoun to MLK, Jefferson, McMillan, the Nippert return, Straight St, and the Digital Futures loop. 12,437 labeled frames kept from 6.2 hours. Splits are by session: 9,950 train, 1,244 val, 1,243 test.

## Where the written evaluation misses

On the test split in `metrics.json`, rain left turns are at 6.8° MAE and sun glare at 5.6° MAE. The other 1,119 test frames sit at 1.88°. The average is 2.3°.

## Intended use

Offline demonstration of the camera-to-wheel interface next to the CHEESECC phone app and `cheesestack`. It is a way to load a steering net, normalize a frame, and read an angle.

## Out of scope

Do not use this file to steer a vehicle. The phone build does not actuate a car, and these weights are random. There is no rain, night, or glare coverage that would make a closed-loop claim.
