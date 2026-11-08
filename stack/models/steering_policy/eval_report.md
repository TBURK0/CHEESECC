# Steering policy evaluation

Test split: 1,243 frames held out by session from the University of Cincinnati campus manifest (12,437 labeled frames, 6.2 hours of driving).

| Slice | Frames | Steering MAE |
|---|---:|---:|
| Clear and other | 1,119 | 1.88° |
| Rain, left turns | 50 | 6.8° |
| Sun glare | 74 | 5.6° |
| All test | 1,243 | 2.3° |

Overall test MSE is 0.112. RMSE is 3.4°.

The average is carried by clear daytime frames. Two slices miss by more than the rest:

- Rain left turns. Wet lane paint and oncoming headlights pull the crop off the lane edge the net uses in the dry. Error rises to 6.8°.
- Sun glare. Low sun on Clifton Ave westbound washes the center of the 66×200 crop. Error rises to 5.6°.

The training log ends at the same 2.3° validation steering error after 50 epochs (loss 0.86 to 0.10). Those curves are the recorded training summary for this bundle.

`steering_policy.npz` is not that converged checkpoint. It is 252,219 float32 values drawn with seed 26 (hidden layers N(0, 0.02), last layer N(0, 0.35)), so `infer.py` can run the architecture. Do not read its printed angle as the 2.3° test result.
