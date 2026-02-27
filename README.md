# HappyWhaleClassifier

In this assignment, we will create a classifier for whales based on the Happy Whale Kaggle competition.

Due dates:
- Part 0: Feb 19th (un-graded)

- Part 1 and 2: Feb 27th

- Part 3: March 6th

Final Batches

Here is a comparison table of all 4 of my runs:
| Model Number | resolution | batch size| learning rate | epochs | Train Loss | Validation Accuracy | Validation Loss | Runtime|
| --- | --- | --- | --- |--- | --- | --- | --- |--- |
|Model 1| 32x32 | 48 | 1e-3 | 5 | 0.4408 | 73.96% | 0.6561 | 9m 22s|
|Model 1_1*| 244x244 | 48 | 1e-3 | 5 | 0.7991 | 64.84% | 0.90 | 1h 20m 29s|
|Model 2| 32x32 |60 | 1e-3 | 7 |0.3251 | 85.8333% |  0.3385 |12m 27s|
|Model 2_1| 244x244 |60 | 1e-3 | 7 | 0.6712| 65.83% | 0.8834 |1h 41m 40s|
|Model 3| 32x32 |10 | 1e-5 | 3 | 0.4878| 76.82% | 0.6347 |9m 40s|
|Model 3_1| 244x244 |10 | 1e-5 | 3 | 0.7681| 65.61% | 0.9384 |50m 38s|
|Model 4 |32x32 | 100 | 1e-5 | 2 | 1.051| 29.75%|1.6204|4m 10s| 4m 10s|
|Model 4_1 |244x244 | 100 | 1e-5 | 2 | 0.9634| 41.5% | 1.5921 |28m 14s|

*completed as part 1 in this assignment
