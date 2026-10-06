# b1.1 manually annotated OBB subset of b1
Images DJI_0566..DJI_0578 (13), annotated by hand in X-AnyLabeling (json beside each image, copied unchanged from b1/images/train).
labels/train/*.txt are Ultralytics OBB (`cls x1 y1 x2 y2 x3 y3 x4 y4`, normalized) converted from those json files. Classes as in b1. b1 itself is untouched.
Note: the manual annotations use extra class names not in b1 (Non-Motorized-Van, Private-Passenger-Car, CNG, Other, Minivan, Bicycle, Pickup [vs PickUp]);
they are appended to the b1 list as ids 14+ in classes.txt/data.yaml with no remapping. 'Pickup' and 'PickUp' are kept distinct as annotated; merge if intended.
