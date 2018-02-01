THUMOS CHALLENGE 2014
http://crcv.ucf.edu/THUMOS14/

------------------------------------------------------------

This folder contains the temporal annotations of action instances in 
the validation vides of 20 classes (list provided below).

The folder 'annotations' contains text files each including the 
temporal locations of instances of one class. Each row denotes 
one instance with the following format:

[video_name starting_time ending_time]

The starting_time and ending_time are in seconds (1 decimal 
point precision).

Note that one video may include actions of multiple classes 
and multiple instances of one action. The meta 
file (http://crcv.ucf.edu/THUMOS14/Validation_set/validation_set_meta.zip) 
specifies  what class(es) are observed in each video.

The file 'Ambiguous_val.txt' includes the temporal parts of the 
video which were hard to be labeled as action or no-action (e.g., the 
moments immediately before the actions starts). Such segments are 
taken out of the evaluations.

The file TH14_Temporal_annotations_validation_ViperXGTF contains the
Viper Annotation Tool raw XGTF file.

The following is the list of 20 action classes for which temporal 
annotation is available:

7 BaseballPitch
9 BasketballDunk
12 Billiards
21 CleanAndJerk
22 CliffDiving
23 CricketBowling
24 CricketShot
26 Diving
31 FrisbeeCatch
33 GolfSwing
36 HammerThrow
40 HighJump
45 JavelinThrow
51 LongJump
68 PoleVault
79 Shotput
85 SoccerPenalty
92 TennisSwing
93 ThrowDiscus
97 VolleyballSpiking
