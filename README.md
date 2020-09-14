# Snapping Algorithm

snapping_algorithm can be called by the following:
python snap.py -v videos/ch08_20190805143300.mp4 -f 50

snap_sampling.py samples every five frames of the entire video for background subtraction. It can be called by the following:
python snap_sampling.py -v videos/ch08_20190805143300.mp4 -f 50

automated testing can be called by the following:
python snap_automated_testing.py -r areas_used.txt

note that all testing videos must be put in the snapping_algorithm/videos directory for the automated testing
