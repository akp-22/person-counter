from personCountingModule import PersonCountingModule
import argparse

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", type=str,
		help="path to input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to output video file")
	ap.add_argument("-v", "--csvfile", type=str,
		help="path to csv file location")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum 'person' class probability for detection")
	ap.add_argument("-s", "--skip-frames", type=int, default=10,
		help=" Number of skip frames between detections for tracking")
	ap.add_argument("-f", "--fps", type=int, default=30,
		help=" Webcamera framerate")
	args = vars(ap.parse_args())

	pc = PersonCountingModule(args)
	pc.run()

main()
