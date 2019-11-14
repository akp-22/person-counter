from centroidAssociation import CentroidAssociation
from trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import csv
import pdb


# Associate Detection module
from detectionModule import DetectionModule

class PersonCountingModule():

	def __init__(self, args):
		self.args = args
		self.totalFrames = 0
		self.countEnter = 0
		self.countExit = 0
		self.trackableObjects = {}

	def run(self):
		# Input webcam or video file
		if not self.args.get("input", False):
			rec = VideoStream(src=0).start()
			time.sleep(1.0)
			fps = self.args["fps"]
			
		else:
			rec = cv2.VideoCapture(self.args["input"])
			fps = rec.get(cv2.CAP_PROP_FPS)

		writer = None
		W = None
		H = None

		# Associate tracked object
		ct = CentroidAssociation(maxDisappeared=10, maxDistance=50)
		trackers = []
		
		# Run detection module
		ObjectDetection = DetectionModule('person', self.args["confidence"])
		ObjectDetection.load_model()
		
		# initialize the csv file 
		with open(self.args["csvfile"], 'w', newline='') as f:
			csv_writer = csv.writer(f)		
			fields=['Entry_count','Exit_count']
			csv_writer.writerow(fields)
		f.close()
		
		# loop over frames from the video
		while True:
			frame = rec.read()
			frame = frame[1] if self.args.get("input", False) else frame

			if self.args["input"] is not None and frame is None:
				break

			# resize the frame to process it faster
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# initialize video writer to disk
			if self.args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(self.args["output"], fourcc, 30,
					(W, H), True)

			rects = []

			# check the object detections for tracker
			if self.totalFrames % self.args["skip_frames"] == 0:
				trackers = []
				detections = ObjectDetection.inference(frame)
				#pdb.set_trace()
				print(self.totalFrames)		
	
				for i in np.arange(0, len(detections)):

					box = detections[i]
					(startX, startY, endX, endY) = box.astype("int")
					# Specify tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
					tracker.start_track(rgb, rect)
					trackers.append(tracker)

			else:
				self.get_tracked_boxes(trackers, rects, rgb)

			self.assosciate_tracked_objects(rects, ct, frame, H, W, writer)
			
			cv2.imshow("frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				 break
			
			# Updating count in csv file every second				
			if self.totalFrames % fps == 0:
				
				#write total count values each second to csv file
				with open(self.args["csvfile"], 'a',newline='') as f:
					csv_writer = csv.writer(f)	
					csv_writer.writerow([self.countEnter, self.countExit])
				
				# write the unique number of exits/entries happened in each second.
				
				#people_entered_per_sec = countExit - people_entered_per_sec
				#people_left_per_sec = countEnter - people_left_per_sec
				#with open(self.args["csvfile"], 'a') as f:
					#csv_writer = csv.writer(f)	
					#csv_writer.writerow([people_entered_per_sec, people_left_per_sec])
					#csv_writer.writerow([countExit, countEnter])
				
				f.close()

		if writer is not None:
			writer.release()

		if not self.args.get("input", False):
			rec.stop()
		else:
			rec.release()

		cv2.destroyAllWindows()

	def get_tracked_boxes(self, trackers, rects, rgb):
		for tracker in trackers:
			tracker.update(rgb)
			pos = tracker.get_position()
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			rects.append((startX, startY, endX, endY))
		
    # Associate/Disassociate tracked object based on centroid distance
	def assosciate_tracked_objects(self, rects, ct, frame, H, W, writer):
		objects = ct.update(rects)

		for (objectID, centroid) in objects.items():
			to = self.trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				if not to.counted:
					if direction < 0: 
						self.countExit += 1
						to.counted = True

					elif direction > 0:
						self.countEnter += 1
						to.counted = True

			self.trackableObjects[objectID] = to

		# Display the count of people entered & exited the store/frame.

		info = [
			("# Exited", self.countExit),
			("# Entered", self.countEnter),
		]
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (W - 170 , (i*20)+30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		# Bounding boxes on the tracked detections
		for rect in rects:
				(startX, startY, endX, endY) = rect
				cv2.rectangle(frame,(startX, startY),(endX, endY) ,(0, 0, 255), 2)
				
		if writer is not None:
			writer.write(frame)

		self.totalFrames += 1