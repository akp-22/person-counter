from tracker import KCFTracker
import copy

#-------------------------------------------------------------------------------

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
	# Create a tracker based on tracker name
	if trackerType == trackerTypes[0]:
		tracker = cv2.TrackerBoosting_create()
	elif trackerType == trackerTypes[1]: 
		tracker = cv2.TrackerMIL_create()
	elif trackerType == trackerTypes[2]:
		tracker = cv2.TrackerKCF_create()
	elif trackerType == trackerTypes[3]:
		tracker = cv2.TrackerTLD_create()
	elif trackerType == trackerTypes[4]:
		tracker = cv2.TrackerMedianFlow_create()
	elif trackerType == trackerTypes[5]:
		tracker = cv2.TrackerGOTURN_create()
	elif trackerType == trackerTypes[6]:
		tracker = cv2.TrackerMOSSE_create()
	elif trackerType == trackerTypes[7]:
		tracker = cv2.TrackerCSRT_create()
	else:
		tracker = None
		print('Incorrect tracker name')
		print('Available trackers are:')
		for t in trackerTypes:
			print(t)
	 
	return tracker


if args.tracker:
	#Images = Image_samples(files, image_size)
	images = img.astype(np.uint8)
	#images = cv2.resize(images ,(512,512))

	if tracker_initialized:
	###################### Tracker Update ###################
		img_sample = images
		timer = cv2.getTickCount()
		track_del = []
		#tracker_boxes = np.zeros((len(Trackers), 4))
		tracker_boxes = []
		cnt_tracker = 0
		for ind_track,tracker in enumerate(Trackers):
			bbox = tracker.update(img_sample)
			bbox = list(map(int, bbox))
			if all(val == 0 for val in 
			):
				track_del.append(ind_track)
				continue
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

			# Tracking success
			p1 = (int(bbox[0]), int(bbox[1]))
			p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
			cv2.rectangle(img_sample, p1, p2, (255, 0, 0), 2, 1)

			# Put FPS
			cv2.putText(img_sample, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
			tracker_boxes.append([int(bbox[0]) ,int(bbox[0] + bbox[2]), int(bbox[1]), int(bbox[1] + bbox[3])])
			cnt_tracker+=1
		tracker_boxes = np.array(tracker_boxes)
		for track_del_ind in track_del:
			tracker_removed = Trackers.pop(track_del_ind)
			print('tracker_removed --->' , tracker_removed)
		#cv2.imshow("Tracking", img_sample)
		cv2.imshow(filename, img_sample)
		cv2.waitKey(0)
		# Exit if ESC pressed
		# k = cv2.waitKey(0) & 0xff
		# if k == 27:
			# break
		cv2.destroyAllWindows()	

###################### Tracker Initilization ###################
	if not len(boxes) == 0:
		box_init = []
		if not tracker_initialized:
			for b in boxes:
				box_cord = get_box(images,b[1])
				box_init.append(box_cord)
		else:
			for box_track in CNN_boxes:
				overlap = jaccard_overlap(box_track , tracker_boxes)
				overlap_area = max(overlap)
				index_ax 	= np.argmax(overlap)
				if overlap_area < 0.4:
					#pdb.set_trace()
					box_init.append(np.array([box_track[0],box_track[2],(box_track[1]-box_track[0]),(box_track[3]-box_track[2])]))

		for Box in box_init:
			#pdb.set_trace()
			print('Initilization')					
			Trackers.append(KCFTracker(True, True, True)) # (hog, fixed_Window, multi_scale)
			#tracker = copy.copy(Tracker)
			#box_cord = get_box(img,bbox[1])
			Trackers[-1].init(Box, images)
		tracker_initialized = True

