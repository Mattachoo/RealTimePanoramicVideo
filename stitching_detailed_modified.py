print(os.cpu_count())
        frame_skip = 1000
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        stitcher_time = ""
        fixed_H_time = ""
        hmatrixtime = ""
        seam_size = 50
        # factors = []
        # bounds = []
        out_1 = cv2.VideoWriter("output_regional_h.mp4", fourcc, 10.0, (1280, 720))
        # H = None
        caps = []
        validation_counter = 0
        for video in videos:
            caps.append((cv2.VideoCapture(video)))
        count = 0
        cams_up = True
        buffer = PriorityQueue()
        priorityList = []
        next_tag = 0
        startTime2 = time.time()
        self.validation_counter = 0
        self.validation_interval = 100
        initial = True