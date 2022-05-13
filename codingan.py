# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#buat argumen parse dan parse argumen
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#menginisialisasi daftar label kelas yang dilatih untuk MobileNet SSD
#detect, lalu buat satu set warna kotak pembatas untuk setiap kela
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

IGNORE = set(["person"])

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#memuat model serial kami dari disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#menginisialisasi aliran video, biarkan sensor kamera melakukan pemanasan,
#dan menginisialisasi penghitung FPS
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

#loop di atas bingkai dari aliran video
while True:
	 #ambil bingkai dari aliran video berulir dan ubah ukurannya
  	#memiliki lebar maksimum 400 piksel
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	 #ambil dimensi bingkai dan ubah menjadi gumpalan
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#loop di atas deteksi
	for i in np.arange(0, detections.shape[2]):
		#ekstrak keyakinan(yaitu, probabilitas) yang terkait dengan
    	#prediksi
		confidence = detections[0, 0, i, 2]

		#memfilter deteksi lemah dengan memastikan ‘keyakinan’nya
    	#lebih besar dari keyakinan minimum
		if confidence > args["confidence"]:
			#ekstrak indeks label kelas dari
     		#’deteksi’

			 # jika label kelas yang diprediksi ada dalam kumpulan kelas
			# kami ingin mengabaikan lalu melewatkan deteksi
			if CLASSES[idx] in IGNORE:
				continue

     		#hitung koordinat(x,y) dari kotak pembatas untuk
      		#objek  
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#gambar prediksi dibingkai
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	#tunjukkan bingkai keluaran 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	 #jika tombol ‘q’ ditekan, keluar dari loop
	if key == ord("q"):
		break

	#perbaharui penghitung FPS
	fps.update()

#hentikan pengatur waktu dan tampilan informasi FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# USAGE
# python codingan.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel