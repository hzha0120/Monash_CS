from imutils.video import VideoStream
import cv2

class MaskDetect(object):
	def __init__(self,model):
		self.vs = VideoStream(src=0).start()
		self.model = model
	def __del__(self):
		cv2.destroyAllWindows()

	def predict_garbage(self,frame, model):
		res,idx = model.predict(frame)
		return (res,idx)

	def get_frame(self,model):
		frame = self.vs.read()
		frame = cv2.flip(frame, 1)
		(res,idx) = self.predict_garbage(frame,model)
		ret, jpeg = cv2.imencode('.jpg', res)
		return jpeg.tobytes()

