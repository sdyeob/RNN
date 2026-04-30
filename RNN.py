import numpy as np

class RNN :
	def __init__(self, Wx, Wh, b) :
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.cache = None

	# x, h는 mini-batch처리로 가정
	# Batch Size = N
	# 입력 벡터 차원 = D
	# 은닉 상태 벡터 차원 = H
	def forward(self, x, h_prev) :
		Wx, Wh, b = self.params
		temp = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
		h_next = np.tanh(temp)

		# Backprop시 사용됨
		self.cache = (x, h_prev, h_next)

	#forward시 h_next가 2개로 repeat되기 때문에, dh_next는 두 기울기를 더한 값이다.
	def backward(self, dh_next) :
		Wx, Wh, b = self.params
		x, h_prev, h_next = self.cache
		dt = dh_next * (1 - h_next ** 2)
		db = np.sum(dt,axis=0) # repeat되기 때문에, 모두 더한다.
		dWh = np.matmul(h_prev.T, dt)
		dh_prev = np.matmul(dt, Wh.T)
		dWx = np.matmul(x.T, dt)
		dx = np.matmul(dt, Wx.T)

		# grads[n]의 방식으로 하면, 그저 이 값을 가리키게 된다. 반면, 아래의 방식은
		# 할당받은 메모리에 값을 쓴다.
		self.grads[0][...] = dWx
		self.grads[1][...] = dWh
		self.grads[2][...] = db

		return dx, dh_prev

