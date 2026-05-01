import numpy as np
import RNN

class TimeRNN :
	def __init__(self, Wh, Wx, b, stateful=False) :
		'''
		Wh, Wx, b는 해당 블록에 포함되는 모든 RNN이 공유한다.
		stateful:Boolean은 현재 해당 블록이 은닉 상태를 가지는지를 저장한다.
		'''
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

		self.layers = None # T개의 RNN계층이 List로 저장될 예정

		self.stateful=stateful
		self.h, self.dh = None, None
		# Truncated BPTT인데 self.dh는 왜 가지는거지? : 이는 Seq2seq에서 사용한다.

	def set_state(self, h) :
		self.h = h

	def reset_state(self) :
		self.h = None

	def forward(self, xs) :
		Wx, Wh, b = self.params
		N, T, D = xs.shape
		D, H = Wx.shape

		self.layers = []
		hs = np.empty((N, T, H), dtype='f') # output이 저장될 메모리 공간 확보

		# stateful과 h이 연관성은 있지만, 이전 forward호출에서 h를 생성해놨더라도,
		# 다음 호출에서는 초기화해서 사용하고 싶을 경우가 있을 수 있음. 이 때, stateful=False로 하면 된다.
		if not self.stateful or self.h is None :
			self.h = np.zeros((N, H), dtype='f')
	
		for t in range(T) :
			layer = RNN(*self.params)
			self.h = layer.forward(xs[:, t, :], self.h) # 우변의 h = h_next, 좌변의 h = h_prev
			hs[:, t, :] = self.h
			self.layers.append(layer)
		
		return hs

	def backward(self, dhs) :
		Wx, Wh, b = self.params
		N, T, H = dhs.shape
		D, H = Wx.shape

		dxs = np.empty((N,T,D), dtype='f')
		dh = 0
		grads = [0, 0, 0]
		for t in reversed(range(T)) :
			layer = self.layers[t]
			dx, dh = layer.backward(dhs[:, t, :] + dh)
			dxs[:, t, :] = dx

			# weight들은 repeat노드를 거쳐서 각 layer에 전달되었기 때문에, 각 기울기를 합해준다.
			for i, grad in enumerate(layer.grads) :
				grads[i] += grad

		# 왜 굳이 이렇게 두번 할까? 캐싱 때문에 그런건가?
		for i, grad in enumerate(grads) :
			self.grads[i][...] = grad

		# 마지막 layer가 출력한 dh를 인스턴스 변수에 저장한다.
		self.dh = dh

		return dxs

