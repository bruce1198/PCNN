import torch
import math
def pre_cal_weight(idx, device_num, input_size, originw):
	size = originw.shape[0]
	size2 = originw.shape[1]
	input_size = int(input_size)
	avg = int(math.floor(input_size/device_num))
	total = avg
	mod = input_size % device_num
	start = 0
	for ii in range(idx):
		if ii < mod:
			start += avg+1
		else:
			start += avg
	if idx < mod:
		total += 1
	height = total
	stride = input_size * input_size
	height1 = int(size * height / input_size)
	w = torch.zeros(height1, size2)
	cnt = 0
	for i in range(start*input_size, size, stride):
		pos = cnt * height*input_size
		w[pos:pos+height*input_size, :] = originw[i:i+height*input_size, :]
		cnt += 1
	return w.transpose(0, 1)
	
def pre_cal_weight2(idx, device_num, originw1, originw2):
	size = originw1.shape[1]
	# print(size)
	b = int(idx*math.ceil(size/device_num))
	e = int(min((idx+1)*math.ceil(size/device_num), size))
	# print(b, e)
	w = originw1[:, b:e]
	w2 = originw2[b:e, :]
	return {
		'w1': w.transpose(0, 1),
		'w2': w2.transpose(0, 1)
	}