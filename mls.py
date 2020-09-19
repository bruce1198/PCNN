import math
from config import *

class Prefetcher:

    def __init__(self, name):
        self.name = name
        self.slicing_blocks = [[] for i in range(total_device_num)]
        self.total_time = [0 for i in range(total_device_num)]
    def set_stride(self, stride):
        self.stride = stride
    def set_filter(self, filter_size):
        self.filter_size = filter_size
    def set_padding(self, padding):
        self.padding = padding
    def set_input(self, input_size):
        self.input_size = input_size
    def set_output(self, output_size):
        self.output_size = output_size
    def set_channel(self, input_channel):
        self.input_channel = input_channel
    def set_layer_type(self, layer_type):
        self.layer_type = layer_type
    def append_slicing_blocks(self, blocks, device_num, total_time):
        self.slicing_blocks[device_num] = blocks
        self.total_time[device_num] = total_time
    def get_fastest_slicing_blocks(self):
        min = self.total_time[0]
        idx = 0
        for i in range(total_device_num):
            if self.total_time[i] < min:
                min = self.total_time[i]
                idx = i
        return idx
    def prefetch(self):
        idx = self.get_fastest_slicing_blocks()
        blocks = self.slicing_blocks[idx]
        for blk in blocks:
            print('block: '+str(blk[0])+' to '+str(blk[1]))
            self.get_prefetch_index(block=blk, device_num=idx+1)
    # return prefetch begin, end index of the block's begin layer
    def get_prefetch_index(self, block, device_num):
        b = [0 for i in range(device_num)]
        e = [0 for i in range(device_num)]
        layer1 = block[0]
        layer2 = block[1]
        is_hybrid = False
        for layer in range(layer2, layer1-1, -1):
            for idx in range(device_num):
                # FL
                if self.layer_type[layer] == 'FL':
                    if is_hybrid:
                        c = self.input_channel[layer]
                        b[idx] = 0
                        e[idx] = int(c-1)
                    else:
                        # prev layer is pool or conv
                        if self.layer_type[layer-1] != 'FL':
                            o = self.output_size[layer-1]
                            b[idx] = int(idx*math.ceil(o/device_num))
                            e[idx] = int(min((idx+1)*math.ceil(o/device_num), o)-1)
                        # consecutive FL
                        else:
                            # dont care the prev layer @@
                            is_hybrid = True
                else:
                    fs = self.filter_size[layer]
                    s = self.stride[layer]
                    p = self.padding[layer]
                    i = self.input_size[layer]
                    o = self.output_size[layer]
                    # last layer
                    if layer == layer2:
                        b[idx] = int(idx*math.ceil(o/device_num))
                        e[idx] = int(min((idx+1)*math.ceil(o/device_num), o)-1)
                    # print(idx, b[idx], e[idx])
                    b[idx] = int(max(b[idx]*s-math.ceil(p/2), 0))
                    e[idx] = int(min(max(e[idx]*s-math.ceil(p/2)+fs-1,0), i-1))
                # if idx == 0:
                #     print(b[idx], e[idx])
                
        for idx in range(device_num):
            print('device '+str(idx)+' should prefetch date from '+str(b[idx])+' to '+str(e[idx]))
if __name__ == '__main__':
    blocks = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 8],
        [9, 10]
    ]
    stride = [4.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, '', '', '']
    padding = [1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, '', '', '']
    filter_size = [11.0, 3.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, '', '', '']
    input_size = [224.0, 55.0, 27.0, 27.0, 13.0, 13.0, 13.0, 13.0, '', '', '']
    output_size = [55.0, 27.0, 27.0, 13.0, 13.0, 13.0, 13.0, 6.0, '', '', '']
    layer_type = ['conv', 'pool', 'conv', 'pool', 'conv', 'conv', 'conv', 'pool', 'FL', 'FL', 'FL']