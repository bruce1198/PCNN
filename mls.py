import math
from config import *
import json

class Prefetcher:

    def __init__(self, name):
        self.name = name
        self.slicing_blocks = [[] for i in range(total_device_num)]
        self.total_time = [0 for i in range(total_device_num)]
        self.json = {}

    def set_stride(self, stride):
        self.stride = stride
        self.json['stride'] = stride

    def set_filter(self, filter_size):
        self.filter_size = filter_size
        self.json['filter'] = filter_size

    def set_padding(self, padding):
        self.padding = padding
        self.json['padding'] = padding

    def set_input(self, input_size):
        self.input_size = input_size
        self.json['input'] = input_size

    def set_output(self, output_size):
        self.output_size = output_size
        self.json['output'] = output_size

    def set_channel(self, input_channel):
        self.input_channel = input_channel
        self.json['in_channel'] = input_channel

    def set_channel_out(self, output_channel):
        self.output_channel = output_channel
        self.json['out_channel'] = output_channel

    def set_layer_type(self, layer_type):
        self.layer_type = layer_type
        self.json['layers'] = layer_type

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
        self.json['devices'] = [{} for i in range(idx+1)]
        for blk in blocks:
            self.get_prefetch_index(block=blk, device_num=idx+1)
            
    # return prefetch begin, end index of the block's begin layer
    def get_prefetch_index(self, block, device_num):
        # print('block '+str(block[0])+' to '+str(block[1]))
        key = str(block[0])+','+str(block[1])
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
                    b[idx] = int(max(b[idx]*s-p, 0))
                    e[idx] = int(min(max(e[idx]*s-p+fs-1,0), i-1))
                # if idx == 0:
                #     print(b[idx], e[idx])
                
        for idx in range(device_num):
            self.json['devices'][idx][key] = [b[idx], e[idx]]
            # print('device '+str(idx)+' should prefetch date from '+str(b[idx])+' to '+str(e[idx]))

    def jsonify(self):
        return json.dumps(self.json, indent=4)
