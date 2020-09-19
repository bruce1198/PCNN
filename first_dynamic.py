import xlrd
import numpy as np
import math
from config import *
import xlwt
from mls import Prefetcher

data = xlrd.open_workbook('DL_config.xlsx')

sheets = data.sheets()
# sheet1 = data.sheets()[2]          #通过索引顺序获取
# table = data.sheet_by_index(0) #通过索引顺序获取
# table = data.sheet_by_name(u'Sheet1')#通过名称获取

model_name = []
"""讀取excel設定檔進來"""
for sheet in sheets:
    # print("\n\n\n")
    tmp_layer_type = []
    tmp_input_size = []
    tmp_input_channel = []
    tmp_filter_size = []
    tmp_output_size = []
    tmp_output_channel = []
    tmp_stride = []
    tmp_padding_start = []
    tmp_padding_end = []
    model_name.append(sheet.name)
    if (sheet.nrows > maximum_row_num):
        maximum_row_num = sheet.nrows
    for i in range(1, sheet.nrows):
        tmp_layer_type.append(sheet.cell_value(i, 2))  # type
        tmp_input_size.append(sheet.cell_value(i, 3))  # input size
        tmp_input_channel.append(sheet.cell_value(i, 5))  # input channel
        tmp_filter_size.append(sheet.cell_value(i, 6))  # filter size
        tmp_output_size.append(sheet.cell_value(i, 8))  # output size
        tmp_output_channel.append(sheet.cell_value(i, 10))  # output channel
        tmp_stride.append(sheet.cell_value(i, 11))  # output channel
        tmp_padding_start.append(sheet.cell_value(i, 12))  # padding start
        tmp_padding_end.append(sheet.cell_value(i, 13))  # padding end
    layer_type.append(tmp_layer_type)
    input_size.append(tmp_input_size)
    input_channel.append(tmp_input_channel)
    filter_size.append(tmp_filter_size)
    output_size.append(tmp_output_size)
    output_channel.append(tmp_output_channel)
    stride.append(tmp_stride)
    padding_start.append(tmp_padding_start)
    padding_end.append(tmp_padding_end)
    """
    sheet_num+=1
    print(layer_type[sheet_num-1])
    print(input_size[sheet_num-1])
    print(input_channel[sheet_num-1])
    print(filter_size[sheet_num-1])
    print(output_size[sheet_num-1])
    print(output_channel[sheet_num-1])
    print(len(output_channel[sheet_num-1]))"""

model_length = len(sheets)

result_time = np.empty((model_length, total_device_num+1,
                        maximum_row_num, maximum_row_num))
transmission_time = np.empty(
    (model_length, total_device_num+1, maximum_row_num, maximum_row_num))
computation_time = np.empty(
    (model_length, total_device_num+1, maximum_row_num, maximum_row_num))
device_used = np.empty((model_length, total_device_num+1,
                        maximum_row_num, maximum_row_num))
parallelism_type = np.empty(
    (model_length, total_device_num+1, maximum_row_num, maximum_row_num), dtype=str)
result_time.fill(0)
transmission_time.fill(0)
computation_time.fill(0)

print (layer_type[1])


def cal_FLOPs(output_height, output_width, layer, layer_type, model_type):
    if (layer_type == "conv"):
        """print("2*")
        print(output_height)
        print(output_width)
        print(input_channel[model_type][layer])
        print(filter_size[model_type][layer])
        print("2+1)*")
        print(output_channel[model_type][layer])
        print(2*output_height*output_width*(input_channel[model_type][layer]*filter_size[model_type][layer]*filter_size[model_type][layer]+1)*output_channel[model_type][layer])"""
        return (2*output_height*output_width*(input_channel[model_type][layer]*filter_size[model_type][layer]*filter_size[model_type][layer]+1)*output_channel[model_type][layer])
    elif (layer_type == "FL"):
        """print("cal")
        print(output_height)
        print(output_width)
        print((2*output_height-1)*output_width)"""
        return (2*output_height-1)*output_width
    elif (layer_type == "pool"):
        return 0


def cal_comp_time(FLOPs):
    return (FLOPs/FLOPs_of_device)


def cal_trans_time(pre_output_transmission_size, output_transmission_size, input_transmission_size, device_num=1):
    """print("Total FLOPs = "+str(FLOPs))
    print("Computation time = "+str(FLOPs/FLOPs_of_device)+" s")
    print("Total Transmission Size = "+str(total_transmission_size))
    print("Transmission time = "+str(total_transmission_size/transmission_speed)+" s")
    print("\033[7;37;40mTotal time = "+str((FLOPs/FLOPs_of_device)+(total_transmission_size/transmission_speed))+" s\033[0m")"""
    # print(pre_output_transmission_size)
    # print(output_transmission_size)
    # ssprint(input_transmission_size)
    # total_transmission_size=0
    total_transmission_size = 0
    while(pre_output_transmission_size > 0):
        # print(pre_output_transmission_size)
        if(pre_output_transmission_size > 1448):
            total_transmission_size += 1488
            pre_output_transmission_size -= 1448
        else:
            total_transmission_size += (pre_output_transmission_size+40)
            pre_output_transmission_size = 0

    while(output_transmission_size > 0):
        if(output_transmission_size > 1448):
            total_transmission_size += 1488
            output_transmission_size -= 1448
        else:
            total_transmission_size += (output_transmission_size+40)
            output_transmission_size = 0

    while(input_transmission_size > 0):
        # print(input_transmission_size)
        if(input_transmission_size > 1448):
            total_transmission_size += 1488
            input_transmission_size -= 1448
        else:
            total_transmission_size += (input_transmission_size+40)
            input_transmission_size = 0
    return (total_transmission_size/(transmission_speed/device_num))


def slice_layer(model_type, start_layer, end_layer, device_num):
    # conv+pooling
    if(end_layer+1 != len(layer_type[model_type]) and layer_type[model_type][end_layer+1] == "pool"):
        #print ("Next layer "+str(end_layer+1)+" is Pooling")
        result_time[model_type][device_num][start_layer][end_layer] = np.finfo(
            np.float64).max
        return 0
    # pooling
    elif(layer_type[model_type][start_layer] == "pool"):
        result_time[model_type][device_num][start_layer][end_layer] = np.finfo(
            np.float64).max
        #print("This layer "+str(start_layer)+" is Pooling")
        return 1

    elif(start_layer == end_layer):  # 1 layer
        if(layer_type[model_type][end_layer] == "FL"):  # compare data and model parallelism
            #print("1 FL")
            """Data parallelism"""
            # calculate slice size
            end_layer_output_size = output_channel[model_type][end_layer]
            slice_output_size = output_channel[model_type][end_layer]
            end_layer_input_size = input_channel[model_type][end_layer]
            slice_input_size = math.ceil(end_layer_input_size/device_num)
            # calculate FLOPs
            FLOPs = cal_FLOPs(slice_input_size, slice_output_size,
                              end_layer, layer_type[model_type][end_layer], model_type)
            #print("input_size= "+str(slice_input_size))
            #print("output_size= "+str(slice_output_size))
            #print("Data Parallelism FLOPs= "+str(FLOPs))
            # calculate input transmission time
            input_transmission_size = slice_input_size*4
            # for k==1 adjustment
            if device_num == 1:
                if start_layer != 0:
                    input_transmission_size = 0
            #print("input transmission size= "+str(input_transmission_size)+" bytes")

            # calculate output transmission time
            # for k==1 adjustment
            output_transmission_size = slice_output_size*4
            if device_num == 1:
                if end_layer != len(layer_type[model_type])-1:
                    output_transmission_size = 0
            #print("output transmission size= "+str(output_transmission_size)+" bytes")

            # calculate total transmission time
            total_transmission_size = input_transmission_size+output_transmission_size
            #print("total transmission size= "+str(total_transmission_size)+" bytes")
            computation_time[model_type][device_num][start_layer][end_layer] = cal_comp_time(
                FLOPs)
            transmission_time[model_type][device_num][start_layer][end_layer] = cal_trans_time(
                0, input_transmission_size, output_transmission_size, device_num)
            result_time[model_type][device_num][start_layer][end_layer] = (
                transmission_time[model_type][device_num][start_layer][end_layer]+computation_time[model_type][device_num][start_layer][end_layer])

            # set parallelism type as data first, and would check it again
            parallelism_type[model_type][device_num][start_layer][end_layer] = "Data"
            """Model parallelism"""
            # calculate slice size
            end_layer_output_size = output_channel[model_type][end_layer]
            slice_output_size = math.ceil(end_layer_output_size/device_num)
            end_layer_input_size = input_channel[model_type][end_layer]
            slice_input_size = end_layer_input_size
            # calculate FLOPs
            FLOPs = cal_FLOPs(slice_input_size, slice_output_size,
                              end_layer, layer_type[model_type][end_layer], model_type)
            #print("input_size= "+str(slice_input_size))
            #print("output_size= "+str(slice_output_size))
            #print("Model Parallelism FLOPs= "+str(FLOPs))
            # calculate input transmission time
            input_transmission_size = slice_input_size*4
            # for k==1 adjustment
            if device_num == 1:
                if start_layer != 0:
                    input_transmission_size = 0
            #print("input transmission size= "+str(input_transmission_size)+" bytes")

            # calculate output transmission time
            # for k==1 adjustment
            output_transmission_size = slice_output_size*4
            if device_num == 1:
                if end_layer != len(layer_type[model_type])-1:
                    output_transmission_size = 0

            # calculate total transmission time

            total_transmission_size = input_transmission_size+output_transmission_size
            tmp_trans_time = cal_trans_time(
                0, input_transmission_size, output_transmission_size, device_num)
            tmp_comp_time = cal_comp_time(FLOPs)
            tmp_result_time = tmp_trans_time+tmp_comp_time
            #print("total transmission size= "+str(total_transmission_size)+" bytes")


            # check which scheme of parallelism is better
            # tmp_result_time=cal_result_time(FLOPs,0,input_transmission_size,output_transmission_size)
            if(tmp_result_time < result_time[model_type][device_num][start_layer][end_layer]):
                #print("Parallelism is faster")
                result_time[model_type][device_num][start_layer][end_layer] = tmp_result_time
                computation_time[model_type][device_num][start_layer][end_layer] = tmp_comp_time
                transmission_time[model_type][device_num][start_layer][end_layer] = tmp_trans_time
                # result_time[model_type][device_num][start_layer][end_layer]=(computation_time[model_type][device_num][start_layer][end_layer]+transmission_time[model_type][device_num][start_layer][end_layer])

                parallelism_type[model_type][device_num][start_layer][end_layer] = "Model"
            # result_time[model_type][device_num][start_layer][end_layer]=min(result_time[model_type][device_num][start_layer][end_layer],cal_result_time(FLOPs,total_transmission_size))
            # Compare two parallelism way
            return 0

        elif(layer_type[model_type][end_layer] == "conv"):  # data parallelism
            #print("NO FL")
            #print("Device_num= "+str(device_num))
            input_height = [[0, 0] for i in range(device_num+1)]
            output_height = [[0, 0] for i in range(device_num+1)]

            pre_output_transmission_size = [0 for i in range(device_num+1)]
            input_transmission_size = [0 for i in range(device_num+1)]
            output_transmission_size = [0 for i in range(device_num+1)]
            transmit_data = []
            if(start_layer > 0):
                no_padding_start = padding_start[model_type][start_layer]
                # no_padding_end=
            for device in range(1, device_num+1):
                # calculate slice size
                end_layer_output_size = output_size[model_type][end_layer]
                # output index, formula (9)
                output_height[device][0] = partition[model_type][device_num][end_layer][device][0]
                output_height[device][1] = partition[model_type][device_num][end_layer][device][1]
                output_width = end_layer_output_size  # original

                #print("output_size= "+str(output_height)+" * "+str(output_width))
                # input index, formula (10)
                input_height[device][0] = max(int(
                    output_height[device][0]*stride[model_type][end_layer]-padding_start[model_type][end_layer]), 0)
                input_height[device][1] = min(max(int(output_height[device][1]*stride[model_type][end_layer]+filter_size[model_type]
                                                      [end_layer]-1-padding_start[model_type][end_layer]), 0), input_size[model_type][end_layer]-1)
                # print(input_height[device])
                input_width = input_size[model_type][end_layer]
                if(model_type == 0 and start_layer == 25):
                    input_transmission_size[device] += (
                        (input_height[device][1]-input_height[device][0]+1)*19*256*4)
                if(start_layer == 0):
                    input_transmission_size[device] += (
                        input_height[device][1]-input_height[device][0]+1)*input_width*input_channel[model_type][end_layer]*4
                else:
                    size_count = 0
                    # print(no_padding_start)
                    # print(input_height[device][0],input_height[device][1])
                    for i in range(int(input_height[device][0]), int(input_height[device][1]+1)):

                        #print ("data count="+str(i))
                        if(i >= partition[model_type][device_num][start_layer-1][device][0] and i <= partition[model_type][device_num][start_layer-1][device][1]):
                            pass
                        else:
                            # print("count")
                            size_count += 1
                            transmit_data.append(i)
                    # print(partition[model_type][device_num][start_layer-1][device])
                    # print(input_height[device])
                    # print("siez_count="+str(size_count))
                    input_transmission_size[device] += size_count * \
                        input_width*input_channel[model_type][end_layer]*4
            transmit_data = list(set(transmit_data))
            # print(transmit_data)
            for device in range(1, device_num+1):
                tmp_count = []
                if(start_layer == 0):
                    pass
                else:
                    for x in transmit_data:
                        # print(partition[model_type][device_num][start_layer-1][device])
                        # print(x)
                        if(x >= partition[model_type][device_num][start_layer-1][device][0] and x <= partition[model_type][device_num][start_layer-1][device][1]):
                            # print("count")
                            pre_output_transmission_size[device] += (
                                input_width*input_channel[model_type][end_layer]*4)
                if(end_layer == (len(layer_type[model_type])-1)):
                    output_transmission_size[device] = (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                elif(layer_type[model_type][end_layer+1] == "FL"):
                    output_transmission_size[device] = (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                else:
                    output_transmission_size[device] = 0
            # print(pre_output_transmission_size)
            # print(output_transmission_size)

            # calculate time of each
            for device in range(1, device_num+1):
                # calculate computation time
                FLOPs = cal_FLOPs((output_height[device][1]-output_height[device][0]+1),
                                  output_width, end_layer, layer_type[model_type][end_layer], model_type)
                if device_num == 1:
                    if start_layer != 0:
                        input_transmission_size[device] = 0
                #print("input transmission size= "+str(input_transmission_size)+" bytes")

                # calculate output transmission time
                # for k==1 adjustment
                if device_num == 1:
                    if end_layer != len(layer_type[model_type])-1:
                        output_transmission_size[device] = 0

                total_transmission_size = pre_output_transmission_size[device] + \
                    output_transmission_size[device] + \
                    input_transmission_size[device]

                tmp_comp_time = cal_comp_time(FLOPs)

                # print(total_transmission_size)
                # print(pre_output_transmission_size)
                tmp_trans_time = cal_trans_time(
                    pre_output_transmission_size[device], output_transmission_size[device], input_transmission_size[device], device_num)
                tmp_result_time = tmp_trans_time+tmp_comp_time
                # print(tmp_trans_time)
                # print(FLOPs)
                if(tmp_result_time > result_time[model_type][device_num][start_layer][end_layer]):
                    computation_time[model_type][device_num][start_layer][end_layer] = tmp_comp_time
                    transmission_time[model_type][device_num][start_layer][end_layer] = tmp_trans_time
                    result_time[model_type][device_num][start_layer][end_layer] = tmp_result_time
                    # result_time[model_type][device_num][start_layer][end_layer]=(computation_time[model_type][device_num][start_layer][end_layer]+transmission_time[model_type][device_num][start_layer][end_layer])

                # result_time[model_type][device_num][start_layer][end_layer]=max(result_time[model_type][device_num][start_layer][end_layer],cal_result_time(FLOPs,pre_output_transmission_size[device],output_transmission_size[device],input_transmission_size[device]))
                # print(result_time[model_type][device_num][start_layer][end_layer])
            return 0
        else:
            print("\033[1;31;40mThere is otrher in Case 1: " +
                  layer_type[model_type][end_layer]+"\033[0m")
            return 0

    elif((end_layer-start_layer) == 1):  # 2 layers
        if(layer_type[model_type][end_layer] == "FL" and layer_type[model_type][end_layer-1] != "FL"):
            #print("1 FL")
            result_time[model_type][device_num][start_layer][end_layer] = np.finfo(
                np.float64).max
            print("\033[1;31;40mNo this situation!\033[0m")
            #print("No this situation!")
            return 0

        elif(layer_type[model_type][end_layer] == "FL" and layer_type[model_type][end_layer-1] == "FL"):  # model->data
            #print("2 FL")
            # Model to Data Paralleslim
            FLOPs = 0
            for layer in range(end_layer, start_layer-1, -1):
                if(layer == end_layer):  # Data Parallelism
                    end_layer_output_size = output_channel[model_type][layer]
                    slice_output_size = end_layer_output_size
                    end_layer_input_size = input_channel[model_type][layer]
                    slice_input_size = math.ceil(
                        end_layer_input_size/device_num)
                    output_transmission_size = slice_output_size*4
                    #print("output transmission size= "+str(output_transmission_size)+" bytes")
                    # calculate FLOPs
                    # FLOPs+=cal_FLOPs(slice_input_size,slice_output_size,layer,layer_type[model_type][layer])
                else:  # Model Parallelism
                    slice_output_size = slice_input_size
                    slice_input_size = input_channel[model_type][layer]
                    input_transmission_size = slice_input_size*4
                    #print("input transmission size= "+str(input_transmission_size)+" bytes")
                FLOPs += cal_FLOPs(slice_input_size, slice_output_size,
                                   layer, layer_type[model_type][layer], model_type)

            total_transmission_size = input_transmission_size+output_transmission_size
            #print("total transmission size= "+str(total_transmission_size)+" bytes")

            # result_time[model_type][device_num][start_layer][end_layer]=cal_result_time(FLOPs,0,input_transmission_size,output_transmission_size)
            computation_time[model_type][device_num][start_layer][end_layer] = cal_comp_time(
                FLOPs)
            transmission_time[model_type][device_num][start_layer][end_layer] = cal_trans_time(
                0, input_transmission_size, output_transmission_size, device_num)
            result_time[model_type][device_num][start_layer][end_layer] = (
                transmission_time[model_type][device_num][start_layer][end_layer]+computation_time[model_type][device_num][start_layer][end_layer])
            return 0
        else:  # No FL
            #print("No FL")
            # Data Paralleslim
            input_height = [[0, 0] for i in range(device_num+1)]
            output_height = [[0, 0] for i in range(device_num+1)]
            slice_output_height = [[0, 0] for i in range(device_num+1)]
            slice_input_height = [[0, 0] for i in range(device_num+1)]
            # print(output_height)
            pre_output_transmission_size = [0 for i in range(device_num+1)]
            input_transmission_size = [0 for i in range(device_num+1)]
            output_transmission_size = [0 for i in range(device_num+1)]
            transmit_data = []
            FLOPs = [0 for i in range(device_num+1)]
            if(start_layer > 0):
                no_padding_start = padding_start[model_type][start_layer]
                # no_padding_end=
            for device in range(1, device_num+1):
                #print("Device= "+str(device))
                for layer in range(end_layer, start_layer-1, -1):
                    #print("Layer = "+str(layer))
                    if(layer == end_layer):
                        end_layer_output_size = output_size[model_type][layer]
                        # print("In")
                        output_height[device][0] = partition[model_type][device_num][layer][device][0]
                        output_height[device][1] = partition[model_type][device_num][layer][device][1]
                        output_width = end_layer_output_size  # original
                        slice_input_height[device][0] = max(int(
                            output_height[device][0]*stride[model_type][layer]-padding_start[model_type][layer]), 0)
                        slice_input_height[device][1] = min(max(int(output_height[device][1]*stride[model_type][layer] +
                                                                    filter_size[model_type][layer]-1-padding_start[model_type][layer]), 0), input_size[model_type][layer]-1)
                        # slice_input_height[device][0]=output_height[device][0]*stride[model_type][layer]
                        # slice_input_height[device][1]=output_height[device][1]*stride[model_type][layer]+filter_size[model_type][layer]-1
                        slice_input_width = input_size[model_type][layer]
                        # print(FLOPs[device])
                        FLOPs[device] += cal_FLOPs((output_height[device][1]-output_height[device][0]+1),
                                                   output_width, layer, layer_type[model_type][layer], model_type)
                        # print(output_height[device])
                        # print(FLOPs[device])
                        # print(stride[model_type][layer])
                        # print(filter_size[model_type][layer])
                        # print(slice_input_height[device])
                        # output_transmission_size+=slice_output_height*slice_output_width*output_channel[model_type][layer]*4
                        #print("output transmission size= "+str(output_transmission_size)+" bytes")
                    else:
                        slice_output_height[device][0] = slice_input_height[device][0]
                        slice_output_height[device][1] = slice_input_height[device][1]
                        slice_output_width = output_size[model_type][layer]
                        slice_input_height[device][0] = max(int(
                            slice_output_height[device][0]*stride[model_type][layer]-padding_start[model_type][layer]), 0)
                        slice_input_height[device][1] = min(max(int(slice_output_height[device][1]*stride[model_type][layer] +
                                                                    filter_size[model_type][layer]-1-padding_start[model_type][layer]), 0), input_size[model_type][layer]-1)
                        slice_input_width = input_size[model_type][layer]
                        # print(slice_output_height[device])
                        # print(slice_input_height[device])
                        FLOPs[device] += cal_FLOPs((slice_output_height[device][1]-slice_output_height[device][0]+1),
                                                   slice_output_width, layer, layer_type[model_type][layer], model_type)
                        # print(FLOPs[device])
                    if(model_type == 0 and layer == 16):
                        tmp_output_height = (
                            partition[model_type][device_num][layer][device][1]-partition[model_type][device_num][layer][device][0]+1)
                        # original conv26 of YOLOv2
                        FLOPs[device] += (2*tmp_output_height *
                                          slice_output_width*(512*1*1+1)*64)
                        output_transmission_size[device] += ((slice_output_height[device][1]-slice_output_height[device]
                                                              [0]+1)*slice_output_width*output_channel[model_type][layer]*4)
                    if(model_type == 0 and layer == 25):
                        input_transmission_size[device] += (
                            (slice_input_height[device][1]-slice_input_height[device][0]+1)*19*256*4)
                if(start_layer == 0):
                    # print("in")
                    input_transmission_size[device] += (slice_input_height[device][1]-slice_input_height[device]
                                                        [0]+1)*slice_input_width*input_channel[model_type][start_layer]*4
                else:
                    size_count = 0
                    # print(no_padding_start)
                    for i in range(int(slice_input_height[device][0]), int(slice_input_height[device][1])+1):
                        #print ("data count="+str(i))
                        if(i >= partition[model_type][device_num][start_layer-1][device][0] and i <= partition[model_type][device_num][start_layer-1][device][1]):
                            pass
                        else:
                            # print("count")
                            size_count += 1
                            transmit_data.append(i)
                    # print(partition[model_type][device_num][start_layer-1][device])
                    # print(slice_input_height[device])
                    # print("siez_count="+str(size_count))
                    # print(size_count)
                    # print(input_transmission_size[device])
                    # print("before")
                    # print(input_transmission_size[device])
                    input_transmission_size[device] += size_count * \
                        slice_input_width * \
                        input_channel[model_type][start_layer]*4
                    # print(input_transmission_size[device])
            transmit_data = list(set(transmit_data))
            # print(transmit_data)
            # calculate previous output transmission size
            for device in range(1, device_num+1):
                tmp_count = []
                if(start_layer == 0):
                    pass
                else:
                    for x in transmit_data:
                        if(x >= partition[model_type][device_num][start_layer-1][device][0] and x <= partition[model_type][device_num][start_layer-1][device][1]):
                            pre_output_transmission_size[device] += (
                                slice_input_width*input_channel[model_type][start_layer]*4)
                if(end_layer == (len(layer_type[model_type])-1)):
                    output_transmission_size[device] += (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                elif(layer_type[model_type][end_layer+1] == "FL"):
                    output_transmission_size[device] += (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                else:
                    pass
                    # output_transmission_size[device]=0
            # print(pre_output_transmission_size)
            # print(output_transmission_size)

            # calculate time of each device
            for device in range(1, device_num+1):
                # calculate computation time
                # FLOPs=cal_FLOPs((output_height[device][1]-output_height[device][0]+1),output_width,end_layer,layer_type[model_type][end_layer],model_type)
                total_transmission_size = pre_output_transmission_size[device] + \
                    output_transmission_size[device] + \
                    input_transmission_size[device]
                # print(total_transmission_size)
                # print(input_transmission_size[device])
                tmp_comp_time = cal_comp_time(FLOPs[device])
                tmp_trans_time = cal_trans_time(
                    pre_output_transmission_size[device], output_transmission_size[device], input_transmission_size[device], device_num)
                tmp_result_time = tmp_trans_time+tmp_comp_time
                if (tmp_result_time > result_time[model_type][device_num][start_layer][end_layer]):
                    computation_time[model_type][device_num][start_layer][end_layer] = tmp_comp_time
                    transmission_time[model_type][device_num][start_layer][end_layer] = tmp_trans_time
                    result_time[model_type][device_num][start_layer][end_layer] = tmp_result_time
                # result_time[model_type][device_num][start_layer][end_layer]=max(result_time[model_type][device_num][start_layer][end_layer],cal_result_time(FLOPs[device],pre_output_transmission_size[device],output_transmission_size[device],input_transmission_size[device]))
                # print(result_time[model_type][device_num][start_layer][end_layer])
            return 0
        """else:
            print("\033[1;31;40mThere is otrher in Case 2: " + layer_type[model_type][end_layer]+"\033[0m")
            #print("NO FL")
            return 0"""
    elif((end_layer-start_layer) >= 2):
        if(layer_type[model_type][end_layer] == "FL" and layer_type[model_type][end_layer-1] != "FL" and layer_type[model_type][end_layer-2] != "FL"):  # data parallelism
            #print("1 FL")
            input_height = [[0, 0] for i in range(device_num+1)]
            output_height = [[0, 0] for i in range(device_num+1)]
            slice_output_height = [[0, 0] for i in range(device_num+1)]
            slice_input_height = [[0, 0] for i in range(device_num+1)]
            pre_output_transmission_size = [0 for i in range(device_num+1)]
            input_transmission_size = [0 for i in range(device_num+1)]
            output_transmission_size = [0 for i in range(device_num+1)]
            transmit_data = []
            FLOPs = [0 for i in range(device_num+1)]
            if(start_layer > 0):
                no_padding_start = padding_start[model_type][start_layer]
                # no_padding_end=
            for device in range(1, device_num+1):
                #print("Device= "+str(device))
                for layer in range(end_layer, start_layer-1, -1):
                    #print("Layer = "+str(layer))
                    if(layer == end_layer):
                        end_layer_output_size = output_size[model_type][layer-1]
                        slice_input_height[device][0] = partition[model_type][device_num][layer-1][device][0]
                        slice_input_height[device][1] = partition[model_type][device_num][layer-1][device][1]
                        # output_width = end_layer_output_size #original
                        slice_input_width = end_layer_output_size
                        slice_input_channel = output_channel[model_type][layer-1]
                        # print(slice_input_width)
                        slice_input_size = (
                            slice_input_height[device][1]-slice_input_height[device][0]+1)*slice_input_width*slice_input_channel
                        slice_output_size = output_channel[model_type][layer]
                        output_transmission_size[device] += slice_output_size*4
                        FLOPs[device] += cal_FLOPs(slice_input_size, slice_output_size,
                                                   layer, layer_type[model_type][layer], model_type)
                    else:
                        slice_output_height[device][0] = slice_input_height[device][0]
                        slice_output_height[device][1] = slice_input_height[device][1]
                        # print(layer)
                        # print(slice_input_height[device])
                        # print(stride[model_type][layer])
                        slice_output_width = output_size[model_type][layer]
                        slice_input_height[device][0] = max(int(
                            slice_output_height[device][0]*stride[model_type][layer]-padding_start[model_type][layer]), 0)
                        # print(slice_output_height[device][1]*stride[model_type][layer]+filter_size[model_type][layer]-1)
                        # print(stride[model_type][layer])
                        # print(filter_size[model_type][layer])
                        # if(layer!=0):
                        # slice_input_height[device][1]=min(max(int(slice_output_height[device][1]*stride[model_type][layer]+filter_size[model_type][layer]-1-padding_start[model_type][layer]),0),output_size[model_type][layer-1]-1)
                        # else:
                        slice_input_height[device][1] = min(max(int(slice_output_height[device][1]*stride[model_type][layer] +
                                                                    filter_size[model_type][layer]-1-padding_start[model_type][layer]), 0), input_size[model_type][layer]-1)
                        # slice_input_width=min((slice_output_width-1)*stride[model_type][layer]+filter_size[model_type][layer],input_size[model_type][layer])
                        slice_input_width = input_size[model_type][layer]
                        FLOPs[device] += cal_FLOPs((slice_output_height[device][1]-slice_output_height[device][0]+1),
                                                   slice_output_width, layer, layer_type[model_type][layer], model_type)
                        # print(slice_input_height[device])
                        # print(slice_output_height[device])
                        # print(slice_output_width)
                        # print(FLOPs[device])
                        # print(FLOPs[device])
                    if(model_type == 0 and layer == 16 and end_layer != 16):
                        output_transmission_size[device] += ((slice_output_height[device][1]-slice_output_height[device]
                                                              [0]+1)*slice_output_width*output_channel[model_type][layer]*4)
            # calculate input transmission size
            for device in range(1, device_num+1):
                if(start_layer == 0):
                    # print(slice_output_height[device])
                    # print(slice_input_height[device])
                    # print()
                    input_transmission_size[device] = (
                        slice_input_height[device][1]-slice_input_height[device][0]+1)*slice_input_width*input_channel[model_type][start_layer]*4
                else:
                    size_count = 0
                    # print(no_padding_start)
                    # print(slice_input_height[device])
                    for i in range(int(slice_input_height[device][0]), int(slice_input_height[device][1]+1)):
                        #print ("data count="+str(i))
                        if(i >= partition[model_type][device_num][start_layer-1][device][0] and i <= partition[model_type][device_num][start_layer-1][device][1]):
                            pass
                        else:
                            # print("count")
                            size_count += 1
                            transmit_data.append(i)
                    input_transmission_size[device] = size_count * \
                        slice_input_width * \
                        input_channel[model_type][start_layer]*4
            transmit_data = list(set(transmit_data))
            # print(transmit_data)
            # calculate previous output transmission size
            for device in range(1, device_num+1):
                #print("Device= "+str(device))
                tmp_count = []
                if(start_layer == 0):
                    pass
                else:
                    for x in transmit_data:
                        if(x >= partition[model_type][device_num][start_layer-1][device][0] and x <= partition[model_type][device_num][start_layer-1][device][1]):
                            #print("x = "+str(x))
                            pre_output_transmission_size[device] += (
                                slice_input_width*input_channel[model_type][start_layer]*4)
                if(end_layer == (len(layer_type[model_type])-1)):
                    output_transmission_size[device] += (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                else:
                    pass
                    # output_transmission_size[device]=0
            # print(input_transmission_size)
            # print(pre_output_transmission_size)
            # print(output_transmission_size)

            # calculate time of each device
            for device in range(1, device_num+1):
                # calculate computation time
                # FLOPs=cal_FLOPs((output_height[device][1]-output_height[device][0]+1),output_width,end_layer,layer_type[model_type][end_layer],model_type)
                total_transmission_size = pre_output_transmission_size[device] + \
                    output_transmission_size[device] + \
                    input_transmission_size[device]
                tmp_comp_time = cal_comp_time(FLOPs[device])
                tmp_trans_time = cal_trans_time(
                    pre_output_transmission_size[device], output_transmission_size[device], input_transmission_size[device], device_num)
                tmp_result_time = tmp_trans_time+tmp_comp_time
                if (tmp_result_time > result_time[model_type][device_num][start_layer][end_layer]):
                    computation_time[model_type][device_num][start_layer][end_layer] = tmp_comp_time
                    transmission_time[model_type][device_num][start_layer][end_layer] = tmp_trans_time
                    result_time[model_type][device_num][start_layer][end_layer] = tmp_result_time
                # result_time[model_type][device_num][start_layer][end_layer]=max(result_time[model_type][device_num][start_layer][end_layer],cal_result_time(FLOPs[device],pre_output_transmission_size[device],output_transmission_size[device],input_transmission_size[device]))
                # print(result_time[model_type][device_num][start_layer][end_layer])
            return 0
        elif(layer_type[model_type][end_layer] == "FL" and layer_type[model_type][end_layer-1] == "FL" and layer_type[model_type][end_layer-2] != "FL"):  # data parallelism
            #print("2 FL")
            result_time[model_type][device_num][start_layer][end_layer] = np.finfo(
                np.float64).max
            #print("Cannot calculate for 2FL")

            return 0
        elif(layer_type[model_type][end_layer] == "FL" and layer_type[model_type][end_layer-1] == "FL" and layer_type[model_type][end_layer-2] == "FL"):  # fix compare
            #print("3 FL")
            result_time[model_type][device_num][start_layer][end_layer] = np.finfo(
                np.float64).max
            #print("Cannot calculate for 3FL")
            return 0
        else:  # No FL
            #print("NO FL")
            # FLOPs=0
            # output_transmission_size=0
            input_height = [[0, 0] for i in range(device_num+1)]
            output_height = [[0, 0] for i in range(device_num+1)]
            slice_output_height = [[0, 0] for i in range(device_num+1)]
            slice_input_height = [[0, 0] for i in range(device_num+1)]
            # print(output_height)
            pre_output_transmission_size = [0 for i in range(device_num+1)]
            input_transmission_size = [0 for i in range(device_num+1)]
            output_transmission_size = [0 for i in range(device_num+1)]
            transmit_data = []
            FLOPs = [0 for i in range(device_num+1)]
            if(start_layer > 0):
                no_padding_start = padding_start[model_type][start_layer]
            for device in range(1, device_num+1):
                #print("Device= "+str(device))
                for layer in range(end_layer, start_layer-1, -1):
                    #print("Layer = "+str(layer))
                    if(layer == end_layer):
                        end_layer_output_size = output_size[model_type][layer]
                        # print(device)
                        output_height[device][0] = partition[model_type][device_num][layer][device][0]
                        output_height[device][1] = partition[model_type][device_num][layer][device][1]
                        output_width = end_layer_output_size  # original
                        # slice_input_height[device][0]=output_height[device][0]*stride[model_type][layer]
                        # slice_input_height[device][1]=output_height[device][1]*stride[model_type][layer]+filter_size[model_type][layer]-1
                        slice_input_height[device][0] = max(int(
                            output_height[device][0]*stride[model_type][layer]-padding_start[model_type][layer]), 0)
                        slice_input_height[device][1] = min(max(int(output_height[device][1]*stride[model_type][layer] +
                                                                    filter_size[model_type][layer]-1-padding_start[model_type][layer]), 0), input_size[model_type][layer]-1)
                        slice_input_width = input_size[model_type][layer]
                        FLOPs[device] += cal_FLOPs((output_height[device][1]-output_height[device][0]+1),
                                                   output_width, layer, layer_type[model_type][layer], model_type)
                        # print(output_height[device])
                        # print(FLOPs[device])
                        # print(slice_input_height[device])
                    else:
                        slice_output_height[device][0] = slice_input_height[device][0]
                        slice_output_height[device][1] = slice_input_height[device][1]
                        slice_output_width = output_size[model_type][layer]
                        slice_input_height[device][0] = max(int(
                            slice_output_height[device][0]*stride[model_type][layer]-padding_start[model_type][layer]), 0)
                        slice_input_height[device][1] = min(max(int(slice_output_height[device][1]*stride[model_type][layer] +
                                                                    filter_size[model_type][layer]-1-padding_start[model_type][layer]), 0), input_size[model_type][layer]-1)
                        slice_input_width = input_size[model_type][layer]
                        # print(slice_output_height[device])
                        # print(slice_input_height[device])
                        FLOPs[device] += cal_FLOPs((slice_output_height[device][1]-slice_output_height[device][0]+1),
                                                   slice_output_width, layer, layer_type[model_type][layer], model_type)
                        # print(FLOPs[device])
                    if(model_type == 0 and layer == 16):
                        tmp_output_height = (
                            partition[model_type][device_num][layer][device][1]-partition[model_type][device_num][layer][device][0]+1)
                        # original conv26 of YOLOv2
                        FLOPs[device] += (2*tmp_output_height *
                                          slice_output_width*(512*1*1+1)*64)
                        output_transmission_size[device] += ((slice_output_height[device][1]-slice_output_height[device]
                                                              [0]+1)*slice_output_width*output_channel[model_type][layer]*4)
                    if(model_type == 0 and layer == 25):
                        input_transmission_size[device] += (
                            (slice_input_height[device][1]-slice_input_height[device][0]+1)*19*256*4)
            # calculate input transmission size
            for device in range(1, device_num+1):
                if(start_layer == 0):
                    input_transmission_size[device] += (slice_input_height[device][1]-slice_input_height[device]
                                                        [0]+1)*slice_input_width*input_channel[model_type][start_layer]*4
                else:
                    size_count = 0
                    # print(no_padding_start)
                    for i in range(int(slice_input_height[device][0]), int(slice_input_height[device][1])+1):
                        #print ("data count="+str(i))
                        if(i >= partition[model_type][device_num][start_layer-1][device][0] and i <= partition[model_type][device_num][start_layer-1][device][1]):
                            pass
                        else:
                            # print("count")
                            size_count += 1
                            transmit_data.append(i)
                    # print(partition[model_type][device_num][start_layer-1][device])
                    # print(slice_input_height[device])
                    # print("siez_count="+str(size_count))
                    input_transmission_size[device] += size_count * \
                        slice_input_width * \
                        input_channel[model_type][start_layer]*4
            transmit_data = list(set(transmit_data))
            # print(transmit_data)
            # calculate previous output transmission size
            for device in range(1, device_num+1):
                tmp_count = []
                if(start_layer == 0):
                    pass
                else:
                    for x in transmit_data:
                        if(x >= partition[model_type][device_num][start_layer-1][device][0] and x <= partition[model_type][device_num][start_layer-1][device][1]):
                            pre_output_transmission_size[device] += (
                                slice_input_width*input_channel[model_type][start_layer]*4)
                if(end_layer == (len(layer_type[model_type])-1)):
                    output_transmission_size[device] += (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                elif(layer_type[model_type][end_layer+1] == "FL"):
                    output_transmission_size[device] += (
                        output_height[device][1]-output_height[device][0]+1)*output_width*output_channel[model_type][end_layer]*4
                else:
                    output_transmission_size[device] = 0
            # print(pre_output_transmission_size)
            # print(output_transmission_size)

            # calculate time of each device
            for device in range(1, device_num+1):
                # calculate computation time
                # FLOPs=cal_FLOPs((output_height[device][1]-output_height[device][0]+1),output_width,end_layer,layer_type[model_type][end_layer],model_type)
                total_transmission_size = pre_output_transmission_size[device] + \
                    output_transmission_size[device] + \
                    input_transmission_size[device]
                tmp_comp_time = cal_comp_time(FLOPs[device])
                tmp_trans_time = cal_trans_time(
                    pre_output_transmission_size[device], output_transmission_size[device], input_transmission_size[device], device_num)
                tmp_result_time = tmp_trans_time+tmp_comp_time
                if (tmp_result_time > result_time[model_type][device_num][start_layer][end_layer]):
                    computation_time[model_type][device_num][start_layer][end_layer] = tmp_comp_time
                    transmission_time[model_type][device_num][start_layer][end_layer] = tmp_trans_time
                    result_time[model_type][device_num][start_layer][end_layer] = tmp_result_time
                # result_time[model_type][device_num][start_layer][end_layer]=max(result_time[model_type][device_num][start_layer][end_layer],cal_result_time(FLOPs[device],pre_output_transmission_size[device],output_transmission_size[device],input_transmission_size[device]))
                # print(result_time[model_type][device_num][start_layer][end_layer])
            return 0


partition = np.empty((model_length, total_device_num+1,
                      maximum_row_num, total_device_num+1, 2))
tmp_partition = [0 for i in range(total_device_num+1)]
# preprofile the
for device in range(1, total_device_num+1):
    # print("\nDevcie"+str(device))
    for model_type in range(0, model_length):
        for layer in range(0, len(layer_type[model_type])):
            # print(layer)
            if(layer_type[model_type][layer] != "FL"):
                floor_data = math.floor(output_size[model_type][layer]/device)
                for i in range(1, device+1):
                    tmp_partition[i] = floor_data
                left_data = output_size[model_type][layer]-floor_data*device
                i = 1
                while left_data > 0:
                    tmp_partition[i] += 1
                    i += 1
                    left_data -= 1
                # print(tmp_partition)
                pre_end = 0
                for i in range(1, device+1):
                    partition[model_type][device][layer][i][0] = pre_end
                    pre_end = pre_end+tmp_partition[i]
                    partition[model_type][device][layer][i][1] = pre_end-1
                # if(device==2 and model_type==0 and layer==4):
                    # print(partition[model_type][device][layer])
        #print ("Next Model\n")

"""for device in range(2,3):
    #print("\nDevcie"+str(device))
    for model_type in range(1):
        for layer in range(4,5):
            #print(layer)
            if(layer_type[model_type][layer]!="FL"):
                print(output_size[model_type][layer])
                floor_data=math.floor(output_size[model_type][layer]/device)
                for i in range(1,device+1):
                    tmp_partition[i]=floor_data
                left_data=output_size[model_type][layer]-floor_data*device
                i=1
                while left_data>0:
                    tmp_partition[i]+=1
                    i+=1
                    left_data-=1
                #print(tmp_partition)
                pre_end=0
                for i in range(1,device+1):
                    partition[model_type][device][layer][i][0]=pre_end
                    pre_end=pre_end+tmp_partition[i]
                    partition[model_type][device][layer][i][1]=pre_end-1
                    print(partition[model_type][device][layer])"""


# build block execution time
for device in range(1, total_device_num+1):
    for i in range(model_length):
        for start_layer in range(0, len(layer_type[i])):
            for end_layer in range(start_layer, len(layer_type[i])):
                #print(str(start_layer)+"  "+str(end_layer))
                stop_inform = slice_layer(i, start_layer, end_layer, device)
                """if(device>2 and result_time[i][device][start_layer][end_layer]>result_time[i][device-1][start_layer][end_layer]):
                    result_time[i][device][start_layer][end_layer]=result_time[i][device-1][start_layer][end_layer]
                    computation_time[i][device][start_layer][end_layer]=computation_time[i][device-1][start_layer][end_layer]
                    transmission_time[i][device][start_layer][end_layer]=transmission_time[i][device-1][start_layer][end_layer]"""
                # if (stop_inform==1):
                # break
        #print ("Next Model\n")


# data.close()
data.release_resources()
del data


# for k==1 adjustment
# for model in range(model_length):
#     for device in range(1, total_device_num+1):
#         for layer in range(0, len(layer_type[model])):
#             #print("layer= "+str(layer))

#             tmp_execution_time[model][device][layer] = result_time[model][device][0][layer]
#             tmp_second_comp_time[model][device][layer] = computation_time[model][device][0][layer]
#             tmp_second_trans_time[model][device][layer] = transmission_time[model][device][0][layer]



for model in range(0, model_length):
    filename = 'Model_'+str(model)+'_cost2.xls'
    book = xlwt.Workbook()
    for device in range(1, total_device_num+1):
        sheet1 = book.add_sheet('Device_num_'+str(device))
        for start_layer in range(0, len(layer_type[model])):
            # print(start_layer)
            sheet1.write(0, start_layer+1, str(start_layer))
            sheet1.write(start_layer+1, 0, str(start_layer))
            for end_layer in range(start_layer, len(layer_type[model])):
                sheet1.write(end_layer+1, start_layer+1,
                             result_time[model][device][start_layer][end_layer])
        book.save(filename)

tmp_execution_time = np.empty(
    (model_length, total_device_num+1, maximum_row_num))
second_execution_time = np.empty((model_length, total_device_num+1))
tmp_second_trans_time = np.empty(
    (model_length, total_device_num+1, maximum_row_num))
tmp_second_comp_time = np.empty(
    (model_length, total_device_num+1, maximum_row_num))
second_trans_time = np.empty((model_length, total_device_num+1))
second_comp_time = np.empty((model_length, total_device_num+1))
# device_slice=np.empty((len(sheets),total_device_num+1,maximum_row_num))

device_slice = [[[[] for m in range(maximum_row_num+1)]
                 for j in range(total_device_num+1)] for i in range(model_length)]
# print(np.array(device_test).shape)
# print(type(device_test[0][0][0]))

prefetchers = []
# MLS_DP
for model in range(model_length):
    filename = 'Second_'+str(model)+'_cost2.xls'
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('second_dynamic')
    prefetcher = Prefetcher(str(model))
    prefetcher.set_stride(stride[model])
    prefetcher.set_filter(filter_size[model])
    prefetcher.set_padding(padding_start[model])
    prefetcher.set_input(input_size[model])
    prefetcher.set_output(output_size[model])
    prefetcher.set_channel(input_channel[model])
    prefetcher.set_layer_type(layer_type[model])
    for device in range(1, total_device_num+1):
        if device == 1:
            tmp_execution_time_k_1 = 0
            tmp_second_comp_time_k_1 = 0
            tmp_second_trans_time_k_1 = 0

            for layer in range(0, len(layer_type[model])):
                # print (result_time[model][device][layer][layer])
                # print (computation_time[model][device][layer][layer])
                # print (layer_type[model][layer])
                if layer_type[model][layer] != "FL" and layer_type[model][layer] != "conv":
                    continue
                if (layer != len(layer_type[model])-1 and layer_type[model][layer] == "conv" and layer_type[model][layer+1] == "pool"):
                    tmp_execution_time_k_1 += result_time[model][device][layer][layer+1]
                    tmp_second_comp_time_k_1 += computation_time[model][device][layer][layer+1]
                    tmp_second_trans_time_k_1 += transmission_time[model][device][layer][layer+1]    
                    # print (result_time[model][device][layer][layer])
                    continue
                # print (computation_time[model][device][layer][layer])
                # print (transmission_time[model][device][layer][layer])
                # print (result_time[model][device][layer][layer])
                tmp_execution_time_k_1 += result_time[model][device][layer][layer]
                tmp_second_comp_time_k_1 += computation_time[model][device][layer][layer]
                tmp_second_trans_time_k_1 += transmission_time[model][device][layer][layer]
            # print (layer)
            # print (tmp_execution_time_k_1)
            device_slice[model][device][layer] = [layer]
            i = 1
            pre_tmp = 0
            sheet1.write(0, (device-1)*2, "Device num=")
            sheet1.write(0, (device-1)*2+1, device)
            blocks = []
            for k in device_slice[model][device][layer]:
                sheet1.write(i, (device-1)*2, str(pre_tmp)+" to "+str(k))
                blocks.append([pre_tmp, k])
                if model == 1 and device==2:
                    print(pre_tmp, k)
                sheet1.write(i, (device-1)*2+1,
                             tmp_execution_time_k_1)
                pre_tmp = k+1
                i += 1
            sheet1.write(i, (device-1)*2, "Total time=")
            sheet1.write(i, (device-1)*2+1,
                         tmp_execution_time_k_1)
            prefetcher.append_slicing_blocks(blocks, 0, tmp_execution_time_k_1)
            second_execution_time[model][device] = tmp_execution_time_k_1
            second_comp_time[model][device] = tmp_second_comp_time_k_1
            second_trans_time[model][device] = tmp_second_trans_time_k_1
        else:
            for layer in range(0, len(layer_type[model])):
                #print("layer= "+str(layer))
                tmp_execution_time[model][device][layer] = result_time[model][device][0][layer]
                tmp_second_comp_time[model][device][layer] = computation_time[model][device][0][layer]
                tmp_second_trans_time[model][device][layer] = transmission_time[model][device][0][layer]

                device_slice[model][device][layer].append(layer)
                for l in range(0, layer):
                    #print("l= "+str(l))
                    if(tmp_execution_time[model][device][layer] > (tmp_execution_time[model][device][l]+result_time[model][device][l+1][layer])):
                        # print("Slice")
                        tmp_execution_time[model][device][layer] = (
                            tmp_execution_time[model][device][l]+result_time[model][device][l+1][layer])
                        tmp_second_comp_time[model][device][layer] = (
                            tmp_second_comp_time[model][device][l]+computation_time[model][device][l+1][layer])
                        tmp_second_trans_time[model][device][layer] = (
                            tmp_second_trans_time[model][device][l]+transmission_time[model][device][l+1][layer])
                        del device_slice[model][device][layer][:]
                        for k in device_slice[model][device][l]:
                            device_slice[model][device][layer].append(k)
                        device_slice[model][device][layer].append(layer)
                        # print(device_slice[model][device][layer])
                    # print("execution time= "+str(tmp_execution_time[model][device][layer]))
                # print(device_slice[model][device][layer])
                # print(tmp_execution_time[model][device][layer])
            i = 1
            pre_tmp = 0
            sheet1.write(0, (device-1)*2, "Device num=")
            sheet1.write(0, (device-1)*2+1, device)
            blocks = []
            for k in device_slice[model][device][layer]:
                sheet1.write(i, (device-1)*2, str(pre_tmp)+" to "+str(k))
                blocks.append([pre_tmp, k])
                sheet1.write(i, (device-1)*2+1,
                             result_time[model][device][pre_tmp][k])
                pre_tmp = k+1
                i += 1
            sheet1.write(i, (device-1)*2, "Total time=")
            sheet1.write(i, (device-1)*2+1,
                         tmp_execution_time[model][device][layer])
            prefetcher.append_slicing_blocks(blocks, device-1, tmp_execution_time[model][device][layer])
            second_execution_time[model][device] = tmp_execution_time[model][device][layer]
            second_comp_time[model][device] = tmp_second_comp_time[model][device][layer]
            second_trans_time[model][device] = tmp_second_trans_time[model][device][layer]
    prefetchers.append(prefetcher)
    book.save(filename)
prefetchers[1].prefetch()

"""execel for python plotlib"""
filename = 'Plot.xls'
book = xlwt.Workbook()
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model])
    for device in range(1, total_device_num+1):
        sheet1.write(0, device-1, second_execution_time[model][device])
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model]+"_layer by layer")
    for device in range(1, total_device_num+1):
        start = 0
        tmp_execution = 0
        for i in range(len(layer_type[model])):
            if(result_time[model][device][start][i] != np.finfo(np.float64).max):
                # print("update")
                tmp_execution += result_time[model][device][start][i]
                # print(start)
                # print(i)
                #print (tmp_execution)
                start = i+1
        sheet1.write(0, device-1, tmp_execution)
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model]+"_max pooling")
    for device in range(1, total_device_num+1):
        start = 0
        tmp_execution = 0
        for i in range(len(layer_type[model])):
            if(layer_type[model][i] == "pool" or layer_type[model][i] == "FL" or i == (len(layer_type[model])-1)):
                # print("update")
                tmp_execution += result_time[model][device][start][i]
                # print(start)
                # print(i)
                #print (tmp_execution)
                start = i+1
        sheet1.write(0, device-1, tmp_execution)
book.save(filename)


"""execel for python plotlib"""
filename = 'time.xls'
book = xlwt.Workbook()
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model])
    for device in range(1, total_device_num+1):
        sheet1.write(0, device-1, second_comp_time[model][device])
        sheet1.write(1, device-1, second_trans_time[model][device])
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model]+"_layer by layer")
    for device in range(1, total_device_num+1):
        start = 0
        tmp_comp_time = 0
        tmp_trans_time = 0
        for i in range(len(layer_type[model])):
            if(result_time[model][device][start][i] != np.finfo(np.float64).max):
                # print("update")
                tmp_comp_time += computation_time[model][device][start][i]
                tmp_trans_time += transmission_time[model][device][start][i]
                # print(start)
                # print(i)
                #print (tmp_execution)
                start = i+1
        sheet1.write(0, device-1, tmp_comp_time)
        sheet1.write(1, device-1, tmp_trans_time)
for model in range(model_length):
    sheet1 = book.add_sheet(model_name[model]+"_max pooling")
    for device in range(1, total_device_num+1):
        start = 0
        tmp_comp_time = 0
        tmp_trans_time = 0
        for i in range(len(layer_type[model])):
            if(layer_type[model][i] == "pool" or layer_type[model][i] == "FL" or i == (len(layer_type[model])-1)):
                # print("update")
                tmp_comp_time += computation_time[model][device][start][i]
                tmp_trans_time += transmission_time[model][device][start][i]
                # print(start)
                # print(i)
                #print (tmp_execution)
                start = i+1
        sheet1.write(0, device-1, tmp_comp_time)
        sheet1.write(1, device-1, tmp_trans_time)
book.save(filename)

"""

maximum_execution_time=np.empty((len(sheets),total_device_num+1))
device_resource=np.empty((len(sheets),total_device_num+1,model_length))
maximum_execution_time.fill(np.finfo(np.float64).max)

filename = 'Final_result.xls'
book = xlwt.Workbook()
sheet1 = book.add_sheet('final_result')
for model in range(model_length):
    print("model= "+str(model))
    sheet1.write(0,model+1,model_name[model])
    for device in range(model+1,total_device_num+1-(model_length-model-1)):
        print("Total device= "+str(device))
        if(model==0):
            maximum_execution_time[model][device]=second_execution_time[model][device]
            device_resource[model][device][0]=device
            for i in range(1,model_length):
                device_resource[model][device][i]=0
            print("resource: "+str(device_resource[model][device]))
        else:
            for r in range(model,device): #前面分配的device數量
                print("r= " +str(r))
                print(maximum_execution_time[model][device])
                print(maximum_execution_time[model-1][r])
                print(second_execution_time[model][device-r])
                if(maximum_execution_time[model][device]>max(maximum_execution_time[model-1][r],second_execution_time[model][device-r])):
                    print("win")
                    maximum_execution_time[model][device]=max(maximum_execution_time[model-1][r],second_execution_time[model][device-r])
                    for i in range(model_length):
                        device_resource[model][device][i]=device_resource[model-1][r][i]
                    device_resource[model][device][model]=(device-r)
                    print("resource: "+str(device_resource[model][device]))
for device in range(model_length,total_device_num+1):
    sheet1.write(device-model_length+1,0,"Device= "+str(device))
    for i in range(model_length):
        sheet1.write(device-model_length+1,i+1,device_resource[model][device][i])
book.save(filename)"""
