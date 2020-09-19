import xlrd
import numpy as np
from config import *
import matplotlib.pyplot as plt

data = xlrd.open_workbook('time.xls')

sheets=data.sheets()

model_length=len(sheets)
#total_device_num=10
# print(total_device_num)

model_name=["YOLOv2","AlexNet","VGG16","VGG19"]
width=0.3
x1 = np.arange(1,total_device_num+1,1)
y1=np.empty((model_length,total_device_num))
y2=np.empty((model_length,total_device_num))
y3=np.empty((model_length,total_device_num))

FLOPs=[62912683682,2271803160,30967614488,39293819928]

for sheet_count in range(0,4):
    
    #y1 = np.sin(x1 * np.pi / 180.0)
    j=0
    MLS_min = 9999999
    MLS_index = 0
    MLS_first = 9999999
    SLS_min = 9999999
    SLS_index = 0
    SLS_first = 9999999
    for sheet in sheets:
        #print(sheet.ncols)
        if(j==sheet_count):
            # y1[j][0]=0
            # y2[j][0]=(FLOPs[sheet_count]/(13.6*1000000000))
            # y3[j][0]=y1[j][0]+y2[j][0]
            for i in range(0,total_device_num):
                #print(i)
                y1[j][i]=sheet.cell_value(1,i)  #communication time
                
                #y2[j][i-1]=(sheet.cell_value(0,i)/(6.2*1000000000))
                y2[j][i]=(sheet.cell_value(0,i))  #computation time
                #print(y2)
                y3[j][i]=y1[j][i]+y2[j][i]
                if (i==0):
                    MLS_first = y3[j][i]
                if (MLS_min > y3[j][i]):
                    MLS_min = y1[j][i]+y2[j][i]
                    MLS_index = i
            #print(y[j])

            #plt.bar(x1, y[j],width=width, label='comm', bottom=y2[j])
            plt.bar(x1, y3[j], width=width, label='MLS',color='#C0C0C0',edgecolor="k")
            #plt.bar(x1,y2[j],"o-",label=str(sheet.name))
            #print(y2[j])
        if(j==(sheet_count+4)):
            # y1[j][0]=0
            # y2[j][0]=(FLOPs[sheet_count]/(13.6*1000000000))
            # y3[j][0]=y1[j][0]+y2[j][0]
            for i in range(0,total_device_num):
                #print(i)
                y1[j][i]=sheet.cell_value(1,i)
                y2[j][i]=(sheet.cell_value(0,i))
                y3[j][i]=y1[j][i]+y2[j][i]
                if (i == 0):
                    SLS_first = y3[j][i]
                if (SLS_min > y3[j][i]):
                    SLS_min = y1[j][i]+y2[j][i]
                    SLS_index = i
            #plt.bar(x1+width, y[j],width=width, label='comm', bottom=y2[j])
            plt.bar(x1+width, y3[j], width=width, label='SLS',color='#808080',edgecolor="k")
            #plt.bar(x1,y[j],"o-",label=str(sheet.name))
            #print(y2[j])
        j+=1
    #plt.xlim(0,11,1)
    print ("================== ", model_name[sheet_count]," ==================")
    print ("MLS_min:", MLS_min, "s, when k = ", MLS_index+1)
    print ("SLS_min:", SLS_min, "s, when k = ", SLS_index+1)
    print ("MLS faster than SLS about", (1.0-(MLS_min/SLS_min))*100.0, "%")
    print ("MLS faster than one device about", (1.0-(MLS_min/MLS_first))*100.0, "%")
    plt.xticks(x1+ (width) / 2, x1)
    #plt.ylim(0,11,1)
    plt.title(model_name[sheet_count])
    plt.xlabel('Number of devices')
    plt.ylabel('Execution time (s)')
    plt.legend(title="Algorithm")
    plt.savefig("layer_figure_output/bar/execution time/output_pi4_"+model_name[sheet_count]+"_"+str(total_device_num)+"_black.png",dpi=500)
    plt.clf()


for model_count in range(0,4):
    print ("================== ", model_name[model_count]," ==================")
    plt.bar(x1, y1[model_count], width=width, label='MLS-trans',color='#C0C0C0',edgecolor="k")
    plt.bar(x1, y2[model_count], width=width, bottom = y1[model_count], label='MLS-comp',color='#E0E0E0',edgecolor="k")
    plt.bar(x1+width, y1[model_count+4], width=width, label='SLS-trans',color='#808080',edgecolor="k")
    plt.bar(x1+width, y2[model_count+4], width=width, bottom = y1[model_count+4], label='SLS-comp',color='#505050',edgecolor="k")
    plt.xticks(x1+ (width) / 2, x1)
    plt.title(model_name[model_count])
    plt.xlabel('Number of devices')
    plt.ylabel('Execution time (s)')
    plt.legend(title="Algorithm")
    plt.savefig("layer_figure_output/final/output_stack_pi4_"+model_name[model_count]+"_black.png",dpi=500)
    plt.clf()

# print(y1)
# print(y2)
data.release_resources()
del data

for sheet_count in range(0,4):
    # print(sheet_count)
    j=0
    for sheet in sheets:
        if(j==sheet_count):
            plt.bar(x1, y1[j]*(bw_abbr/8), width=width, label='MLS',color='#C0C0C0',edgecolor="k")
        if(j==(sheet_count+4)):
            plt.bar(x1+width, y1[j]*(bw_abbr/8), width=width, label='SLS',color='#808080',edgecolor="k")
        j+=1
    plt.xticks(x1+ (width) / 2, x1)
    #plt.ylim(0,11,1)
    plt.title(model_name[sheet_count])
    plt.xlabel('Number of devices')
    plt.ylabel('Communication size (Mbytes)')
    plt.legend(title="Algorithm")
    #plt.ylim(0,1.7,0.2)
    plt.savefig("layer_figure_output/bar/comm/comm_"+model_name[sheet_count]+"_"+str(total_device_num)+"_black.png",dpi=500)
    #plt.show()

    plt.clf()

for sheet_count in range(0,4):
    j=0
    for sheet in sheets:
        if(j==sheet_count):
            plt.bar(x1, y2[j]*FLOPs_abbr, width=width, label='MLS',color='#C0C0C0',edgecolor="k")
        if(j==(sheet_count+4)):
            plt.bar(x1+width, y2[j]*FLOPs_abbr, width=width, label='SLS',color='#808080',edgecolor="k")
        j+=1
    plt.xticks(x1+ (width) / 2, x1)
    #plt.ylim(0,11,1)
    plt.title(model_name[sheet_count])
    plt.xlabel('Number of devices')
    plt.ylabel('FLOPs (GFLOPs)')
    plt.legend(title="Algorithm")

    plt.savefig("layer_figure_output/bar/comp/comp_"+model_name[sheet_count]+"_"+str(total_device_num)+"_black.png",dpi=500)
    #plt.show()

    plt.clf()

"""countries = ['USA', 'GB', 'China', 'Russia', 'Germany']
bronzes = np.array([38, 17, 26, 19, 15])
silvers = np.array([37, 23, 18, 18, 10])
golds = np.array([46, 27, 26, 19, 17])
width = 0.35
#ind = [x for x, _ in enumerate(countries)]


ind = np.arange(5) 
plt.bar(ind, silvers, width=0.35, label='silvers', color='silver', bottom=bronzes)
plt.bar(ind, bronzes, width=0.35, label='bronzes', color='#CD853F')
plt.bar(ind +width, golds, width=0.35, label='golds', color='gold')

plt.xticks(ind + (2*width) / 2, countries)
plt.ylabel("Medals")
plt.xlabel("Countries")
plt.legend(loc="upper right")
plt.title("2012 Olympics Top Scorers")

plt.show()"""

"""N = 5
men_means = (20, 35, 30, 35, 27)
women_means = (25, 32, 34, 20, 25)

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, men_means, width, label='Men')
plt.bar(ind + width, women_means, width,
    label='Women')

plt.ylabel('Scores')
plt.title('Scores by group and gender')

plt.xticks(ind + width / 2, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.legend(loc='best')
plt.show()"""
