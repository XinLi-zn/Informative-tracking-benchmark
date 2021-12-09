# Informative-tracking-benchmark
An informative tracking benchmark comprising 9 scenarios, 180 diverse videos, and 86,260 frames with new challenges. <img src="visevent_art.png" width="400" align="right"> 






<!--
# VisEvent_SOT_Benchmark <img src="visevent_art.png" width="400" align="right"> 
The First Large-scale Benchmark Dataset for Reliable Object Tracking by fusing RGB and Event Cameras. 

> **VisEvent: Reliable Object Tracking via Collaboration of Frame and Event Flows**, Xiao Wang, Jianing Li, Lin Zhu, Zhipeng Zhang, Zhe Chen, Xin Li, Yaowei Wang, Yonghong Tian, Feng Wu 
> **[[Paper](https://arxiv.org/pdf/2108.05015.pdf)] [[Project](https://sites.google.com/view/viseventtrack/)] [[DemoVideo](https://www.youtube.com/watch?v=U4uUjci9Gjc)] [[VideoTutorial](https://www.youtube.com/watch?v=vGwHI2d2AX0&ab_channel=XiaoWang)]** 
-->

## News: 
* 2021.12.09 The informative tracking benchmark is released. 



## Introduction 
Along with the rapid progress of visual tracking, existing benchmarks become less informative due to redundancy of samples and weak discrimination between current trackers, making evaluations on all datasets extremely time-consuming. Thus, a small and informative benchmark, which covers all typical challenging scenarios to facilitate assessing the tracker performance, is of great interest. In this work, we develop a principled way to construct a small and informative tracking benchmark (ITB) with 7\% out of 1.2 M frames of existing and newly collected datasets, which enables efficient evaluation while ensuring effectiveness. Specifically, we first design a quality assessment mechanism to select the most informative sequences from existing benchmarks taking into account 1) challenging level, 2) discriminative strength, 3) and density of appearance variations. Furthermore, we collect additional sequences to ensure the diversity and balance of tracking scenarios, leading to a total of 20 sequences for each scenario. By analyzing the results of 15 state-of-the-art trackers re-trained on the same data, we determine the effective methods for robust tracking under each scenario and demonstrate new challenges for future research direction in this field. 


## Dataset Samples 
![visevent-example](https://github.com/wangxiao5791509/RGB_Event_Tracking_Benchmark/blob/main/videosamples.png)

## How to Download VisEvent Dataset? 
We provide both the original **aedat4** (contains the RGB frames, event, time-stamp) and **image** format for VisEvent dataset. 

**The "Image" only version (about 63.7 GB):** 
Preview of each files: <img src="https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark/blob/main/Screenshot%20from%202021-08-27%2009-08-23.png" width="200" align="left"> 
[[**BaiduYun (Code: pclt)**](https://pan.baidu.com/s/1E7dgAxHV2QFPByKs3is7nw)] 
[[**GoogleDrive**]] 
[[**OneDrive**](https://stuahueducn-my.sharepoint.com/:f:/g/personal/e16101002_stuahueducn_onmicrosoft_com/Em_Cv5OzNpBAjlhzOHeqwxEBR4B2Xrj3hqMIk-U0RvKXzg?e=EBJWkR)] 


**The "aedat4" version (about 226.0 GB):** 
Preview of each files: <img src="https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark/blob/main/Screenshot%20from%202021-08-27%2008-57-19.png" width="200" align="left"> 
[[**BaiduYun(Code: pclt)**](https://pan.baidu.com/s/122DXDc7OO5mB78kbU-G98Q)] 
[[**Googledrive**](https://drive.google.com/drive/folders/188pkivkfshpLSMADx9kgzw0PIcDqhclO?usp=sharing)] 
[[**Onedrive**](https://stuahueducn-my.sharepoint.com/:f:/g/personal/e16101002_stuahueducn_onmicrosoft_com/EkEqjE5_M1lKjyc__fq8o5oBYR9cVqyFOvmSguz-ro111A?e=4K8Vue)]





## Evaluation ToolKit 
Only matlab version is available. 

**1. Download this github:**
    
    git clone https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark

**2. Download the tracking results of our benchmark:**
[[**GoogleDrive (185MB)**](https://drive.google.com/file/d/1fILCNMrwt2PiITPWIQFZpk1PJvg_JAjX/view?usp=sharing)]

    unzip tracking_results_VisEvent_SOT_benchmark.zip, and put it into the folder "tracking_results". 

    unzip the "annos.zip" in the folder "annos"

**3. Open your matlab, and run the script "Evaluate_VisEvent_SOT_benchmark.m". Wait and check the final evaluated figures**

<img src="res_fig/VisEvent_benchmark_results.png" width="650"> 



## Contact
If you have any questions about this benchmark, please feel free to contact Xin Li at xinlihitsz@gmail.com.

<!--
## More Related Materials 
* [**Github-1**] https://github.com/wangxiao5791509/SNN_CV_Applications_Resources 
* [**Github-2**] https://github.com/uzh-rpg/event-based_vision_resources 
* [**Github-3**] https://github.com/wangxiao5791509/Single_Object_Tracking_Paper_List
* [**Survey**] **神经形态视觉传感器的研究进展及应用综述**，计算机学报，李家宁, 田永鸿 [[Paper](https://drive.google.com/file/d/1d7igUbIrEWxmUI7xq75P6h_I4H7uI3FA/view?usp=sharing)] 
* [**Survey**] **Event-based Vision: A Survey**, Guillermo Gallego, et al., IEEE T-PAMI 2020, [[Paper](https://arxiv.org/abs/1904.08405)]
* [**FE108 dataset**] **Object Tracking by Jointly Exploiting Frame and Event Domain**, Jiqing Zhang, et al., ICCV 2021, [[Project](https://zhangjiqing.com/dataset/)] [[DemoVideo](https://www.youtube.com/watch?v=EeMRO8XVv04&ab_channel=JiqingZhang)] [[Github](https://github.com/Jee-King/ICCV2021_Event_Frame_Tracking)] [[Dataset](https://zhangjiqing.com/dataset/contact.html)] [[Paper](https://arxiv.org/pdf/2109.09052.pdf)]
* [**SpikingJelly**] (SpikingJelly is an open-source deep learning framework for Spiking Neural Network (SNN) based on PyTorch) [[OpenI from PCL](https://git.openi.org.cn/OpenI/spikingjelly)] [[GitHub](https://github.com/fangwei123456/spikingjelly)] [[Documents](https://spikingjelly.readthedocs.io/zh_CN/latest/)]
* [**Event-Toolkit**] https://github.com/TimoStoff/event_utils (Various representations can be obtained with (a) the raw events, (b) the voxel grid, (c) the event image, (d) the timestamp image.)

<img src="res_fig/event_representations.png" width="650"> 


## :page_with_curl: BibTex: 
If you find this work useful for your research, please cite the following papers: 

```bibtex
@article{wang2021viseventbenchmark,
  title={VisEvent: Reliable Object Tracking via Collaboration of Frame and Event Flows},
  author={Xiao Wang, Jianing Li, Lin Zhu, Zhipeng Zhang, Zhe Chen, Xin Li, Yaowei Wang, Yonghong Tian, Feng Wu},
  journal={arXiv:2108.05015},
  year={2021}
}
```
-->

















