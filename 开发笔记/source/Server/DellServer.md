# DellServer

for T640

## CLI

### 查看信息

```bash
# 查看服务器可用的PCIe
$ sudo racadm get System.PCIESlotLFM
# System.PCIESlotLFM.1 [Key=PCIESlotLFM#PCIESlotLFM]
# System.PCIESlotLFM.2 [Key=PCIESlotLFM#PCIESlotLFM]
# System.PCIESlotLFM.3 [Key=PCIESlotLFM#PCIESlotLFM]
# ...

# 查看传感器信息
$ sudo racadm getsensorinfo

# 查看系统信息（如服务编号）
$ getsysinfo
```

### [调整PCIe的模式](https://www.dell.com/support/manuals/en-us/oth-t640/idrac9_racadm_ar_refguide/system.pcieslotlfm.lfmmode-read-or-write?guid=guid-adc0ce99-37d7-411c-a0a0-ec85f9ed6ec1&lang=en-us)

```bash
# 设置自动模式
$ sudo racadm set System.PCIESlotLFM.6.LFMMode 0
# 设置自定义模式
$ sudo racadm set System.PCIESlotLFM.1.LFMMode 2
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220611213053308.png" alt="image-20220611213053308" style="zoom: 50%;" />

## Practice

- [构建bash脚本和服务来调整风扇转速](https://gist.github.com/Natsu-Akatsuki/a4d5650041dfb9330d570665067a5f50#file-auto_adjust_fan)

## Software

### [Dell EMC Repository Manager](https://www.dell.com/support/home/en-in/drivers/DriversDetails?driverid=4d32v)

用于安装最新的驱动程序

#### [CLI](https://www.dell.com/support/manuals/en-in/repository-manager-v3.1/drm3.1_qig/sample-commands?guid=guid-94f7696b-7aa7-496d-9160-5db042e148c6&lang=en-us)

```bash
# 安装
$ sudo ./DRMInstaller.bin
# 运行
$ drm
```

### [Dell EMC OpenManage](https://linux.dell.com/repo/community/openmanage/)

```bash
$ sudo echo 'deb http://linux.dell.com/repo/community/openmanage/10100/focal focal main' | sudo tee -a /etc/apt/sources.list.d/linux.dell.com.sources.list
$ sudo wget https://linux.dell.com/repo/pgp_pubkeys/0x1285491434D8786F.asc
$ sudo apt-key add 0x1285491434D8786F.asc
$ sudo apt-get update
$ sudo apt-get install srvadmin-all
```

### [Dell EMC iDRAC Service Module](https://www.dell.com/support/home/en-us/product-support/product/poweredge-t640/drivers)

设置IDRAC服务

```bash
# Extract the tar file
$ bash setup.sh
```

## Reference

- [T640 安装和服务手册](https://www.dell.com/support/manuals/en-us/poweredge-t640/pet640_ism_pub/%E6%B3%A8%E6%84%8F%E3%80%81%E5%B0%8F%E5%BF%83%E5%92%8C%E8%AD%A6%E5%91%8A?guid=guid-5b8de7b7-879f-45a4-88e0-732155904029&lang=zh-cn)
- [T640 官方技术文档](https://i.dell.com/sites/csdocuments/Shared-Content_data-Sheets_Documents/ja/jp/Dell-EMC-PowerEdge-T640-Technical-Guide-2018Jun24.pdf)
- [官方故障排错指南](https://dl.dell.com/topics/pdf/troubleshootingguide_zh-cn.pdf)（含指示灯的说明）
- [终端报错信息排错](https://www.dell.com/support/manuals/en-in/dell-opnmang-sw-v8.0.1/eemi_13g-v1/uefi-event-messages?guid=guid-c1c6f253-f8ef-43bf-b8ed-1a9b2a910ac4&lang=en-us)
