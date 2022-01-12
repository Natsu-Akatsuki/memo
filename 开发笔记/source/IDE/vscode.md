# vscode

<p align="right">Author: kuzen, Natsu-Akatsuki</p>

## 插件

### 代码格式化

本部分介绍相关的格式化插件

* 触发保存时格式化，在配置文件`setting.json`中添加配置项

```json
"editor.formatOnSave": true
```

#### bash, dockerfile, ignore

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/O9pE80ad5tSzvWGn.png!thumbnail" alt="img" style="zoom:50%; " />

#### CMakeLists

使用前需 `pip3 install cmake_format`

 <img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/3JNlJDmC4jiLlsVQ.png!thumbnail" alt="img" style="zoom: 67%; " />

#### c/c++

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/rwMxwrqpHdPxgLFX.png!thumbnail" alt="img" style="zoom: 67%; " />

* [自定义配置](https://blog.csdn.net/star871016/article/details/109526408)

#### xml

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815142421076.png" alt="image-20210815142421076" style="zoom: 67%; " />

### 括号高亮

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210815144939462.png" alt="image-20210815144939462" style="zoom:50%; " />

.. note::  `vscode 1.6版已经内置了括号高亮功能 <https://code.visualstudio.com/updates/v1_60#_editor>`_ ，在配置文档添加   ``"editor.bracketPairColorization.enabled": true`` 即可

### 代码块

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210822091455860.png" alt="image-20210822091455860" style="zoom:67%; " />

### 代码运行快捷键

![image-20210822092237158](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210822092237158.png)

### 代码补全

![image-20210822092912158](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210822092912158.png)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210915135337974.png" alt="image-20210915135337974" style="zoom: 67%; " />

* [安装](https://www.kite.com/linux/)

```bash
# 使用该插件前需要安装kite engine
$ bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)"
$ systemctl --user start kite-autostart
```

### 查看API文档

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210919093438088.png" alt="image-20210919093438088" style="zoom:67%; " />

对应的默认快捷键： `ctrl+h` , `ctrl+alt+h`

### 生成doxygen文档

![image-20211002021724564](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002021724564.png)

使用 `/**` 和回车键进行触发

### 使用文件模板

![image-20210927231828413](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210927231828413.png)

### 快速编译和运行代码

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928115134786.png" alt="image-20210928115134786" style="zoom:67%; " />

* 设置编译项

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928115123099.png" alt="image-20210928115123099" style="zoom:67%; " />

### DEBUG

[cppcheck](https://cppcheck.sourceforge.io/), [flawfinder](https://github.com/david-a-wheeler/flawfinder), clang, flexelint...

```bash
# cppcheck
$ sudo apt install cppcheck
# flawfinder
$ sudo pip3 install flawfinder
```

### [markdown插件](https://code.visualstudio.com/docs/languages/markdown)

.. hint:: 官方教程包括了： `preview` , `outline` , `code snippet`

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141059733.png" alt="image-20210901141059733" style="zoom: 67%; " />

* 表格美化

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141231319.png" alt="image-20210901141231319" style="zoom:67%; " />

* 格式化

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901141415436.png" alt="image-20210901141415436" style="zoom:67%; " />

* [链接粘贴（自动生成rst和markdown格式的超链接）](https://marketplace.visualstudio.com/items?itemName=kukushi.pasteurl)

等价于typora的超链接功能

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210907093539625.png" alt="image-20210907093539625" style="zoom:67%; " />

### 其他

* markdown math：给vscode中markdown添加数学支持（latex）
* remote ssh：远程连接
* ros：添加ros支持
* printcode：代码打印
* Code Spell Checker：拼写检查与修正
* Live Share: 实时协作写代码（微软官方插件）
* Live Share Audio：为Live Share开启语音交流（微软官方插件）

## 实战

### 配置文档

本部分介绍vscode涉及的配置文档

* `tasks.json`：告诉编译器怎么构建程序
* `launch.json`：告诉GDB怎么启动Debug

#### [全局配置](https://code.visualstudio.com/docs/getstarted/settings#_default-settings)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210828002103557.png" alt="image-20210828002103557" style="zoom:67%; " />

### [同步配置信息（配置文档、插件）](https://code.visualstudio.com/docs/editor/settings-sync)

### [构建插件组(expansion pack)](https://code.visualstudio.com/blogs/2017/03/07/extension-pack-roundup)

步骤一：[安装高版本的node.js](https://github.com/nodejs/help/wiki/Installation)（用apt安装的有如下警告和报错）

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/LjKQ3d57TAU133aE.png!thumbnail" alt="img" style="zoom:67%; " />

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uQkyGFoF25MOUDCs.png!thumbnail" alt="img" style="zoom:67%; " />

.. hint:: 安装时不用像官网一样导入到系统路径

步骤二：安装 `Yeoman generator`

```bash
$ npm install -g yo generator-code
```

步骤三：构建一个 `expansion pack` 文件夹

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210901234442008.png" alt="image-20210901234442008" style="zoom:67%; " />

步骤四：安装 `vsce` 和在该文件夹下构建拓展插件

```bash
$ npm install -g vsce
# vsce报错缺什么，package.json就加哪个字段的信息
$ vsce package
```

.. hint:: 上述命令行执行时并非一步到位，此处省略了根据提示而进行操作的步骤；其中要vsce package构建成功需要合适的package.json文件；更多可参考 [github实例](https://github.com/robertoachar/vscode-extension-pack)

### 自定义Button来触发pandoc

* [插件](https://marketplace.visualstudio.com/items?itemName=seunlanlege.action-buttons)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210905204620163.png" alt="image-20210905204620163" style="zoom:67%; " />

#### pandoc

.. hint:: 正如[此处](https://github.com/miyakogi/m2r)所述，在实际的测试中，pandoc对markdown->rst的转换效果一般，如容易丢失图片，不建议用pandoc进行转换

* 安装

```bash
$ wget -c https://github.com/jgm/pandoc/releases/download/2.14.2/pandoc-2.14.2-linux-amd64.tar.gz -O ~/application/pandoc-2.14.2-linux-amd64.tar.gz
$ cd application && tar -xzvf pandoc-2.14.2-linux-amd64.tar.gz
$ cd pandoc-2.14.2
$ sudo ln -s $(pwd)/bin/pandoc /usr/local/bin
$ sudo ln -s $(pwd)/share/man/man1/pandoc.1.gz /usr/share/man/man1
# 添加自动补全
$ echo 'eval "$(pandoc --bash-completion)"' >> ~/.bashrc
```

---

**NOTE**

pandoc常用选项：

| option    | 作用                                               |
| --------- | -------------------------------------------------- |
| -f/--from | 指定输入格式                                       |  |
| -t/--to   | 指定输出格式（若无指定格式则会根据文件名进行推导） |

---

* 配置配置文件

```json
"actionButtons": {
    "defaultColor": "#ff0034", // Can also use string color names.
    "loadNpmCommands": false, // Disables automatic generation of actions for npm commands.
    "reloadButton": "♻️", // 触发配置生效
    "commands": [
        {
            "cwd": "${workspaceFolder}", // cd workspace
            "name": "pandoc",            // terminal name/ tip name
            "color": "green",
            "singleInstance": true,
            "command": "pandoc -s -f markdown -t rst ${file} >> ${fileDirname}/${fileBasenameNoExtension}.rst", // This is executed in the terminal.
        }
    ]
},
```

* 拓展资料
  * [支持的转换格式](https://docs.onap.org/en/dublin/guides/onap-developer/how-to-use-docs/converting-formats.html#fixing-the-converted-document)
  * [用例](https://pandoc.org/demos.html)

### web端查看github代码

将 `.com` 改为 `.dev`

### 设置滚轮速度

![img](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/CuLMsVboZB2jId0c.png!thumbnail)

### 设置自动格式化

![image-20210928102245843](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210928102245843.png)

### [取消标签页的重用（取消preview模式）](https://code.visualstudio.com/docs/getstarted/userinterface#_preview-mode)

![image-20211002022616570](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211002022616570.png)

### 显示时根据视窗调整显示

![image-20211105210939685](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211105210939685.png)

## 常用快捷键

### 代码/文件间反复横跳(code navigation)

|       作用       |            快捷键            |
| :--------------: | :--------------------------: |
|     括号跳转     | ctrl+shift+\  (i.e. ctrl+\|) |
| 打开最近工作空间 |            ctrl+r            |

