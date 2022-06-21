# DebugUtil

## PDB（命令行调试工具）

* 进入命令行DEBUG模式

```bash
$ python -m pdb <python_file>
```

* [常用命令行 python docs](https://docs.python.org/3/library/pdb.html#debugger-commands)、[常用命令行（中文）](https://www.cnblogs.com/xiaohai2003ly/p/8529472.html)、[CheatSheet](https://appletree.or.kr/quick_reference_cards/Python/Python%20Debugger%20Cheatsheet.pdf)

| 作用             | 命令行        |
| ---------------- | ------------- |
| 打断点           | b(break) 行号 |
| 查看帮助手册     | h(help)       |
| 显示变量值       | p(print)      |
| 退出debug        | q(quit)       |
| 打印当前执行堆栈 | w(where)      |

.. note:: PDB支持输入python语句，对于不认识的命令行，PDB会认为是python语句来执行），要显式表示该语句为python语句，可以在输入前加入 `!`

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210914104258430.png" alt="image-20210914104258430" style="zoom:67%;" />

---

**NOTE**

给当前文件的第3行打断点：b 3

给文件A的第15行打断点：b A:15

---

## [PDB++](https://github.com/pdbpp/pdbpp)

安装后会默认替换PDB；加强版的特性包括：语法高亮、添加颜色

* 安装

```bash
$ pip3 install pdbpp
```

## [py-spy](https://github.com/benfred/py-spy)（性能分析工具）

```bash
# 安装
$ pip3 install py-spy

# 生成火焰图
$ py-spy record -o profile.svg --pid <pid>
# OR
$ py-spy record -o profile.svg -- python <file_name>

# 实时查看函数调用情况
$ py-spy top --pid <pid>
# OR
$ py-spy top -- python <file_name>
```

## [pyroscope](https://github.com/pyroscope-io/pyroscope)（性能分析工具）

待补充...

```bash
# 安装(for ubuntu)
$ wget https://dl.pyroscope.io/release/pyroscope_0.0.39_amd64.deb
$ sudo apt-get install ./pyroscope_0.0.39_amd64.deb

# 快速上手
$ export PYROSCOPE_SERVER_ADDRESS=http://<127.0.0.1:4040>
$ pyroscope exec python <python_file>
# connect     Connect to an existing process and profile it
# exec        Start a new process from arguments and profile it
# help        Help about any command
# server      Start pyroscope server. This is the database + web-based user interface

# attach a process
$ pyroscope connect -pid {my-app-pid}
```

## jupyter notebook

### [安装](https://jupyter.org/install)

现代版为JupyterLab，经典版为Jupyter Notebook

```bash
$ conda install -c conda-forge jupyterlab
$ jupyter-lab

$ conda install -c conda-forge notebook
$ jupyter-notebook
```

### 常用快捷键

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/UJDCF6nWuPO2254k.png" alt="img" style="zoom:67%;" />

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/RdT3F013Ud82KdwE.png!thumbnail" alt="img" style="zoom:67%;" />

### 拓展插件

```bash
$ conda install -c conda-forge jupyter_contrib_nbextensions jupyter_nbextensions_configurator
```
### [magic function](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

- 执行`全部执行`时，跳过某一个cell不运行

```python
%%script echo skipping
```

- 测单元格时间

```python
%%timeit -r 5
# -r: 指定执行次数
```

