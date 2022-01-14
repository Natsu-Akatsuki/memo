# third-party library

## easydict

### 用dot的方式取字典值

```python
from easydict import EasyDict
... = EasyDict(<字典>）
```

- 安装

```bash
$ pip3 install esaydict
```

## [pathlib](https://docs.python.org/3.11/library/pathlib.html)

.. note:: 支持跨系统使用，解析路径友好；一般会用该模块，来取代 `os` 模块的功能；其支持sorted()方法；一些module比如 `open3d` 不支持 `PosixPath` 类，传参时需要转化为 `str` 型；在Path对象中可使用 `..` 等进行拼接，后续调用 `resolve()` 方法进行解析

### 常用代码块

```python
# 01.导入库
from pathlib import Path

# 02.判断文件或文件夹是否存在
<Path object>.exists()：

# 03.将相对路径转换为绝对路径（resolving any symlinks）    
p = Path()   # 默认使用的是当前路径    

# 04.创建文件夹
# parents：若parent目录缺失，则会递归的创建；
# exist_ok：文件夹已存在时，不会报错也不会覆盖建文件夹
<Path object>.mkdir(parents=True, exist_ok=True)

# 05.通配符模式（列出通配符的文件）
# 返回的是generator，可以使用list()将其转换为列表
image_path = (Path('/home/helios/image/').glob('*.jpg'))

# 06.添加后缀
<Path object>.with_suffix('.jpg')
```

.. hint:: 一些常用属性，以\"/home/helios/path.py\"为例

- 其 `name` (即basename) 为path.py
- 其 `parent` (即dirname) 为/home/helios
- 其 `stem` 为path（不带后缀的basename）

### 参考资料

- [csdn资料](https://blog.csdn.net/itanders/article/details/88754606)

## numpy

### 实战

#### numpy矩阵相乘运算cpu占用率大

进行矩阵运算时默认使用多线程进行运算，可以通过限制线程数来减少占用率（运算时间会提高）

```python
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
```

## scipy

### [计算凸包](https://www.tutorialspoint.com/scipy/scipy_spatial.htm)

```bash
import numpy as np
from scipy.spatial import ConvexHull
points = np.random.rand(10, 2) # 30 random points in 2-D
hull = ConvexHull(points)
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()
```

<https://realpython.com/python-menus-toolbars/>

## opencv

### VideoWriter

生成视频流

```python
for split, dataset in zip(splits, datasets):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # 编码方式
    vout = cv2.VideoWriter(<"输出的文件名">, fourcc , 30.0, (img_w, img_h))
    for i, data in enumerate(tqdm.tqdm(loader)):
        ...            
        vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
        vout.write(vis)

    vout.release()
```

[读写视频流](https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)

## collections

## 图形化

### pygui

.. note:: 暂无排上用场

#### 创建一个窗口

- 添加按钮
- 添加文本

```python
def save_callback():
    print("Save Clicked")
    
with dpg.window(label="Example Window"):
    dpg.add_text("Hello world")
    dpg.add_button(label="Save", callback=save_callback)
    dpg.add_input_text(label="string")
    dpg.add_slider_float(label="float")
```

![image-20211129142358432](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20211129142358432.png)

#### [添加菜单栏](https://dearpygui.readthedocs.io/en/latest/documentation/menus.html)

- 包括子窗口菜单栏和主窗口菜单栏

#### [Glossary](https://dearpygui.readthedocs.io/en/latest/extra/glossary.html)

- alias - A string that takes the place of the regular **int** ID. Aliases can be used anywhere UUID’s can be used.
- item - Everything in **Dear PyGui** created with a context manager or a add_ command.
- root - An item which has no parent (i.e. window, registries, etc.)
- window - A **Dear ImGui** window created with add_window(…).
- viewport - The operating system window.

- tag：组件的ID / alias

## packing

### [pyinstaller](https://github.com/pyinstaller/pyinstaller)

```bash
$ pip install pyinstaller

# for windows
$ pyinstaller -F -c .\<file_-name>

# option:
# -F/-D：将所有依赖打包成一个文件/非一个文件
# -c(default)/-w：是否需要控制台/终端来显示标准输入和输出
```

---

**NOTE**

1. 如果打包成一个文件的话，到时运行时需要解压操作，所以打开时较慢.
2. 实测，不能打包文件和资源文件夹同名

---

### [auto_py_to_exe](https://nitratine.net/blog/post/issues-when-using-auto-py-to-exe/?utm_source=auto_py_to_exe&utm_medium=application_link&utm_campaign=auto_py_to_exe_help&utm_content=bottom)

pyinstaller的GUI版本

### [nuitka](https://nuitka.net/doc/index.html)

#### [安装](https://nuitka.net/doc/user-manual.html#tutorial-setup-and-build-on-windows) (for windows)

.. note:: 实测只能使用**纯python环境**，否则会有如下报错：FATAL: Error, usable static libpython is not found for this Python installation. You might be missing required '-dev' packages. Disable with --static-libpython=no" if you don't want to install it.

```plain
# 使用纯python环境时
$ pip install -U nuitka

# 使用conda环境时
$ conda install -c conda-forge nuitka
```

---

**NOTE**

- [python 安装](https://www.python.org/downloads)

---

#### [nuitka推荐教程](https://zhuanlan.zhihu.com/p/133303836)
