# LatexPractice

## [Tex Live](https://www.tug.org/texlive)

### 通过图形化界面安装Tex Live

```bash
$ wget -c https://mirror.ctan.org/systems/texlive/tlnet/install-tl.zip
$ unzip install-tl.zip
# cd到解压目录
# note: 该图形化界面可以选择镜像源
$ sudo ./install-tl -gui
```

#### 添加环境变量

```bash
# 配置latex环境
$ export MANPATH=${MAT_PATH}:"/usr/local/texlive/2022/texmf-dist/doc/man" 
$ export INFOPATH=${INFOPATH}:"/usr/local/texlive/2022/texmf-dist/doc/info" 
$ export PATH=${PATH}:"/usr/local/texlive/2022/bin/x86_64-linux"
```

.. note:: 注意避免同时apt安装texlive，否则会有版本冲突问题，导致某些包无法找到

## 工作流

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220526234810223.png" alt="image-20220526234810223" style="zoom: 67%;" />

### 生成aux文件

```bash
$ pdflatex <file.tex>
# 指定编译产物导出的路径（需要该文件夹已创建 / 文件名需要放在命令行option的后面）
$ pdflatex -output-directory=build template.tex 
```

- aux文件包含了latex文件(.tex)的\cite等argument和相关的元信息（比如tex文件需要哪些参考文献，这些参考文献，latex文件中的顺序，导入参考文献的参考位置）
- tex中的文献索引暂时用`?`来替代(reference will show `?` for now)

![image-20220525090301599](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220525090301599.png)

### 生成bbl文件

bibtex根据aux的元信息，从bib文件中提取相关的参考文献并进行格式化，生成bbl文件

```bash
# 可加可不加.aux后缀
$ bibtex <file>
```

- 相关日志信息可参考.blg文件
- 编译诊断顺序，可参考[details](https://tex.stackexchange.com/questions/63852/question-mark-or-bold-citation-key-instead-of-citation-number)

### 增加reference

```bash
$ pdflatex -output-directory=build template.tex 
```

### 增加citation

```bash
$ pdflatex -output-directory=build template.tex 
```

## 编译工具

### [Biber](https://github.com/plk/biber)

- 安装（TexLive安装了即有Biber）

![image-20220526100644134](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220526100644134.png)

### Latexmk

```bash
# 查看编译选项
$ latexmk -h
$ latexmk -showextraoptions
```

## Semantic

### 公式对齐

```latex
\begin{aligned}
 a + b + c &= d \
 e + f &= g  
\end{aligned}
```

### 公式编号

```latex
% 注意加*在equation后不生成公式编号
\begin{equation}
  a+b=\gamma\label{eq}
\end{equation}
```

### 图片插入

```latex
% 需要特定格式的图片，不是所有图片格式都能用
% width项用于调整图片大小
\begin{figure}[htbp]
  \includegraphics[width=7cm]{elbow_robot_arm.png}
  \caption{肘型机械臂}
\end{figure}
```

### [文本颜色](https://tex.stackexchange.com/questions/17104/how-to-change-color-for-a-block-of-texts)

```latex
\usepackage{xcolor}
\begin{document}

This is a sample text in black.
\textcolor{blue}{This is a sample text in blue.}

\end{document}
```

### 文本居中

```latex
\centerline{$r=x_4^2+y_4^2$}
```

### [字体大小](https://blog.csdn.net/zou_albert/article/details/110532165)

### 字体类型

- `\cal`[花体](https://www.cnblogs.com/xiaofeisnote/p/13423726.html)  ；`\mathbb` [空体](https://www.overleaf.com/learn/latex/Mathematical_fonts)

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/uBiXd1DVMqM5e3o5.png!thumbnail" alt="img" style="zoom:50%;" />

### [表格](https://albertyzp.github.io/2019/10/15/LaTex%E5%9F%BA%E7%A1%80%E6%89%8B%E5%86%8C/#6-%E8%A1%A8%E6%A0%BC)

- chktex标准倾向于在表格中不添加竖线

```latex
\begin{table}[htb]
  \centering
  \caption{This is a table caption}\label{tab:ref}
  \begin{tabular}{llllllll}
    \toprule
    Tag type & NDEF & Secure messaging & SDM & Random ID & Digital Sig. & Authentication & Memory access protection \\
    \midrule
    NT4H2421Gx & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
    NTAG21x    & \checkmark & & & & \checkmark & & \checkmark \\
    NTAG210 &  & & & & \checkmark & & \\
    \bottomrule
  \end{tabular}
\end{table}
```

### 构建引用

```latex
\bibliographystyle{IEEEtran} 
\bibliography{<.bst文件名>}
```

## IDE

### Textify

for Jetbrain; 使用内置pdf需要再下一个pdf viewer插件

#### [特性](https://github.com/Hannah-Sten/TeXiFy-IDEA/wiki/Features#bibtex-1)

#### 实战

- [源文件需要和build文件放在一起](https://github.com/Hannah-Sten/TeXiFy-IDEA/wiki/BibTeX#troubleshooting)

### [Texstudio](http://texstudio.sourceforge.net/)

### Vscode

#### [LaTeX Workshop](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install#usage)

#### Chktex

[语法检查工具](https://www.nongnu.org/chktex/)；安装tex live后自带

![image-20220508214254785](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20220508214254785.png)

- [Use \\( ... \\) instead of \$ ... \$](https://tex.stackexchange.com/questions/510/are-and-preferable-to-dollar-signs-for-math-mode)

- Delete this space to maintain correct pagereferences.

```latex
\caption{Eg} \label{fig:eg} % wrong
\caption{Eg}\label{fig:eg}  % true
```

#### [格式化](https://github.com/James-Yu/LaTeX-Workshop/wiki/Format#LaTeX-files)

安装tex live后自带，ctrl+shirt+I触发

#### Code Spell Checker

- 词汇补全和正确性校验

#### LTeX

latex/ markdown的文本语法检查器

#### 同步pdf和latex文本的位置

根据pdf定位到latex的位置：ctrl+点击pdf某个位置

根据latex位置定位到pdf的位置：命令行SyncTeX

## 实战

### [IEEE中文模板](https://blog.csdn.net/qq_34447388/article/details/86488686)

### Incompatible Problem

- [LaTeX Error: File `newtxmath.sty' not found.](https://tex.stackexchange.com/questions/251405/problem-with-new-mnras-style-files-newtx-on-arxiv)
- [Package xcolor Warning: Incompatible color definition on line xxx](https://tex.stackexchange.com/questions/150369/incompatible-color-definition-when-using-tikz-with-color-package)

## IEEE模板

### 关键词

```latex
\begin{IEEEkeywords}
  Dynamic trajectory planning, MPC, obstacle avoidance.
\end{IEEEkeywords}
```

### 贡献分段

```latex
\begin{enumerate}
  \item ...
  \item ...
\end{enumerate}
```

## 拓展包

### [插入pdf文件](https://blog.csdn.net/bendanban/article/details/51850659)

```latex
\documentclass[a4paper]{article}
\usepackage{pdfpages}
\begin{document}
\includepdf[pages={1,2}]{example.pdf} 
\end{document}
```

### [伪代码库](https://tex.stackexchange.com/questions/29429/how-to-use-algorithmicx-package)

```latex
\usepackage{algorithm} % http://ctan.org/pkg/algorithms
\usepackage{algpseudocode} % http://ctan.org/pkg/algorithmicx
```

### [文本高亮](https://zhuanlan.zhihu.com/p/354838863)

```latex
\usepackage{soul}
\hl{...}
```

## 拓展插件

### [CTEX](http://www.ctex.org/HomePage)

支持中文的拓展插件

### 格式化

- latexindent

```bash
$ latexindent a.tex -o b.tex
```

## 拓展资料

- [latex 使用说明](https://albertyzp.github.io/)

- [awesome latex](https://asmcn.icopy.site/awesome/awesome-LaTeX/)
