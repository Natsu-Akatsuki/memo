# MyTmux

tmux配置

## Installation

```bash
$ git clone https://github.com/Natsu-Akatsuki/MyTmux
$ cd MyTmux
$ ln -s -f $(pwd)/.tmux.conf ~/.tmux.conf
# 安装tmux插件管理器
$ git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
$ tmux
# 创建新终端后使用"前导符+I"来安装插件
$ tmux source ~/.tmux.conf
```

## Example

```bash
$ sudo ln -s $(pwd)/example/* /usr/local/bin/
$ chmod +x  $(pwd)/example/*
```
