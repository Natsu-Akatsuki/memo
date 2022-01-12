# Password Manage

## 基于命令行的密码管理([pass](https://wiki.archlinux.org/title/Pass))

### 安装

```bash
$ sudo apt install pass
```

### 配置gpg

由于pass是基于gpg进行加密和解密的，因此需要配置gpg生成私钥和公钥

```bash
# 或者 gpg --gen-key
$ gpg --full-generate-key
```

### [gpg分享密钥](https://unix.stackexchange.com/questions/481939/how-to-export-a-gpg-private-key-and-public-key-to-a-file)

```bash
# 显示已有的密钥
$ gpg --list-secret-keys

# 方法一：导出密钥与导入密钥
$ gpg --export-secret-key <gpg_id> > <file_name.asc>
$ gpg --import <file_name.asc>

# 方法二：利用ssh+管道操作直接一步到位（完成导出与导入）
gpg --export-secret-key -a | ssh helios@10.23.21.164 gpg --import -
```

### 常用命令行

* 显示已存储的密码名(pass_name)、当前密码

```bash
$ pass
# 显示密码
$ pass <pass_name>
# 将密码拷贝到粘贴板
$ pass -c <pass_name>
```

<img src="https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909125637720.png" alt="image-20210909125637720" style="zoom:67%; " />

.. hint:: <pass_name>与目录树有对应关系

* 加、修、删密码

```bash
# 添加密码
$ psss insert <pass_name>
# 修改密码
$ pass edit <pass_name>
# 删除密码
$ pass rm <pass_name>
```

### 初始化pass

存储和读取任何密码都需要进行初始化，以使能gpg进行加密和解密

```bash
$ pass init <gpg-id or email>>
```

.. note:: 密码名可以带斜杠 `/`

![image-20210909125220221](https://natsu-akatsuki.oss-cn-guangzhou.aliyuncs.com/img/image-20210909125220221.png)

.. attention:: 不需要使用sudo权限

### 远程仓同步

```bash
# 推送到远程仓
$ pass git init
$ pass git remote add origin <github_remote_repository_url>
$ pass git push <-f>
# 拉取到本地
$ git clone <github_remote_repository_url> ~/.password-store
```

.. note:: 这种比git命令行多了个pass的优势在于可以不用cd到对应文件夹就能进行git操作
