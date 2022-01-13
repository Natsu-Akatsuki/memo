# 00. 基础案例

``` yaml
# This is a basic workflow to help you get started with Actions
name: CI

# Controls when the workflow will run
on:
  # 定义触发事件和指定的branch
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ] 
  # 允许手动触发
  workflow_dispatch:

# 一个workflow run由一个或多个jobs组成，这些job或并行或串行执行
jobs:  
  # 设计了一个叫做build的job
  build:
    # 指定操作系统
    runs-on: ubuntu-latest
    # 指定steps(sequence of tasks) of jobs
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE, so your job can access it
      - uses: actions/checkout@v2
          
      # 使用runner的shell来执行一条指令
      - name: Run a one-line script
        run: echo Hello, world!
    
      # 使用runner的shell来执行多条指令
      - name: Run a multi-line script
        run: |
          echo Add other actions to build,
          echo test, and deploy your project.      
```



# 01. ReadTheDocs

```yaml
- name: Sphinx Docs build with ReadTheDocs Docker
  # You may pin to the exact commit or the version.
  # uses: DavidLeoni/readthedocs-to-actions@15481740b436da601fcacd634f5ae68a9a1f3ac0
  uses: DavidLeoni/readthedocs-to-actions@v1.2
  with:
    # ReadTheDocs project name - also used as name for pdfs and epubs. 
NOTE: you don't need to actually have a project on readthedocs servers!    

    RTD_PRJ_NAME: # default is myproject
    # Full git url to clone the repo
    GIT_URL: 
    # tag or branch
    GIT_TAG: # optional, default is master
    # version as named on the website
    VERSION: # optional, default is latest
    # requirements file for pip install
    REQUIREMENTS: # optional, default is requirements.txt
    # Documentation language
    LANGUAGE: # optional, default is en
    # builds single page html for offline use. Requires built project to have readthedocs_ext.readthedocs  sphinx extension
    RTD_HTML_SINGLE: # optional, default is true
    # builds html exactly as in RTD website. Requires built project to have readthedocs_ext.readthedocs sphinx extension
    RTD_HTML_EXT: # optional, default is true
    # If you want to make Sphinx believe you are running on ReadTheDocs server, set this to 'True' (as RTD server does). NOTE: variable MUST be set to a string with capital first character, like 'True' or 'False'!
    READTHEDOCS: # optional, default is True
    # A code like UA-123-123-123
    GOOGLE_ANALYTICS: # optional, default is 
```

