# Workbench
## Conda
``` py
conda create -n myenv python=3.9 # Create an environment
conda activate myenv # Enter the environment
conda install python=3.10 -c <channel> # Install packages
conda remove python # Remove packages
conda deactivate # Exit the environment
conda info --envs # List all envs
conda remove -n myenv --all # Remove the environment
conda list
```

## Pip
``` py
python -m pip install packages
```
## Screen
``` py
screen -ls  # list 
screen -S name # create 
command+A D # quit
screen -r name # go back
screen -X -S name quit # delete

```
## tmux
```py
tmux new -s baby # 创建新的会话a
control+B D # 退出当前状态
tmux attach -t baby # 重新进入
```
## Errors and Warnings
### RuntimeError: CUDA out of memory. Tried to allocate .. MiB
在测试阶段和valid阶段插入代码```with torch.no_grad()```
```py
def test(model,dataloader):
    model.eval()
    with torch.no_grad(): ###插在此处
        for batch in tqdm(dataloader):
			……

```
