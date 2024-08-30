# Large Language Model-based Actor-Critic
This is the implementation of Large Language Model-based Actor-Critic (LLaMAC) for our paper accepted by LLM Agents Workshop@ICLR2024: [Controlling large language model-based agents for large-scale decision-making: An actor-critic approach](https://arxiv.org/abs/2311.13884).


## Requirements
Please install the following Python packages.
```
pip install numpy openai re random time copy tiktoken
```

Then you need to get your OpenAI key from https://beta.openai.com/
Put that OpenAI key starting 'sk-' into the LLM.py, line8

## Create testing trial environments
Run the env1_create.py/env2_create.py to create the environments, remember change the Code_dir_path in the last lines.

```
python env1_create.py
```

## Usage
Run the env1-box-arrange.py/env2-box-arrange.py to test our approaches in different frameworks and dialogue history methods. In around Line270, set up the models(GPT-3/4), frameworks (LLAMAC,HMAS-2,HMSA-1, DMAS,CMAS), dialogue history method, and your working path dir. Then run the script:

```
python env1-box-arrange.py
```


##　Citation
```
@article{zhang2023controlling,
  title={Controlling large language model-based agents for large-scale decision-making: An actor-critic approach},
  author={Zhang, Bin and Mao, Hangyu and Ruan, Jingqing and Wen, Ying and Li, Yang and Zhang, Shao and Xu, Zhiwei and Li, Dapeng and Li, Ziyue and Zhao, Rui and others},
  journal={arXiv preprint arXiv:2311.13884},
  year={2023}
}
```



