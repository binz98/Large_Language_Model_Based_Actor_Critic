import os
import numpy as np
folder_path = 'folder/path'
env = 'Env2_BoxNet2/env_pg_state_2_2/'
state = 'pg_state0/'
method = 'my_w_only_state_action_history/'


file_names = ['env_action_times.txt', 'feed_back_times.txt', 'token_num_count.txt'] #'success_failure.txt',
# 遍历文件列表
for file_name in file_names:
    num = [] 
    success=0
    for i in range(10):
        state = f'pg_state{i}/'
        path = folder_path + env + state + method
        file_path = os.path.join(path, file_name)
        succes_path = os.path.join(path, 'success_failure.txt')
    
        # 检查文件是否存在
        if os.path.exists(file_path):
            # 打开文件并读取内容
            with open(succes_path, 'r') as file:
                content = file.read()
                if content == 'success':
                    print(f"pg_state{i} success")
                    success += 1
                    with open(file_path, 'r') as file:
                        content = file.read()
                        if file_name == 'token_num_count.txt':
                            token_num = content.splitlines()
                            token_sum = sum(int(line) for line in token_num)
                            num.append(token_sum)
                        else:
                            num.append(int(content))
        else:
            print('文件', file_name, '不存在')

            
    # 打印统计结果
    print('-------------------------')
    print('成功率:', success/10)
    num = np.array(num)
    print('文件名:', file_name)
    print('结果:', num)
    print(f'统计值:{num.mean()}({num.std()})')
    print('-------------------------')
