# Box moving to target without collision

from prompt_env1 import *
from LLM import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import ast
import math

def surround_index_func(row_num, coloum_num, row_index, coloum_index):
  surround_index_list = []
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    if i>=0 and i<=row_num-1 and j>=0 and j<=coloum_num-1 and not (i == row_index and j == coloum_index):
      surround_index_list.append([i,j])
  return surround_index_list

def transform_string(string):
    x, y = map(int, string.split('_'))
    return f'square[{x}, {y}]'

def state_update_func(pg_row_num, pg_column_num, pg_dict):
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt = ''
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      square_item_list = pg_dict_copy[str(i)+'_'+str(j)]
      square_item_only_box = [item for item in square_item_list if item[:3]=='box']
      surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
      state_update_prompt += f'Agent[{i}, {j}]: I am in square[{i}, {j}], I can observe {square_item_list}, I can do '
      action_list = []
      for box in square_item_only_box:
        for surround_index in surround_index_list:
          action_list.append(f'move({box}, square{surround_index})')
        if 'target'+box[3:] in square_item_list:
          action_list.append(f'move({box}, target{box[3:]})')
      state_update_prompt += f'{action_list}\n'
  return state_update_prompt

def state_update_func_local_agent(pg_row_num, pg_column_num, pg_row_i, pg_column_j, pg_dict):
  '''
  为本智能体以及其它智能体加入可用动作,将box移动到相邻位置,将box移动到target中
  '''
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt_local_agent = ''
  state_update_prompt_other_agent = ''

  for i in range(pg_row_num):
    for j in range(pg_column_num):
      if not (i == pg_row_i and pg_column_j == j):
        square_item_list = pg_dict_copy[str(i)+'_'+str(j)]
        square_item_only_box = [item for item in square_item_list if item[:3]=='box']
        surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)  #相邻位置的square
        state_update_prompt_other_agent += f'Agent[{i}, {j}]: I am in square[{i}, {j}], I can observe {square_item_list}, I can do '
        action_list = []
        for box in square_item_only_box:
          for surround_index in surround_index_list:
            action_list.append(f'move({box}, square{surround_index})')
          if 'target'+box[3:] in square_item_list:
            action_list.append(f'move({box}, target{box[3:]})')
        state_update_prompt_other_agent += f'{action_list}\n' #加入可用动作,将box移动到相邻位置,将box移动到target中

  square_item_list = pg_dict_copy[str(pg_row_i)+'_'+str(pg_column_j)]
  square_item_only_box = [item for item in square_item_list if item[:3]=='box']
  surround_index_list = surround_index_func(pg_row_num, pg_column_num, pg_row_i, pg_column_j)
  state_update_prompt_local_agent += f'Agent[{pg_row_i}, {pg_column_j}]: in square[{pg_row_i}, {pg_column_j}], can observe {square_item_list}, can do '
  action_list = []
  target_list = []
  for box in square_item_only_box:
    for surround_index in surround_index_list:
      action_list.append(f'move({box}, square{surround_index})')
    if 'target'+box[3:] in square_item_list:
      action_list.append(f'move({box}, target{box[3:]})')
    location_of_target_box = [transform_string(key) for key, values in pg_dict_copy.items() if f'target_{box[4:]}' in values]
    target_list.append(f'target_{box[4:]} is in {location_of_target_box}')
  state_update_prompt_local_agent += f'{action_list}\n'
  return state_update_prompt_local_agent, state_update_prompt_other_agent, target_list

def with_action_syntactic_check_func(pg_dict_input, response, user_prompt_list_input, response_total_list_input, model_name, dialogue_history_method, cen_decen_framework):
  '''
  用于检测central planner生成的动作是否可用,若不可用则生成反馈信息.
  '''
  FB = False
  user_prompt_list = copy.deepcopy(user_prompt_list_input)
  response_total_list = copy.deepcopy(response_total_list_input)
  iteration_num = 0
  token_num_count_list_add = []
  while iteration_num < 6:
    response_total_list.append(response)
    try:
      original_response_dict = json.loads(response)
      # if original_response_dict == {}:
      #   feedback = 'There are still boxes that have not been moved to the target, but you do not provid a plan for any agent.'
      # else:
      pg_dict_original = copy.deepcopy(pg_dict_input)
      transformed_dict = {}
      for key, value in original_response_dict.items():
        coordinates = tuple(map(int, re.findall(r"\d", key)))

        # match the item and location in the value
        match = re.match(r"move\((.*?),\s(.*?)\)", value)
        if match:
          item, location = match.groups()

          if "square" in location:
            location = tuple(map(int, re.findall(r"\d", location)))

          transformed_dict[coordinates] = [item, location]

      feedback = ''
      for key, value in transformed_dict.items():
        # print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
        if value[0] in pg_dict_original[str(key[0]) + '_' + str(key[1])] and type(value[1]) == tuple and (
                (np.abs(key[0] - value[1][0]) == 0 and np.abs(key[1] - value[1][1]) == 1) or (
                np.abs(key[0] - value[1][0]) == 1 and np.abs(key[1] - value[1][1]) == 0)):
          pass
        elif value[0] in pg_dict_original[str(key[0]) + '_' + str(key[1])] and type(value[1]) == str and value[1] in \
                pg_dict_original[str(key[0]) + '_' + str(key[1])] and value[0][:4] == 'box_' and value[1][
                                                                                                :7] == 'target_' and \
                value[0][4:] == value[1][7:]:
          pass
        else:
          # print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
          feedback += f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '
    except:
      # return 'wrong json format', token_num_count_list_add, True
      raise error(f'The response in wrong json format: {response}')
      feedback = 'Your assigned plan is not in the correct json format as before. If your answer is empty dict, please check whether you miss to move box into the same colored target like move(box_blue, target_blue)'

    if feedback != '':
      FB = True
      feedback += 'Please replan again with the same ouput format. Your only final plan:'
      print('----------Syntactic Check----------')
      print(f'Response original: {response}')
      print(f'Feedback: {feedback}')
      user_prompt_list.append(feedback)
      messages = message_construct_func(user_prompt_list, response_total_list, dialogue_history_method) # message construction
      print(f'Length of messages {len(messages)}')
      response, token_num_count = GPT_response(messages, model_name)
      token_num_count_list_add.append(token_num_count)
      print(f'Response new: {response}\n')
      if response == 'Out of tokens':
        return response, token_num_count_list_add, FB
      iteration_num += 1
    else:
      return response, token_num_count_list_add, FB
  return 'Syntactic Error', token_num_count_list_add, FB

def get_actor_feedback_or_not(local_agent_row_i,local_agent_column_j, response_agent, target_location):
    # 找到target_blue的位置
    key_agent = f'Agent[{local_agent_row_i}, {local_agent_column_j}]'
    central_task = response_agent[key_agent]
    original_position = [local_agent_row_i, local_agent_column_j]
    if "target" in central_task:
        return False
    else:
        target_positions = []
        for item in target_location:
            if re.findall(r'box_(\w+)', central_task)[0] in item:
                lst = ast.literal_eval(item.split("is in")[1].strip())
                target_positions = [list(ast.literal_eval(x.split('[')[1].split(']')[0])) for x in lst]

        # 计算原始位置与target_blue所有位置的最小距离
        min_distance_agent = math.inf
        for position in target_positions:
            # position = ast.literal_eval(position)
            distance = abs(original_position[0] - position[0]) + abs(original_position[1] - position[1])
            if distance < min_distance_agent:
                min_distance_agent = distance

        # 计算plan位置与target_blue所有位置的最小距离
        agent_position = list(ast.literal_eval(central_task.split('[')[1].split(']')[0]))
        min_distance_plan = math.inf
        for position in target_positions:
            # position = ast.literal_eval(position)
            distance = abs(agent_position[0] - position[0]) + abs(agent_position[1] - position[1])
            if distance < min_distance_plan:
                min_distance_plan = distance

        # 比较两个距离，找出哪个更大
        if min_distance_agent > min_distance_plan:
            return False
        else:
            return True



def action_from_response(pg_dict_input, original_response_dict):
  system_error_feedback = ''
  pg_dict_original = copy.deepcopy(pg_dict_input)
  transformed_dict = {}
  for key, value in original_response_dict.items():
    coordinates = tuple(map(int, re.findall(r"\d", key)))

    # match the item and location in the value
    match = re.match(r"move\((.*?),\s(.*?)\)", value)
    if match:
      item, location = match.groups()
      if "square" in location:
          location = tuple(map(int, re.findall(r"\d", location)))
      transformed_dict[coordinates] = [item, location]

  for key, value in transformed_dict.items():
    #print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
    if value[0] in pg_dict_original[str(key[0])+'_'+str(key[1])] and type(value[1]) == tuple and ((np.abs(key[0]-value[1][0])==0 and np.abs(key[1]-value[1][1])==1) or (np.abs(key[0]-value[1][0])==1 and np.abs(key[1]-value[1][1])==0)):
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[0])
      pg_dict_original[str(value[1][0])+'_'+str(value[1][1])].append(value[0])
    elif value[0] in pg_dict_original[str(key[0])+'_'+str(key[1])] and type(value[1]) == str and value[1] in pg_dict_original[str(key[0])+'_'+str(key[1])] and value[0][:4] == 'box_' and value[1][:7] == 'target_' and value[0][4:] == value[1][7:]:
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[0])
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[1])
    else:
      #print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
      system_error_feedback += f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '

  return system_error_feedback, pg_dict_original

def env_create(pg_row_num = 5, pg_column_num = 5, box_num_low_bound = 2, box_num_upper_bound = 2, color_list = ['blue', 'red', 'green', 'purple', 'orange']):
  # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
  pg_dict = {}
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      pg_dict[str(i)+'_'+str(j)] = []

  for color in color_list:
    box_num = random.randint(box_num_low_bound, box_num_upper_bound)
    for _ in range(box_num):
      N_box = random.randint(0, pg_row_num*pg_column_num - 1)
      a_box = N_box // pg_column_num
      b_box = N_box % pg_column_num
      N_target = random.randint(0, pg_row_num*pg_column_num - 1)
      a_target = N_target // pg_column_num
      b_target = N_target % pg_column_num
      pg_dict[str(a_box)+'_'+str(b_box)].append('box_' + color)
      pg_dict[str(a_target)+'_'+str(b_target)].append('target_' + color)
  return pg_dict

def create_env1(Saving_path, repeat_num = 10):
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  for i ,j in [(2,2), (2,4), (4,4), (4,8)]:

    if not os.path.exists(Saving_path+f'/env_pg_state_{i}_{j}'):
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{i}_{j}')
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)

    for iteration_num in range(repeat_num):
      # Define the total row and column numbers of the whole playground, and the item number of each colored target and box
      pg_row_num = i; pg_column_num = j; box_num_low_bound = 1; box_num_upper_bound = 3
      # Define the used colors
      color_list = ['blue', 'red', 'green', 'purple', 'orange']
      pg_dict = env_create(pg_row_num, pg_column_num, box_num_low_bound, box_num_upper_bound, color_list)
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}', exist_ok=True)
      with open(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
        json.dump(pg_dict, f)

Code_dir_path = r'C:\Users\win10\Desktop\robot_ai_control/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env1_BoxNet1'
# The first time to create the environment, after that you can comment it
# create_env1(Saving_path, repeat_num = 10)