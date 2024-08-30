from LLM import *
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")
input_prompt_token_limit = 3000
from sre_constants import error

def LLM_summarize_func(state_action_prompt_next_initial):
  prompt1 = f"Please summarize the following content as concise as possible: \n{state_action_prompt_next_initial}"
  messages = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt1}]
  response = GPT_response(messages, model_name='gpt-4')
  return response

collision_avoidance_prompt = '[Do remember that each corner can only contain at most one box! Hence, you need to avoid the collision of boxes. Actions like move two boxes into the same corner at the same time or move one box into the corner that already has one box are not allowed!]'

def explore_and_exploit_prompt_func(state_update_prompt, response_total_list,
                                                pg_state_list, dialogue_history_list,
                                                dialogue_history_method, cen_decen_framework):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' or cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

    The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

    Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    You need to tell me two plans: one is more far-sighted and consistent with long-term benefits, one is more in line with the current benefits. 

    Specify your action plan in this format: 
    Plan 1: {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[1.5, 0.5]":"move...}} 
    Plan 2: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.0, 1.0])", "Agent[1.5, 0.5]":"move...}}

    Remember:
    1. Make sure your assigned task for each agent is in doable action list.
    2. Avoid placing the box where another box is already present, or moving multiple boxes to the identical position! Otherwise, your mission will directly fail!
    3. Do not let different agents move the same box in one plan!
    
    Hence, the current state is {pg_state_list[-1]}, with the possible actions:
    {state_update_prompt}

    Include an agent only if it has a task next. Do not output additional content. Now, plan the next step:
      '''

  return user_prompt_1


def judge_prompt_func(state_update_prompt, response_total_list,
                                                pg_state_list, dialogue_history_list,
                                                dialogue_history_method, cen_decen_framework, plan):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' or cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

    The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

    Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    Hence, the current state is {pg_state_list[-1]}, with the possible actions:
    {state_update_prompt}

    Here are two plans that may be useful for you to consider:
    {plan}

    Remember:
    1. Make sure your assigned task for each agent are in possible action list!
    2. Avoid placing the box where another box is already present, or moving multiple boxes to the identical position! Otherwise, your mission will directly fail!
    3. Do not let different agents move the same box in one plan!
    Now, specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.0, 1.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Do not output additional content. Now, plan the next step:
      '''

  return user_prompt_1





def input_prompt_1_func(state_update_prompt):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
  '''
  return user_prompt_1


def input_prompt_1_only_state_action_func(state_update_prompt, response_total_list, pg_state_list):
  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  state_action_prompt = ''
  for i in range(len(response_total_list) - 1, -1, -1):
    state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
    if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
      state_action_prompt = state_action_prompt_next
    else:
      break

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.
  
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  return user_prompt_1


def input_prompt_1_func_total(state_update_prompt, response_total_list,
                                                pg_state_list, dialogue_history_list,
                                                dialogue_history_method, cen_decen_framework):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1 and cen_decen_framework != 'CMAS':
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

  The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

  Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  Hence, the current state is {pg_state_list[-1]}, with the possible actions:
  {state_update_prompt}

  Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
    '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history' or cen_decen_framework == 'CMAS':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history' and cen_decen_framework != 'CMAS':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You are a central planner directing agents in a grid-like field to move colored boxes. Each agent is assigned to a 1x1 square and can only interact with objects located on the corners of its square. Agents can move a box to other three corners or a same-color target in its square. Each square can contain many targets.

    The squares are identified by their center coordinates, e.g., square[0.5, 0.5]. Actions are like: move(box_red, target_red) or move(box_red, position[1.0, 0.0]). {collision_avoidance_prompt}

    Your task is to instruct each agent to match all boxes to their color-coded targets. After each move, agents provide updates for the next sequence of actions. Your job is to coordinate the agents optimally.

    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

    Hence, the current state is {pg_state_list[-1]}, with the possible actions:
    {state_update_prompt}

    Specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move...}}. Include an agent only if it has a task next. Now, plan the next step:
      '''

  return user_prompt_1

def input_prompt_local_agent_DMAS_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, dialogue_history, response_total_list,
                                                                         pg_state_list, dialogue_history_list,
                                                                         dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''  
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''
  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in ('_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        #state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  All the agents coordinate with others together to come out a plan and achieve the goal: match each box with its color-coded target. {collision_avoidance_prompt}
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
    '''

  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_fast_plan_func(state_update_prompt_local_agent, state_update_prompt_other_agent,
                                                 dialogue_history, response_total_list, pg_state_list, dialogue_history_list,
                                                 dialogue_history_method, initial_plan=''):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
  '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:
  {state_action_prompt}
  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by outputting the final plan, must strictly follow [Action Output Instruction]!
  Your response:
    '''
  return user_prompt_1


def input_prompt_local_agent_HMAS1_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, dialogue_history, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method, initial_plan = ''):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.


  [Action Output Instruction]
  Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
  Include an agent only if it has a task next.
  Example#1: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}

  Example#2: 
  EXECUTE
  {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}

  The initial plan is: {{{initial_plan}}}
  The previous dialogue history is: {{{dialogue_history}}}
  Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
  Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
  Propose exactly one action for yourself at the **current** round.
  {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
  End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
  Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
          '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
    
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
  
    [Action Output Instruction]
    Must first output 'EXECUTE', then on the new line specify your action plan in this format: {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move..."}}.
    Include an agent only if it has a task next.
    Example#1: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, position[0.0, 2.0])", "Agent[1.5, 0.5]":"move(box_green, position[0.0, 0.0])"}}
  
    Example#2: 
    EXECUTE
    {{"Agent[0.5, 0.5]":"move(box_blue, target_blue)", "Agent[2.5, 1.5]":"move(box_red, position[0.0, 2.0])"}}
  
    The initial plan is: {{{initial_plan}}}
    The previous dialogue history is: {{{dialogue_history}}}
    Think step-by-step about the task, initial plan, and the previous dialogue history. Carefully check and correct them if they made a mistake.
    Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
    Propose exactly one action for yourself at the **current** round.
    {collision_avoidance_prompt} Avoid the situation that the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step.
    End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan as soon as possible, must strictly follow [Action Output Instruction]!
    Your response:
    '''
  return user_prompt_1

def input_prompt_local_agent_HMAS2_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, central_response, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.

  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  {collision_avoidance_prompt} Please check the given plan, especially avoiding the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
          '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break

    user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
    The previous state and action pairs at each step are:
    {state_action_prompt}
    Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.
    
    The current state is {pg_state_list[-1]}
    The central planner\'s current action plan is: {{{central_response}}}.
  
    {collision_avoidance_prompt} Please check the given plan, especially avoiding the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
    '''
  return user_prompt_1

def input_prompt_local_agent_my_dialogue_func(state_update_prompt_local_agent, state_update_prompt_other_agent, central_response, response_total_list, pg_state_list, dialogue_history_list, dialogue_history_method, target_location, bos_pos, collision_feedback):
  if len(pg_state_list) - len(response_total_list) != 1:
    raise error('state and response list do not match')
  if len(pg_state_list) - len(dialogue_history_list) != 1:
    raise error('state and dialogue history list do not match')

  user_prompt_1 = f'''
  You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
  A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.

  The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
  The current states and possible actions of all other agents are: {{{state_update_prompt_other_agent}}}.
  The previous state and action pairs at each step are:

  Please learn from previous steps. Not purely repeat the actions but learn why the state changes or remains in a dead loop. Avoid being stuck in action loops.

  The current state is {pg_state_list[-1]}
  The central planner\'s current action plan is: {{{central_response}}}.

  {collision_avoidance_prompt} Please check the given plan, especially avoiding the box you are moving will collide with other boxes in the corner. Avoid the case that two boxes move to the same corner at the same step. If you agree with it, respond 'I Agree', without any extra words. If not, briefly explain your objections to the central planner. Your response:
  '''

  token_num_count = len(enc.encode(user_prompt_1))

  if dialogue_history_method == '_wo_any_dialogue_history':
    pass
  elif dialogue_history_method in (
          '_w_only_state_action_history', '_w_compressed_dialogue_history', '_w_all_dialogue_history'):
    if dialogue_history_method == '_w_only_state_action_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_compressed_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        dialogue_summary = LLM_summarize_func(dialogue_history_list[i])
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nSummary of Dialogues in each step{i + 1}: {dialogue_summary}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        # state_action_prompt_next = LLM_summarize_func(state_action_prompt_next_initial)
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    elif dialogue_history_method == '_w_all_dialogue_history':
      state_action_prompt = ''
      for i in range(len(response_total_list) - 1, -1, -1):
        state_action_prompt_next = f'State{i + 1}: {pg_state_list[i]}\nDialogue{i + 1}: {dialogue_history_list[i]}\nAction{i + 1}: {response_total_list[i]}\n\n' + state_action_prompt
        if token_num_count + len(enc.encode(state_action_prompt_next)) < input_prompt_token_limit:
          state_action_prompt = state_action_prompt_next
        else:
          break
    if collision_feedback != '':
      collision_feedback_prompt = f'Actually, the given plan will directly cause a collision. {collision_feedback}'
    else:
      collision_feedback_prompt = ''

    user_prompt_1 = f'''
    You\'re a box-moving agent in a multi-agent system, stationed on a 1x1 square in a grid playground. You can only interact with objects located on the corners of its square. Squares are denoted by their center coordinates (e.g., square[0.5, 0.5]), and actions involve moving boxes to targets or other three corners, represented by colors (e.g., move(box_red, target_red)). Each square can contain many targets.
    A central planner coordinates all agents to achieve the goal: match each box with its color-coded target.
  
    The current state and possible actions of yourself are: {{{state_update_prompt_local_agent}}}.
    The central planner\'s current action plan for you is: {{{central_response}}}.
    {bos_pos}
    The target location of box in your square are: {target_location}.


    The current state is {pg_state_list[-1]}

    {collision_feedback_prompt}
    Central planner seems to take the box further away from the target. If so, please briefly propose a better solution (in the doable action list) to the central planner.
    Your response:
    '''
  return user_prompt_1


def input_reprompt_func(state_update_prompt):
  user_reprompt = f'''
  Finished! The updated state is as follows(combined targets and boxes with the same color have been removed):

  {state_update_prompt}

  The output should be like json format like: {{Agent[0.5, 0.5]:move(box_blue, position[0.0, 1.0]), Agent[1.5, 0.5]:move...}}. If no action for one agent in the next step, just do not include its action in the output. Also remember at most one action for each agent in each step. {collision_avoidance_prompt}

  Next step output:
  '''
  return user_reprompt

def message_construct_func(user_prompt_list, response_total_list, dialogue_history_method):
  if f'{dialogue_history_method}' == '_w_all_dialogue_history':
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    #print('length of user_prompt_list', len(user_prompt_list))
    for i in range(len(user_prompt_list)):
      messages.append({"role": "user", "content": user_prompt_list[i]})
      if i < len(user_prompt_list)-1:
        messages.append({"role": "assistant", "content": response_total_list[i]})
    #print('Length of messages', len(messages))
  elif f'{dialogue_history_method}' in ('_wo_any_dialogue_history', '_w_only_state_action_history'):
    messages=[{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": user_prompt_list[-1]})
    #print('Length of messages', len(messages))
  return messages
