import os
from flask import Flask, request
import json
from json import loads
import numpy as np
import gym
from gym import wrappers
#from rlgraph.agents import Agent
#from rlgraph.environments import OpenAIGymEnv
#from gym.envs.registration import register
import data
import linear_model
import bot
import threading
#import store
#from datetime import date
from datetime import datetime as dt
from pytz import timezone
import pytz

# ============== load offline policy ======================
from offline.deploy import load_pytorch_agent, DeployPolicy

policy = load_pytorch_agent()
print("\n******************* Loaded the Offline policy *******************\n")
# =========================================================

date_format='%m-%d-%Y-%H:%M:%S-%Z'
date = dt.now(tz=pytz.utc)
print('Current date & time is:', date.strftime(date_format))

date = date.astimezone(timezone('US/Pacific'))
now = date.strftime(date_format)

print('Local date & time is  :', now)

#tf.get_logger().setLevel('ERROR')

#today = date.today()
#today = '20200810'

# dd/mm/YY
# today = today.strftime("%Y-%m-%d %H:%M:%S")
# print("today =", today)



update_count = 0
user_count = 0

# create the environment
# print("Checking gym env {}, specs = {}".format(gym.envs.registry, gym.envs.registry.env_specs))
# import code
# code.interact(local=locals())

# comment out for offline policy
# try:
#     register(
#         id='SmartPrimer-v1',
#         entry_point='envs:SmartPrimerEnv',
#     )
# except:
#     pass

# env = OpenAIGymEnv.from_spec({
#     "type": "openai",
#     "gym_env": 'SmartPrimer-v1'
# })

# print("\n******************* Created the Environment *******************\n")

# # create the PPO agent
# agent_config_path = 'ppoSmartPrimer_config.json'

# with open(agent_config_path, 'rt') as fp:
#     agent_config = json.load(fp)

agent = None
f = None
glove_model = None
helpwords = None


def log_print(info, file):
    print(info)
    file.write(info+"\n")


def loadAgent():
    # comment out for offline policy
    # global agent
    # agent = Agent.from_spec(
    #     agent_config,
    #     state_space=env.state_space,
    #     action_space=env.action_space)
    # if not os.path.exists('./store'):
    #     print("\n******************* Create an empty store folder *******************\n")
    #     os.mkdir('./store')
    # #print("\n******************* Try to download the model and logs from AWS *******************\n")
    # #store.download_store()
    # if os.path.exists('./store/checkpoint'):
    #     agent.load_model('./store')
    #     print("\n******************* Load the model from the last checkpoint *******************\n")
    #     # check for nan
    #     w = agent.get_weights()
    #     print(w)
    # else:
    #     print("\n!!!!!!!!!!!!!!!!!!!!! Store Folder is Empty !!!!!!!!!!!!!!!!!!!!!\n")

    print("\n******************* Start Logging *******************\n")
    global f
    f = open("./store/log.txt", "a+")
    log_print("\nStart logging: " + now + "\n", f)

    global glove_model, helpwords
    glove_model, helpwords = data.loadGloveModel("./glove.6B.50d.txt")
    log_print("\n******************* Loaded Glove *******************\n", f)
    # print("\n******************* Created the PPO Agent *******************\n")
    # if agent is None:
    #     print("\n!!!!!!!!!!!!!!!!!!!!! ERROR! NONE AGENT !!!!!!!!!!!!!!!!!!!!!\n")


# comment out for offline policy
#thread = threading.Thread(target=loadAgent)
#thread.start()
loadAgent()

app = Flask(__name__)
print("\n******************* Initialized the Flask App *******************\n")

data.fetch_all_hints()
print("\n******************* Fetched all the Hints *******************\n")

###
#   bot will be available only after agent is loaded, otherwise agent is None
###

@app.route('/generate', methods=['POST'])
def generate():

    #print("*** debug: generate()")

    # extract input message
    message = request.json['event']['data']['new']
    if message['from'] == 1 or message['tutorial'] == True:
        return ('', 200)

    data.fetch_all_users()
    data.fetch_all_responses()
    data.fetch_all_events()

    global f
    global glove_model
    global helpwords

    # generate response (hint_id = -1 indicates error, thus nothing will be sent to the user)
    #hint_id = linear_model.generate_response(message)
    hint_id = bot.generate_response(message, agent, f, glove_model, helpwords, policy)

    reply = [hint['answer'] for hint in data.hints if hint["id"] == hint_id][0]
    log_print("Return: " + str(hint_id) + " " + loads(reply)['blocks'][0]['text'] + "\n", f)

    if hint_id == -1:
        return ('', 200)
    else:
        payload = json.dumps({
            "text": reply
        })
    params = {
        "hint_id": hint_id,
        "input_id": message["id"],
        "payload": payload,
        "to": message["from"]
    }
    data.insert_response(params)  # comment out to avoid error in postman

    #print("\n"+"-"*80+"\npayload: "+params["payload"]+"\n"+"-"*80+"\n")
    return ('', 200)


@app.route('/update', methods=['POST'])
def update():
    # extract input message
    test = request.json['event']['data']['new']

    # model updating may need to fetch some of the following
    data.fetch_all_users()
    data.fetch_all_responses()
    data.fetch_all_events()

    # generate response
    global user_count, update_count, f, glove_model, helpwords
    #user_count, update_count = bot.update(agent, test, user_count, update_count, f, glove_model, helpwords)
    #log_print("User Count since last restart: " + str(user_count), f)
    #log_print("Update Count since last restart: " + str(update_count), f)

    # comment out for offline policy
    #agent.reset()
    #log_print("\n" + "-" * 20 + " Agent is Reset " + "-" * 20, f)

    date = dt.now(tz=pytz.utc)
    print('Current date & time is:', date.strftime(date_format))
    date = date.astimezone(timezone('US/Pacific'))
    now = date.strftime(date_format)

    # comment out for offline policy
    #agent.store_model(path="./store/model-"+now)
    #log_print("\n" + "-" * 20 + " Agent is Stored " + "-" * 20, f)
    #log_print("./store/model-"+now+"\n", f)

    f.close()
    f = open("./store/log.txt", "a+")
    #store.upload_store()
    #log_print("\n" + "-" * 20 + " Upload the model and log to AWS " + "-" * 20 +"\n", f)

    return (str(test['post_score']), 200)






