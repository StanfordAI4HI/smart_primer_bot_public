from json import loads
import numpy as np
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import data
import json
from numpy import dot
from numpy.linalg import norm


# nltk.download('stopwords')
nltk.download('vader_lexicon')


int2action = ['hint', 'nothing', 'encourage', 'question']
action2int = {}
for index, action in enumerate(int2action):
    action2int[action] = index


def generate_response(input_message, agent, f, glove_model, helpwords, policy):
    """
    @param input_message: Message

    @return (hint_id, reply): (Int, String)
    """
    #log_print(json.dumps(input_message), f)

    log_print("\n"+"-" * 20 + " Action Selection Starts " + "-" * 20, f)

    #print(data.users)
    #print(data.events)

    user_id = input_message['from']

    user_entry = [user for user in data.users if user['id'] == user_id][0]
    # print(user_entry)
    log_print("\nUser: " + str(user_id) + " " + user_entry['username'], f)

    # text = data.extract_text_from_payload(input_message['payload'])
    input_text = loads(loads(input_message["payload"])["text"])["blocks"][0]["text"]
    log_print("Input: " + input_text, f)

    pos, neg, hel = vectorize(input_text, f, glove_model, helpwords)

    step = input_message["step"]
    if step == None or step == "":
        return -1

    stage = data.step2stage[step]
    log_print("Current Stage: " + str(stage) + " " + step, f)

    event_entry = [event for event in data.events if (event["user_id"] == user_id and event["page"] == "/tasks/"+str(stage+3))]
    #print(event_entry)
    log_print("Total Attempts: " + str(len(event_entry)), f)

    grade = user_entry['grade']
    log_print("Grade: " + str(grade), f)
    pre_score = user_entry['test'][0]['pre_score']
    log_print("Pre Score: " + str(pre_score), f)
    anxiety = user_entry['anxiety']
    log_print("Anxiety: " + str(anxiety), f)

    responses_for_user = [
        response for response in data.responses if response["input_message"]["from"] == input_message["from"]]

    responses_for_step = [
        response for response in responses_for_user if response["input_message"]["step"] == step]

    log_print('Num of previous responses for step {}: '.format(step) + str(len(responses_for_step)), f)

    hints_for_step = [
        response for response in responses_for_step if (
                response['hint'] is not None and response['hint']['id'] in data.hint_ids[step])]
    hint_for_step_count = len(hints_for_step)
    log_print('Num of previous hints for step {}: '.format(step) + str(hint_for_step_count), f)

    # ------------------------- select an action_name using PPO agent -------------------------
    current_ob = normalize(np.array((grade, pre_score, stage, len(event_entry), pos, neg, hel, anxiety), dtype=float))  # (pre_score, stage)
    log_print("Observation: " + np.array2string(current_ob), f)
    # comment out for offline policy
    # action = agent.get_action(current_ob)  # return an int
    action = policy.get_action(current_ob)
    # log_print("Action ID: " + str(action), f)


    action_name = int2action[action]
    # log_print("New Action: " + str(action) + " " + action_name, f)
    log_print("New Action: " + action_name, f)

    if action_name == 'hint':
        # generate output
        if hint_for_step_count >= len(data.hint_ids[step]):
            hint_id = 44  # used all hints
        else:
            hint_id = data.hint_ids[step][hint_for_step_count]
    elif action_name == 'encourage':
        hint_id = random.choice(data.encourage_ids)
    elif action_name == 'question':
        if hint_for_step_count >= len(data.probe_ids[step]):
            hint_id = 88  # used all indirect help
        else:
            hint_id = data.probe_ids[step][hint_for_step_count]
    else:
        hint_id = random.choice(data.nothing_ids)

    log_print("\n" + "-" * 20 + " Action Selection Finishes " + "-" * 20, f)

    return hint_id


def update(agent, test, user_count, update_count, f, glove_model, helpwords):
    # fetch score and perform the last observation
    log_print("\n" + "-" * 20 + " Update Starts " + "-" * 20, f)

    user_entry = [user for user in data.users if user['id'] == test['user_id']][0]
    log_print(json.dumps(user_entry), f)
    log_print("\nUser: " + str(test['user_id']) + " " + user_entry['username'], f)
    grade = user_entry["grade"]
    log_print("Grade: " + str(grade), f)
    completed = user_entry["complete_task"]
    log_print("Completed: " + str(completed), f)

    pre_score = test['pre_score']
    post_score = test['post_score']
    log_print("Pre Score: " + str(pre_score), f)
    log_print("Post Score: " + str(post_score), f)
    adjusted_score = test['adjusted_score']
    log_print("Adjusted Score: " + str(adjusted_score), f)

    anxiety = user_entry["anxiety"]
    log_print("Anxiety: " + str(anxiety), f)

    # -------------------------- For loop update --------------------------
    responses_for_user = [
        response for response in data.responses if response["input_message"]["from"] == test['user_id']]

    H = 0  # H as in jonathan and ethan's paper
    psai = 0.3  # psai as in their paper too

    for i, response in enumerate(responses_for_user):
        if i+1 >= len(responses_for_user):
            break

        log_print("\n" + "-" * 20 + " Update " + str(i) + " " + "-" * 20, f)
        #print(response)
        log_print("\nUser: " + str(test['user_id']) + " " + user_entry['username'], f)

        input_text = loads(loads(response['input_message']['payload'])['text'])['blocks'][0]['text']
        log_print('Input: ' + input_text, f)
        pos, neg, hel = vectorize(input_text, f, glove_model, helpwords)

        step = response["input_message"]["step"]
        stage = data.step2stage[step]
        event_entry = [event for event in data.events if
                       (event["user_id"] == test['user_id'] and event["page"] == "/tasks/" + str(stage + 3))]
        log_print("Current Stage: " + str(stage) + " " + step, f)
        log_print("Total Attempts: " + str(len(event_entry)), f)
        #print("Failed Attempts:", len(event_entry))
        ob = normalize(np.array((grade, pre_score, stage, len(event_entry), pos, neg, hel, anxiety), dtype=float))  # (pre_score, last stage)
        log_print('Observation: ' + np.array2string(ob), f)

        next_step = responses_for_user[i+1]["input_message"]["step"]
        next_stage = data.step2stage[next_step]
        next_event_entry = [event for event in data.events if
                       (event["user_id"] == test['user_id'] and event["page"] == "/tasks/" + str(next_stage + 3))]
        next_input_text = loads(loads(responses_for_user[i+1]['input_message']['payload'])['text'])['blocks'][0]['text']
        log_print('Next Input: ' + next_input_text, f)
        next_pos, next_neg, next_hel = vectorize(next_input_text, f, glove_model, helpwords)
        log_print("Next Stage: " + str(next_stage) + " " + next_step, f)
        log_print("Next Total Attempts: " + str(len(next_event_entry)), f)
        next_ob = normalize(np.array((grade, pre_score, next_stage, len(next_event_entry), next_pos, next_neg, next_hel, anxiety), dtype=float))  # (pre_score, new stage)
        log_print('Next Observation: ' + np.array2string(next_ob), f)

        done = False

        step_id = response['hint']['step']['id']

        action_name = data.step2action[step_id]
        action = action2int[action_name]
        #log_print('Action: ' + str(action) + " " + action_name, f)
        log_print('Action: ' + action_name, f)
        # print("Last Action:", step_id, last_action_name)
        log_print("Hint ID: " + str(response['hint']['id']), f)
        log_print('Output: ' + loads(loads(response['output_message']['payload'])['text'])['blocks'][0]['text'], f)
        positive_feedback = response['positive_feedback']
        log_print("Positive Feedback: " + str(positive_feedback), f)

        reward = 0.0
        if positive_feedback is not None:
            reward = 0.1
        if action == 0:  # Jonathan and Ethan's intermediate reward
            reward += 0.01
            H += 1
        log_print("Reward: " + str(reward), f)

        # for testing reward
        # reward = np.random.uniform(0, 1)
        # print('Reward:', reward)

        # comment out for offline policy
        # agent.observe(ob, action, None, reward, next_ob, done)

        # ob = next_ob

    if not responses_for_user:
        log_print("\n" + "-" * 20 + " Response list for " + user_entry['username'] + " is empty " + "-" * 20 + "\n",
                  f)
        return user_count, update_count

    log_print("\n" + "-" * 20 + " Final Update "+ "-" * 20, f)
    log_print("\nUser: " + str(test['user_id']) + " " + user_entry['username'], f)
    last_response = responses_for_user[-1]
    step = last_response["input_message"]["step"]
    stage = data.step2stage[step]
    event_entry = [event for event in data.events if
                   (event["user_id"] == test['user_id'] and event["page"] == "/tasks/" + str(stage + 3))]
    input_text = loads(loads(last_response['input_message']['payload'])['text'])['blocks'][0]['text']
    log_print('Input: ' + input_text, f)
    pos, neg, hel = vectorize(input_text, f, glove_model, helpwords)
    ob = normalize(np.array((grade, pre_score, stage, len(event_entry), pos, neg, hel, anxiety), dtype=float))  # (pre_score, last stage)
    log_print('Observation: ' + np.array2string(ob), f)

    next_ob = normalize(np.array((grade, pre_score, 6, 0, 0, 0, 0, anxiety), dtype=float))  # 6 is finish
    log_print('Next Observation (Finish): ' + np.array2string(next_ob), f)

    step_id = last_response['hint']['step']['id']
    action_name = data.step2action[step_id]
    action = action2int[action_name]
    #log_print('Action: ' + str(action) + " " + action_name, f)
    log_print('Action: ' + action_name, f)
    # print("Last Action:", step_id, last_action_name)
    log_print("Hint ID: " + str(last_response['hint']['id']), f)
    log_print('Output: ' + loads(loads(last_response['output_message']['payload'])['text'])['blocks'][0]['text'], f)
    positive_feedback = last_response['positive_feedback']
    log_print("Positive Feedback: " + str(positive_feedback), f)

    if completed:
        log_print("User completes the task", f)
        reward = adjusted_score + 0.001
        log_print("Abs score: " + str(post_score - pre_score), f)
        log_print("Adj score: " + str(adjusted_score), f)
        if positive_feedback:
            reward += 0.1
        if action == 0:  # Jonathan and Ethan's intermediate reward
            reward += 0.01
            H += 1
        log_print("Reward before subtraction of (1 + psai) * H * 0.01: " + str(reward), f)
        log_print("psai: " + str(psai), f)
        log_print("H: " + str(H), f)
        reward -= (1 + psai) * H * 0.01
    else:
        log_print("User drops out", f)
        reward = -8

    log_print("Reward: " + str(reward), f)

    done = True
    # comment out for offline policy
    #agent.observe(ob, action2int["nothing"], None, reward, next_ob, done)

    log_print("\n" + "-" * 20 + " " +user_entry['username'] + " Observation Finishes " + "-" * 20 + "\n", f)

    user_count += 1
    if user_count % 5 == 0:  # TO BE CHANGED IN REAL STUDY
        # comment out for offline policy
        #agent.update()
        # log_print("\n" + "-" * 20 + " Perform an Update " + "-" * 20 , f)
        update_count += 1

    return user_count, update_count



def vectorize(input_text, f, glove_model, helpwords):
    tokenized_text = word_tokenize(input_text)
    log_print("\tTokenized Input: " + " ".join(tokenized_text), f)
    words = [word.lower() for word in tokenized_text if word.isalpha()]

    sid = SentimentIntensityAnalyzer()
    polarity_score = sid.polarity_scores(input_text)
    log_print('\tPolarity Score: ' + json.dumps(polarity_score), f)
    neg = polarity_score['neg']
    pos = polarity_score['pos']

    hel = 0
    #stop_words = set(stopwords.words("english"))

    exact_help = ["help", "what", "how", "know", "dont", "sos", "hint", "hints", "teach", "why", "understand",
                  "do", "pointers", "stuck", "answer", "answers", "confused", "confusing",
                  "difficult", "hard", "helpful", "?"]

    for token in words:
        if token in exact_help:
            hel = 1
            log_print('\tHelp Score: 1', f)
            return pos, neg, hel

    max_cosine_score = -1
    for token in words:
        if token in glove_model:
            token_vec = glove_model[token]
            for help_vec in helpwords:
                cosine_score = dot(token_vec, help_vec)/(norm(token_vec)*norm(help_vec))
                if cosine_score > max_cosine_score:
                    max_cosine_score = cosine_score
    hel = (max_cosine_score + 1)/4
    log_print('\tHelp Score: ' + str(hel), f)

    return pos, neg, hel


def normalize(ob):
    return (ob - [3, 4, 3, 10, 0.5, 0.5, 0.5, 27])/ [1, 4, 3, 10, 0.5, 0.5, 0.5, 18]


def log_print(info, f):
    print(info)
    f.write(info+"\n")
