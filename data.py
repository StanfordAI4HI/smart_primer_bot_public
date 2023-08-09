from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import json
import numpy as np

lock = False

# Read from the step_id column from the hint table on Heroku
step_ids = {
    "hint": [1, 2, 3, 4, 8, 9, 10],  # 10 is Temp (use up)
    "encourage": [15],
    "question": [16],  # probing
    "nothing": [13]
}

step2stage = {"height": 0, "width": 1, "length": 2, "volume": 3, "weight": 4, "comparison": 5, "finish": 6}

# [step]: [hint_id]
hint_ids = {
    "height": [40, 41, 43],
    "width": [37, 19, 20],
    "length": [21, 22, 23],
    "volume": [24, 25, 27, 28, 45, 30],
    "weight": [38, 33, 34],
    "comparison": [39, 36],
    "use up": [44, 89, 90, 91]
}

probe_ids = {
    "height": [88],
    "width": [88],
    "length": [88],
    "volume": [64, 65, 66, 67, 68],
    "weight": [69, 70],
    "comparison": [88],
    "use up": [88]
}

encourage_ids = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87]

nothing_ids = [51]


step2action = {}
for action, steps in step_ids.items():
    for step in steps:
        step2action[step] = action

transport = RequestsHTTPTransport(
    url='https://smart-primer.herokuapp.com/v1/graphql',
    use_json=True,
    headers={
        "Content-type": "application/json",
        "x-hasura-admin-secret": "smart.primer@Stanford"
    },
    verify=True
)

client = Client(
    retries=3,
    transport=transport,
    fetch_schema_from_transport=True,
)

INSERT_RESPONSE = gql('''
mutation InsertResponse($hint_id: Int!, $input_id: uuid!, $payload: String!, $to: Int!) {
    insert_response(objects: {hint_id: $hint_id, input_message_id: $input_id, output_message: {
        data: {from: 1, payload: $payload, to: $to}}}) {
            returning {
            id
            }
    }
}
''')

GET_USER = gql('''
query GetUsers {
  user {
    id
    username
    test {
      id
      pre_score
      post_score
    }
    grade
    complete_task
    anxiety
  }
}
''')

GET_RESPONSE = gql('''
query GetResponses {
  response (where: {output_message: {tutorial: {_eq: false}}}) {
    input_message {
      id
      from
      to
      created_at
      payload
      step
    }
    output_message {
      id
      created_at
      payload
    }
    hint {
      id
      name
      step {
        id
        name
      }
    }
    created_at
    positive_feedback
  }
}
''')

GET_HINT = gql('''
query GetHints {
  hint {
    id
    name
    questions
    answer
    created_at
    step {
      id
      name
    }
  }
}
''')

GET_EVENT = gql('''
query GetEvents {
  event (where: {type: {_eq: "answer"}}) {
    user_id
    page
  }
}
''')


def insert_response(variables):
    return client.execute(INSERT_RESPONSE, variable_values=variables)


users = []
hints = []
responses = []
events = []


def fetch_all_users():
    global users
    users = client.execute(GET_USER)['user']


def fetch_all_hints():
    global hints
    hints = client.execute(GET_HINT)['hint']


def fetch_all_responses():
    global responses
    responses = client.execute(GET_RESPONSE)['response']


def fetch_all_events():
    global events
    events = client.execute(GET_EVENT)['event']


def extract_text_from_payload(payload):
    return json.loads(json.loads(payload)['text'])[
        'blocks'][0]['text']


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")

    help_words = ["help", "what", "how", "know", "dont", "sos", "hint", "hints", "teach", "why", "understand",
                  "do", "pointers", "stuck", "answer", "answers", "confused", "confusing",
                  "difficult", "hard", "helpful"]

    help_word_vecs = []

    for help_word in help_words:
        if help_word in gloveModel:
            help_word_vec = gloveModel[help_word]
            help_word_vecs.append(help_word_vec)
        else:
            print("\t" + help_word + "is not in glove")

    #print(help_word_vecs[10])

    return gloveModel, help_word_vecs
