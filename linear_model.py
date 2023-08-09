from json import loads
import data


def generate_response(input_message):
    """
    @param input_message: Message

    @return hint_id, reply: Int, String
    """
    # text = data.extract_text_from_payload(input_message['payload'])

    # get data
    # data.users
    # data.hints
    # data.responses
    print(input_message)

    input_text = loads(loads(input_message["payload"])["text"])["blocks"][0]["text"]
    print("\nInput: ", input_text)

    if input_text == "ok":
        return 48  # ignore

    responses_for_user = [
        response for response in data.responses if response["input_message"]["from"] == input_message["from"]]

    step = input_message["step"]

    if step == None or step == "":
        return -1

    responses_for_step = [
        response for response in responses_for_user if response["input_message"]["step"] == step]
    hint_for_step_count = len(responses_for_step)

    # generate output
    if hint_for_step_count >= len(data.hint_ids[step]):
        hint_id = 44  # used all hints
    else:
        hint_id = data.hint_ids[step][hint_for_step_count]

    return hint_id
