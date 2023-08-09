# Smart Primer Bot

Generate hints for Smart Primer.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Develop

```bash
FLASK_APP=app.py FLASK_ENV=development FLASK_DEBUG=1 flask run
```

or in VSCode press F5 (debug) and select configuration `Python: Flask`.

When you save a related file, flask will auto reload the server to show the latest result.

Sherry: note that to run locally, have to turn off the debugging mode because tensorflow.keras conflicts with it.

```bash
FLASK_APP=app.py FLASK_ENV=development FLASK_DEBUG=0 flask run
```

## Test

When the server is ready, you can test endpoints through **<http://127.0.0.1:5000/>**.

- <http://127.0.0.1:5000/generate>

  Called by Hasura backend when a message is sent from users. It uses current incoming message and history actions to generate a response and save it into the database.

- <http://127.0.0.1:5000/update>

  Called by Hasura backend when a post-quiz score (or anything that may update the model) is saved, and the model shall be updated.

`/generate` accepts POST and the payload is like this:

```json
{
  "event": {
    "session_variables": {
      "x-hasura-role": "user",
      "x-hasura-user-id": "9"
    },
    "op": "INSERT",
    "data": {
      "old": null,
      "new": {
        "to": 1,
        "from": 9,
        "payload": "{\"text\":\"{\\\"blocks\\\":[{\\\"key\\\":\\\"c5vhs\\\",\\\"text\\\":\\\"hello\\\",\\\"type\\\":\\\"unstyled\\\",\\\"depth\\\":0,\\\"inlineStyleRanges\\\":[],\\\"entityRanges\\\":[],\\\"data\\\":{}}],\\\"entityMap\\\":{}}\"}",
        "updated_at": "2020-03-24T14:56:36.703655+00:00",
        "created_at": "2020-03-24T14:56:36.703655+00:00",
        "id": "5f2063ef-760f-4fc7-aa86-64d960148631",
        "tutorial": false,
        "step": "height"
      }
    }
  },
  "created_at": "2020-03-24T14:56:36.703655Z",
  "id": "cad0942b-ed7c-4edc-8891-adf07625fd17",
  "delivery_info": {
    "max_retries": 0,
    "current_retry": 0
  },
  "trigger": {
    "name": "respond_message"
  },
  "table": {
    "schema": "public",
    "name": "message"
  }
}
```

> Note:
>
> Using this payload directly can cause the bot to generate a response but the last step to save the response into live database will fail because a needed response has already existed.
