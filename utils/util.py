OPENAI_COMPLETION_MODELS = ["text-davinci-003", "text-davinci-002"]
OPENAI_CHAT_MODELS = ["gpt-3.5-turbo"]


def get_token_unit_price(model):
    if model in OPENAI_COMPLETION_MODELS:
        return 0.00002
    elif model in OPENAI_CHAT_MODELS:
        return 0.000002
    else:
        raise ValueError("Model not found")
