def get_chat_template(mapping):
    assistant_value = mapping.get("assistant", "default_assistant_value")
    return (
        "some_template"
        .replace("'assistant'", "'" + assistant_value + "'")
    )
