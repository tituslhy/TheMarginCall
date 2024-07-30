from llama_index.tools.google import GmailToolSpec

def get_gmail_tool():
    return GmailToolSpec().to_tool_list()