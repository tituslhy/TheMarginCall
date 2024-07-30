import chainlit as cl
from src.autogen.groupchat import get_groupchat
from src.vn_utils import vn

import warnings
warnings.filterwarnings('ignore')

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name = "The Group Chat",
            markdown_description = "The crew that provides investment tips and strategies",
            default = True,
            icon = "https://cdn-icons-png.flaticon.com/512/6387/6387947.png"
        ),
        cl.ChatProfile(
            name = "The Workspace",
            markdown_description = "Conduct your own analysis of stock trends with the help of an LLM!",
            icon = "https://static-00.iconduck.com/assets.00/workspace-icon-2048x2048-480656cg.png",
        )
    ]

@cl.step(language="sql", name="Vanna")
async def gen_query(human_query: str):
    return vn.generate_sql(human_query)

@cl.step(name="Vanna")
async def execute_query(query):
    current_step = cl.context.current_step
    df = vn.run_sql(query)
    current_step.output = df.head().to_markdown(index=False)
    return df

@cl.step(name="Plot", language="python")
async def plot(human_query, sql, df):
    current_step = cl.context.current_step
    plotly_code = vn.generate_plotly_code(question=human_query,
                                          sql=sql,
                                          df=df)
    fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df, dark_mode=False)
    current_step.output=plotly_code
    return fig

@cl.step(name="Vanna")
async def generate_follow_up(human_query, sql, df):
    current_step = cl.context.current_step
    questions = vn.generate_followup_questions(question = human_query, sql = sql, df = df)
    questions = questions[:3]
    current_step.output = ", ".join(questions)
    return questions

@cl.step(type="run", name="Vanna")
async def chain(human_query: str):
    sql_query = await gen_query(human_query)
    df = await execute_query(sql_query)
    fig = await plot(human_query, sql_query, df)
    follow_ups = await generate_follow_up(human_query, sql_query, df)
    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
    await cl.Message(content=human_query,
                     elements=elements,
                     author="Vanna").send()
    actions = [
        cl.Action(name="question",
                  value=question,
                  label=question) for question in follow_ups
    ]
    await cl.Message(
        content = "Here are some follow-up questions you can ask me",
        author="Vanna",
        actions=actions
    ).send()

@cl.action_callback(name="question")
async def action_callback(action):
    message = action.value
    await chain(message)

@cl.on_chat_start
async def on_chat_start():
    
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "The Group Chat":
        user_proxy, manager, groupchat = get_groupchat()
        cl.user_session.set('user_proxy', user_proxy)
        cl.user_session.set('manager', manager)
        cl.user_session.set('group_chat', groupchat)
        
        msg = cl.Message(
            content="""Hello and welcome to The Margin Call! All our agents have joined this group chat.
            How can we help you today?""",
            author = "chat_manager"
        )
    else:
        msg = cl.Message(
            content = """Hello and welcome to The Margin Call's workspace!
            Feel free to ask questions here and we will get you setup with
            the tools you need for stock analysis.
            """,
            author = "chat_manager"
        )
    await msg.send()

@cl.on_message
async def run_conversation(message: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    if chat_profile == "The Group Chat":
        CONTEXT = message.content
        MAX_ITER = 20
        manager = cl.user_session.get('manager')
        user_proxy = cl.user_session.get('user_proxy')
        groupchat = cl.user_session.get('group_chat')

        if len(groupchat.messages) == 0:
            message = f"""Do the task based on the user input: {CONTEXT}."""
            await cl.Message(content=f"""Starting agents on task...""").send()
            await cl.make_async(user_proxy.initiate_chat)(manager, message=message)
        elif len(groupchat.messages) < MAX_ITER:
            await cl.make_async(user_proxy.send)(manager, message=CONTEXT)
        elif len(groupchat.messages) == MAX_ITER:  
            await cl.make_async(user_proxy.send)(manager, message="exit")
    else:
        await chain(message.content)