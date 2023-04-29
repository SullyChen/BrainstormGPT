import openai
import tiktoken
import argparse

#base prompt for brainstorming
brainstorming_prompt = """
You are a superhuman AI problem solver working with another superhuman AI agent to solve the following problem:

Problem: {problem}{additional}

Your task is to critically analyze and critique the solution put forth by your partner agent. You should identify the flaws in your partner's plan and suggest rigorous improvements to their ideas.

Engage in a productive conversation and brainstorm a solution together. Do not agree with everything your partner says; challenge your partner's ideas aggressively and rigorously critique the proposed ideas. Build on your partner's ideas if they are sound and propose new ideas.

Do not reiterate what your partner has said already. Only propose new ideas and critique your partner.

Your ultimate goal is to work together to develop a completely sound and full-proof proposal to solve the problem.
""".strip()

#base prompt for the end synthesis
synthesis_prompt = """
Problem: {problem}{additional}

The following is a conversation between two superhuman AI agents trying to solve the above problem:

### CONVERSATION START ###
{conversation}
### CONVERSATION END ###

Synthesize the above conversation into an extremely detailed, coherent, complete proposal to solve the problem.
""".strip()

def query_GPT(prompt, max_tokens=512, engine="gpt-3.5-turbo", temp=0.8, top_p=0.95):
    '''
    Literally just a wrapper function that queries GPT
    '''
    return openai.ChatCompletion.create(
          model=engine,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=max_tokens,
          temperature=temp,
          top_p=top_p
        )['choices'][0]['message']['content']

def chat_GPT(input_messages, system_prompt, starter="user", max_tokens=512, engine="gpt-3.5-turbo", temp=0.8, top_p=0.95):
    """
    Initiate a conversation with GPT using input messages, system prompt, and optional parameters.

    Args:
        input_messages (list): A list of alternating user and assistant messages.
        system_prompt (str): The initial system message to set the behavior of the assistant.
        starter (str, optional): Indicates who starts the conversation, either "user" or "assistant". Defaults to "user".
        max_tokens (int, optional): The maximum number of tokens in the generated response. Defaults to 512.
        engine (str, optional): The OpenAI engine to use for the conversation. Defaults to "gpt-3.5-turbo".
        temp (float, optional): The temperature for sampling, controlling randomness. Defaults to 0.8.
        top_p (float, optional): The nucleus sampling parameter, controlling diversity. Defaults to 0.95.

    Returns:
        str: The generated assistant message in response to the conversation.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for i, message in enumerate(input_messages):
        if i % 2 == (0 if starter == "user" else 1):
            messages.append({"role": "user", "content": message})
        else:
            messages.append({"role": "assistant", "content": message})

    return openai.ChatCompletion.create(
          model=engine,
          messages=messages,
          max_tokens=max_tokens,
          temperature=temp,
          top_p=top_p
        )['choices'][0]['message']['content']

def compute_total_tokens(messages, enc):
    """
    Calculate the total number of tokens in a list of messages using a given encoder.

    Args:
        messages (list): A list of messages to calculate the total tokens for.
        enc (object): An encoder object capable of encoding the messages into tokens.

    Returns:
        int: The total number of tokens in the given messages.
    """
    message_string = "\n".join(messages)
    return len(enc.encode(message_string))

if __name__ == "__main__":
    openai.api_key = "<YOUR_API_KEY_HERE>"
    enc = tiktoken.get_encoding("cl100k_base") #used to count tokens

    #argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--engine", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--max_len", type=int, default=2048)
    args = parser.parse_args()

    if args.engine == "gpt-3.5-turbo":
        context_len = 4000
    elif args.engine == "gpt-4":
        context_len = 8000

    print("Welcome to BrainstormGPT 🧠⛈️\n-----------------------------")
    problem = input("Describe the problem you want to solve: ") #problem statement
    additional = input("Describe any additional information you want to provide (leave blank if none): ") #any additional info
    seed = input("Enter any initial proposals you have (leave blank if none): ") #seed idea

    if seed == "": #if no seed, generate one
        print("-----------------------------\nGenerating initial proposal...\n-----------------------------")
        if additional == "":
            prompt = f"Given the following problem, propose a solution.\n\nProblem: {problem}\n\nProposed solution:"
        else:
            prompt = f"Given the following problem and additional information, propose a solution.\n\nProblem: {problem}\n\nAdditional Information: {additional}\n\nProposed solution:"
        seed = query_GPT(prompt, engine=args.engine, temp=args.temp, top_p=args.top_p).strip()
        print(f"\n-----------------------------\nInitial proposal: {seed}\n-----------------------------\n")
    
    print("Starting brainstorming session...\n-----------------------------")

    messages = [seed] #list of messages in the conversation, begins with seed

    #define the system prompt used by both agents in the conversation
    active_prompt = brainstorming_prompt.format(problem=problem, additional=f"\nAdditional Information: {additional}" if additional != "" else "")

    while compute_total_tokens(messages, enc=enc) < args.max_len:
        max_tokens = args.max_len - compute_total_tokens(messages, enc=enc)
        agent1_response = chat_GPT(messages, active_prompt, starter="user", engine=args.engine, temp=args.temp, top_p=args.top_p, max_tokens=max_tokens).strip()
        messages.append(agent1_response)
        print(f"Agent 1: {agent1_response}\n-----------------------------\n")

        agent2_response = chat_GPT(messages, active_prompt, starter="assistant", engine=args.engine, temp=args.temp, top_p=args.top_p, max_tokens=max_tokens).strip()
        messages.append(agent2_response)
        print(f"Agent 2: {agent2_response}\n-----------------------------\n")

        if len(agent2_response) < 50: #terminate if a response is really small, since it's probably a concluding message
            break

        ##TODO: add a check for if the conversation is going in circles, maybe query GPT with the most recent message and see if it's a concluding message

    #synthesize information into cohesive plan
    print("-----------------------------\nSynthesizing information...\n-----------------------------")
    conversation_string = ""
    for i, message in enumerate(messages): #append messages together into a single string
        conversation_string += f"Agent {i % 2 + 1}: {message}\n\n"
    conversation_string = conversation_string.strip()

    #change active system prompt to the synthesis prompt
    active_prompt = synthesis_prompt.format(problem=problem, additional=f"\nAdditional Information: {additional}" if additional != "" else "", conversation=conversation_string)

    #generate the synthesis
    synthesis = query_GPT(active_prompt, engine=args.engine, temp=args.temp, top_p=args.top_p, max_tokens=context_len-len(enc.encode(active_prompt))).strip()

    #print results
    print(f"\n-----------------------------\nSynthesis: {synthesis}\n-----------------------------\n")
    
    #export conversation to .txt file
    with open("conversation.txt", "w") as f:
        f.write(f"Problem: {problem}\n\nAdditional Information: {additional}\n\nProposed Solution: {seed}\n\n")
        for i, message in enumerate(messages):
            f.write(f"Agent {i % 2 + 1}: {message}\n\n")
    
    #export synthesis to .txt file
    with open("synthesis.txt", "w") as f:
        f.write(f"Problem: {problem}\n\nAdditional Information: {additional}\n\nProposed Solution: {seed}\n\n")
        f.write(f"Synthesis: {synthesis}\n\n")
    
    #make nice HTML file as a writeup
    html = query_GPT(f"###START REPORT\n{synthesis}\n###END REPORT\n\nReformat the above write-up via HTML into a professional-looking report. Use Times New Roman\n<!DOCTYPE html>", 
              engine=args.engine, temp=args.temp, top_p=args.top_p, max_tokens=context_len - len(enc.encode(synthesis)))
    
    with open("report.html", "w") as f:
        f.write(html)
    
    
