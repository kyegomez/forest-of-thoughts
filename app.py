import os
from swarms import OpenAIChat, Agent
from fot.main import ForestOfAgents
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify

global forest 
forest = None

app = Flask(__name__)

# Load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global forest
    input_text = request.form['inputText']
    # Here you can add processing logic
    output_text = forest.run(input_text)
    return jsonify({'outputText': output_text})

def initalize_forest(num_agents:int = 5, max_loops:int = 1, max_new_tokens:int = 100):
    # create llm
    llm = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo", 
    )

    summarizer_llm = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-4-turbo-preview"
    )
    
    query_creator_llm = OpenAIChat(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"
    )
    
    forest = ForestOfAgents(
        llm, 
        summarizer_llm,
        query_creator_llm,
        num_agents=num_agents,
        max_loops=max_loops,
        max_new_tokens=max_new_tokens
    )
    return forest


if __name__ == '__main__':
    forest = initalize_forest()
    app.run(debug=True,
            port=5001
            )
