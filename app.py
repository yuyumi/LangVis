# The main app of this flask todo app
# The code are adapted from the source code on https://www.python-engineer.com/posts/flask-todo-app/ by Python Engineer

# Import flask and SQLAlchemy library
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from lang_viz_interp_generator import main as generate_tokens
import os

import argparse

import random

initialized=False

args = argparse.Namespace()
args.lang_model = "gpt2"
args.random_state = 0
args.saliency = True
args.device = "cuda"
args.saliency_metric = "mean"
args.attn_layer_sel = "attn_layer_11"
args.agg_method = "mean"
args.num_tokens = 10
args.num_tokens_buffed = 5
args.attn_pairs = True
args.return_output = True
args.do_sample = False

# Create the flask with static folder name static in the same level of directory with the app.py
app = Flask(__name__, static_folder='./static')


# Configure the SQL database for the app
# Specify the location of the SQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
# Don't track modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Conifgure the SQL database with above specifications
db = SQLAlchemy(app)

keyword_pairs = None
current_output_full_text = "Write your prompt here!"


# Model class for the Todo table
class Tokens(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(100))
    saliency = db.Column(db.String(100))
    intext_index = db.Column(db.Integer)
    token_len = db.Column(db.Integer)
    keyword_ind = db.Column(db.Integer)


# API route for generating a new token
@app.route("/generate", methods=["POST"])
def generate():
    global keyword_pairs, current_output_full_text
    Tokens.query.delete()
    # Get the description of todo task from the user submitted form
    # and use it as the todo item's title in the database
    num_tokens = request.form.get("numTokens")
    if len(num_tokens) > 0:
        try:
            num_tokens = int(num_tokens)
        except:
            num_tokens = 10
    else:
        num_tokens = 10
    input_text = request.form.get("textInput")

    layer_sel = request.form.get("attentionLayer")
    if request.form.get("random_state") == "deterministic":
        print(args.random_state)
        args.random_state = 0
        args.do_sample = False
    elif request.form.get("random_state") == "random":
        args.random_state = random.randint(0, 1e8)
        args.do_sample = True

    if layer_sel == "":
        layer_sel = "12"
    layer_sel = int(layer_sel) - 1
    args.attn_layer_sel = f"attn_layer_{layer_sel:d}"
    
    args.prompt = input_text.rstrip(" ")
    print(args.prompt)

    if len(args.prompt) <= 0:
        return redirect(url_for("home"))

    args.num_tokens = int(num_tokens)

    output = generate_tokens(args)
    keyword_pairs = output["key_attn_pairs_ind"]
    keyword_ind_map = {}
    for keyword_pair in keyword_pairs:
        keyword_ind_map[keyword_pair[-1]] = keyword_pair[-2]


    for i in range(len(output["input_tokens"])):
        token_text = output["input_tokens"][i]
        index = i 
        saliency = ""
        token_len = len(token_text)

        if index in keyword_ind_map.keys():
            keyword_ind = keyword_ind_map[index]
        else:
            keyword_ind = -1 

        new_token = Tokens(token=token_text, saliency=saliency, 
                           intext_index=index, token_len=token_len,
                           keyword_ind=keyword_ind)

        db.session.add(new_token)
        # Commit the change
        db.session.commit()
    cur_index = i

    for i in range(len(output["tokens"])):
        token_text = output["tokens"][i]
        saliency = "".join([f"{score:.2f}," for score in output["saliency"][i]])
        index = cur_index + i
        token_len = len(token_text)

        if index in keyword_ind_map.keys():
            keyword_ind = keyword_ind_map[index]
        else:
            keyword_ind = -1 

        new_token = Tokens(token=token_text, saliency=saliency, 
                           intext_index=index, token_len=token_len,
                           keyword_ind=keyword_ind)

        db.session.add(new_token)
        # Commit the change
        db.session.commit()
    
    current_output_full_text = output["output_full_text"]

    # Redirect to the home page with updated todo list
    return redirect(url_for("home"))


# API route for directing to the main page
@app.route("/")
def home():
    global initialized
    # Get all stored todo items
    if not initialized:
        Tokens.query.delete()
        initialized = True

    token_list = Tokens.query.all()
    if len(token_list) > 0:
        for i in range(len(token_list)):
            if len(token_list[i].saliency) > 0:
                saliency_scores = token_list[i].saliency[:-1].split(",")
                saliency_scores = [float(score) for score in saliency_scores]
                min_score = min(saliency_scores)
                saliency_scores = [score - min_score for score in saliency_scores]
                max_score = max(saliency_scores) + 1e-6
                saliency_scores = [score / max_score for score in saliency_scores]
                scale = 12
                base = 8
                saliency_scores = [int(score * scale + base) for score in saliency_scores]

                token_list[i].saliency = ("".join([f"{score:d}," for score in saliency_scores]))[:-1]
                # token_list[i].saliency = "12"
                # token_list[i].token_len = len(token_list[i].token)
    # Display them in the main page

    num_tokens = args.num_tokens
    sel_attn_ind = int(args.attn_layer_sel[args.attn_layer_sel.rfind("_")+1:]) + 1
    random = "checked" if args.do_sample else ""
    deterministic = "checked" if not args.do_sample else ""

    return render_template("base.html", token_list=token_list, current_output_full_text=current_output_full_text,
                            num_tokens=num_tokens, sel_attn_ind=sel_attn_ind, deterministic=deterministic, random=random)


# When run this file call the functions below
if __name__ == "__main__":
    # Within the context of this application
    with app.app_context():
        # Create the SQL database
        db.create_all()
    # Find the port defined in the environment (if not exist use default port 5000)
    port = int(os.environ.get('PORT', 5000))
    # Run the app on the local host with assigned port
    app.run(host='0.0.0.0', port=port)
