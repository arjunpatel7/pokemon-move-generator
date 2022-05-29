import gradio as gr
from transformers import AutoTokenizer
from transformers import pipeline
from utils import format_moves
import pandas as pd
model_checkpoint = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

generate = pipeline("text-generation",
                    model="arjunpatel/distilgpt2-finetuned-pokemon-moves",
                    tokenizer=tokenizer)
# load in the model
seed_text = "This move is called "
import tensorflow as tf
tf.random.set_seed(0)

#need a function to sanitize imputs
# - remove extra spaces
# - make sure each word is capitalized
# - format the moves such that it's clearer when each move is listed
# - play with the max length parameter abit, and try to remove sentences that don't end in periods.

def update_history(df, move_name, move_desc, generation, parameters):
    # needs to format each move description with new lines to cut down on width

    new_row = [{"Move Name": move_name,
               "Move Description": move_desc,
               "Generation Type": generation,
               "Parameters": parameters}]
    return pd.concat([df, pd.DataFrame(new_row)])

def create_move(move, history):
    generated_move = format_moves(generate(seed_text + move, num_return_sequences=1))
    return generated_move, update_history(history, move, generated_move,
                                                        "baseline", "None")


def create_greedy_search_move(move):
    generated_move = generate(seed_text + move, do_sample=False)
    return format_moves(generated_move)


def create_beam_search_move(move, num_beams=2):
    generated_move = generate(seed_text + move, num_beams=num_beams,
                              num_return_sequences=1,
                              do_sample=False, early_stopping=True)
    return format_moves(generated_move)


def create_sampling_search_move(move, do_sample=True, temperature=1):
    generated_move = generate(seed_text + move, do_sample=do_sample, temperature= float(temperature),
                              num_return_sequences=1,  topk=0)
    return format_moves(generated_move)


def create_top_search_move(move, topk=0, topp=0.90):
    generated_move = generate(
        seed_text + move,
        do_sample=True,
        num_return_sequences=1,
        top_k=topk,
        top_p=topp,
        force_word_ids=tokenizer.encode("The user", return_tensors='tf'))
    return format_moves(generated_move)




demo = gr.Blocks()

with demo:
    gr.Markdown("<h1><center>What's that Pokemon Move?</center></h1>")
    gr.Markdown("This Gradio demo is a small GPT-2 model fine-tuned on a dataset of Pokemon moves! It'll generate a move description given a name.")
    gr.Markdown("Enter a two to three word Pokemon Move name of your imagination below!")
    with gr.Tabs():
        with gr.TabItem("Standard Generation"):
            with gr.Row():
                text_input_baseline = gr.Textbox(label = "Move",
                                                 placeholder = "Type a two or three word move name here! Try \"Wonder Shield\"!")
                text_output_baseline = gr.Textbox(label = "Move Description",
                                                  placeholder= "Leave this blank!")
            text_button_baseline = gr.Button("Create my move!")
        with gr.TabItem("Greedy Search"):
            gr.Markdown("This tab lets you learn about using greedy search!")
            with gr.Row():
                text_input_greedy = gr.Textbox(label="Move")
                text_output_greedy = gr.Textbox(label="Move Description")
            text_button_greedy = gr.Button("Create my move!")
        with gr.TabItem("Beam Search"):
            gr.Markdown("This tab lets you learn about using beam search!")
            with gr.Row():
                num_beams = gr.Slider(minimum=2, maximum=10, value=2, step=1,
                                      label="Number of Beams")
                text_input_beam = gr.Textbox(label="Move")
                text_output_beam = gr.Textbox(label="Move Description")
            text_button_beam = gr.Button("Create my move!")
        with gr.TabItem("Sampling and Temperature Search"):
            gr.Markdown("This tab lets you experiment with adjusting the temperature of the generator")
            with gr.Row():
                temperature = gr.Slider(minimum=0.3, maximum=4.0, value=1.0, step=0.1,
                                        label="Temperature")
                sample_boolean = gr.Checkbox(label = "Enable Sampling?")
                text_input_temp = gr.Textbox(label="Move")
                text_output_temp = gr.Textbox(label="Move Description")
            text_button_temp = gr.Button("Create my move!")
        with gr.TabItem("Top K and Top P Sampling"):
            gr.Markdown("This tab lets you learn about Top K and Top P Sampling")
            with gr.Row():
                topk = gr.Slider(minimum=10, maximum=100, value=50, step=5,
                                 label="Top K")
                topp = gr.Slider(minimum=0.10, maximum=0.95, value=1, step=0.05,
                                 label="Top P")
                text_input_top = gr.Textbox(label="Move")
                text_output_top = gr.Textbox(label="Move Description")
            text_button_top = gr.Button("Create my move!")
    with gr.Box():
        # Displays a dataframe with the history of moves generated, with parameters
        history = gr.Dataframe(headers= ["Move Name", "Move Description", "Generation Type", "Parameters"])


    text_button_baseline.click(create_move, inputs=[text_input_baseline, history], outputs=[text_output_baseline, history])
    text_button_greedy.click(create_greedy_search_move, inputs=text_input_greedy, outputs=text_output_greedy)
    text_button_temp.click(create_sampling_search_move, inputs=[text_input_temp, sample_boolean, temperature],
                           outputs=text_output_temp)
    text_button_beam.click(create_beam_search_move, inputs=[text_input_beam, num_beams], outputs=text_output_beam)
    text_button_top.click(create_top_search_move, inputs=[text_input_top, topk, topp], outputs=text_output_top)

    #Whenever any of the output boxes updates, take that output box and add it to the History dataframe
    #text_output_baseline.change(update_history, inputs = [history, text_input_baseline, text_output_baseline], outputs = history)
demo.launch(share=True)
