import gradio as gr
from transformers import AutoTokenizer
from transformers import pipeline
from utils import format_moves
import pandas as pd
import tensorflow as tf

model_checkpoint = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

generate = pipeline("text-generation",
                    model="arjunpatel/distilgpt2-finetuned-pokemon-moves",
                    tokenizer=tokenizer)
# load in the model
seed_text = "This move is called "

tf.random.set_seed(0)


# need a function to sanitize imputs
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


def create_greedy_search_move(move, history):
    generated_move = format_moves(generate(seed_text + move, do_sample=False))
    return generated_move, update_history(history, move, generated_move,
                                          "greedy", "None")


def create_beam_search_move(move, num_beams, history):
    generated_move = format_moves(generate(seed_text + move, num_beams=num_beams,
                                           num_return_sequences=1,
                                           do_sample=False, early_stopping=True))
    return generated_move, update_history(history, move, generated_move,
                                          "beam", {"num_beams": 2})


def create_sampling_search_move(move, do_sample, temperature, history):
    generated_move = format_moves(generate(seed_text + move, do_sample=do_sample, temperature=float(temperature),
                                           num_return_sequences=1, topk=0))
    return generated_move, update_history(history, move, generated_move,
                                          "temperature", {"do_sample": do_sample,
                                                          "temperature": temperature})


def create_top_search_move(move, topk, topp, history):
    generated_move = format_moves(generate(
        seed_text + move,
        do_sample=True,
        num_return_sequences=1,
        top_k=topk,
        top_p=topp,
        force_word_ids=tokenizer.encode("The user", return_tensors='tf')))
    return generated_move, update_history(history, move, generated_move,
                                          "top", {"top k": topk,
                                                  "top p": topp})


demo = gr.Blocks()

with demo:
    gr.Markdown("<h1><center>What's that Pokemon Move?</center></h1>")
    gr.Markdown(
        """This Gradio demo allows you to generate Pokemon Move descriptions given a name, and learn more about text 
        decoding methods in the process! Each tab aims to explain each generation methodology available for the 
        model. The dataframe below allows you to keep track of each move generated, to compare!""")
    gr.Markdown("<h3> How does text generation work? <h3>")
    gr.Markdown("""Roughly, text generation models accept an input sequence of words (or parts of words, known as tokens. 
                These models then output a corresponding set of words or tokens. Given the input, the model
                estimates the probability of another possible word or token appearing right after the given sequence. In
                other words, the model estimates conditional probabilities and ranks them in order to generate sequences
                . """)
    gr.Markdown("Enter a two to three word Pokemon Move name of your imagination below, with each word capitalized!")
    gr.Markdown("<h3> Move Generation <h3>")
    with gr.Tabs():
        with gr.TabItem("Standard Generation"):
            gr.Markdown(
                """The default parameters for distilgpt2 work well to generate moves. Use this tab to have fun and as 
                a baseline for your experiments.""")
            with gr.Row():
                text_input_baseline = gr.Textbox(label="Move",
                                                 placeholder="Type a two or three word move name here! Try \"Wonder "
                                                             "Shield\"!")
                text_output_baseline = gr.Textbox(label="Move Description",
                                                  placeholder="Leave this blank!")
            text_button_baseline = gr.Button("Create my move!")
        with gr.TabItem("Greedy Search Decoding"):
            gr.Markdown("""
            
            Greedy search is a decoding method that relies on finding words that has the highest estimated 
            probability of following the sequence thus far. 
            
            Therefore, the model \"greedily\" grabs the highest 
            probability word and continues generating the sentence. 
            
            This has the side effect of finding sequences that are reasonable, but avoids sequences that are 
            less probable but way more interesting. 
            Try the other decoding methods to get sentences with more variety!
            """)
            with gr.Row():
                text_input_greedy = gr.Textbox(label="Move")
                text_output_greedy = gr.Textbox(label="Move Description")
            text_button_greedy = gr.Button("Create my move!")
        with gr.TabItem("Beam Search"):
            gr.Markdown("This tab lets you learn about using beam search!")
            gr.Markdown("""Beam search is an improvement on Greedy Search. Instead of directly grabbing the word that 
            maximizes probability, we conduct a search with B number of candidates. We then try to find the next word 
            that would most likely follow each beam, and we grab the top B candidates of that search. This may 
            eliminate one of the original beams we started with, and that's okay! That is how the algorithm decides 
            on an optimal candidate. Eventually, the beam sequence terminate or are eliminated due to being too improbale. 
            
            Increasing the number of beams will increase model generation time, but also result in a more thorough search. 
            Decreasing the number of beams will decrease decoding time, but it may not find an optimal sentence. 
            
            Play around with the num_beams parameter to experiment! """
            )
            with gr.Row():
                num_beams = gr.Slider(minimum=2, maximum=10, value=2, step=1,
                                      label="Number of Beams")
                text_input_beam = gr.Textbox(label="Move")
                text_output_beam = gr.Textbox(label="Move Description")
            text_button_beam = gr.Button("Create my move!")
        with gr.TabItem("Sampling and Temperature Search"):
            gr.Markdown("This tab lets you experiment with adjusting the temperature of the generator")
            gr.Markdown(
                """
                Greedy Search and Beam Search were both good at finding sequences that are likely to follow our input text,
                but when generating cool move descriptions, we want some more variety! 
                
                Instead of choosing the word or token that is most likely to follow a given sequence, we can instead
                ask the model to sample across the probability distribution of likely words. It's kind of like walking
                into the tall grass and finding a Pokemon encounter. There are different encounter rates, which allow
                for the most common mons to appear (looking at you, Zubat), but also account for surprise, like shinys!
                
                We might even want to go further, though. We can rescale the probability distributions directly instead, 
                allowing for rare words to temporarily become more frequently. We do this using the temperature parameter.
                
                Turn the temperature up, and rare tokens become very likely! Cool down, and we approach more sensible output. 
                
                Experiment with turning sampling on and off, and by varying temperature below!.  
                """)
            with gr.Row():
                temperature = gr.Slider(minimum=0.3, maximum=4.0, value=1.0, step=0.1,
                                        label="Temperature")
                text_input_temp = gr.Textbox(label="Move")
            with gr.Row():
                sample_boolean = gr.Checkbox(label="Enable Sampling?")
                text_output_temp = gr.Textbox(label="Move Description")
            text_button_temp = gr.Button("Create my move!")
        with gr.TabItem("Top K and Top P Sampling"):
            gr.Markdown(
                """
                When we want more control over the words we get to sample from, we turn to Top K and Top P decoding methods!
                
                
                The Top K sampling method selects the K most probable words given a sequence, and then samples from that subset, 
                rather than the whole vocabulary. This effectively cuts out low probability words. 
                
                
                Top P also reduces the available vocabulary to sample from, but instead of choosing the number of 
                words or tokens in advance, we sort the vocabulary from most to least likely word, and we 
                grab the smallest set of words that sum to P. This allows for the number of words we look at to 
                change while sampling, instead of being fixed. 
                
                We can even use both methods at the same time! To disable Top K, set it to 0 using the slider. 
                To disable Top P, set it to 1""")

            with gr.Row():
                topk = gr.Slider(minimum=0, maximum=200, value=0, step=5,
                                 label="Top K")

                text_input_top = gr.Textbox(label="Move")
            with gr.Row():
                topp = gr.Slider(minimum=0.10, maximum=1, value=1, step=0.05,
                                 label="Top P")
                text_output_top = gr.Textbox(label="Move Description")
            text_button_top = gr.Button("Create my move!")
    with gr.Box():
        gr.Markdown("<h3> Generation History <h3>")
        # Displays a dataframe with the history of moves generated, with parameters
        history = gr.Dataframe(headers=["Move Name", "Move Description", "Generation Type", "Parameters"])
    with gr.Row():
        gr.Markdown("<h3>How did you make this?<h3>")
        gr.Markdown("""
        I collected the dataset from Serebii (https://www.serebii.net) , a news source and aggregator of Pokemon info.
        
        
        I then added a seed phrase  "This move is called" just before each move in order to assist the model in generation. 
        
        
        I then followed HuggingFace's handy language_modeling.ipynb for fine-tuning distillgpt2 on this tiny dataset, and
        it surprisingly worked! 
        
        
        I learned all about text generation using the book Natural Language Processing with Transformers by  Lewis Turnstall, 
        Leandro von Werra and  Thomas Wolf, as well as this fantastic article (https://huggingface.co/blog/how-to-generate) by
        Patrick von Platen. Thanks to all of these folks for creating these learning materials, and thanks to the 
        Hugging Face team for developing this product! 
        """)
    text_button_baseline.click(create_move, inputs=[text_input_baseline, history],
                               outputs=[text_output_baseline, history])
    text_button_greedy.click(create_greedy_search_move, inputs=[text_input_greedy, history],
                             outputs=[text_output_greedy, history])
    text_button_temp.click(create_sampling_search_move, inputs=[text_input_temp, sample_boolean, temperature, history],
                           outputs=[text_output_temp, history])
    text_button_beam.click(create_beam_search_move, inputs=[text_input_beam, num_beams, history],
                           outputs=[text_output_beam, history])
    text_button_top.click(create_top_search_move, inputs=[text_input_top, topk, topp, history],
                          outputs=[text_output_top, history])

demo.launch(share=True)
