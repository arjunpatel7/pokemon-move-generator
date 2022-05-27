import gradio as gr
from transformers import AutoTokenizer
from transformers import pipeline

model_checkpoint = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

generate = pipeline("text-generation",
                    model="arjunpatel/distilgpt2-finetuned-pokemon-moves",
                    tokenizer=tokenizer)


def filter_text(generated_move):
    # removes any moves that follow after the genrated move
    print(generated_move)
    sentences = generated_move.split(".")
    if len(sentences) > 2:
        ret_set = " ".join(sentences[0:1])
    else:
        ret_set = generated_move
    return ret_set

def create_move(move):
    seed_text = "This move is called "
    generated_move = generate(seed_text + move, num_return_sequences=2,
                              no_repeat_ngram_size=4)[0]["generated_text"]
    return generated_move


# # demo = gr.Interface(fn=greet, inputs = "text", outputs="text")
#
# gr.Interface(fn=create_move,
#              inputs="text", outputs="text").launch()
# # demo.launch()

def filler_move(test_move, temperature):
    return test_move + " with temperature " + str(temperature)

demo = gr.Blocks()

with demo:
    gr.Markdown("What's that Pokemon Move?")
    with gr.Tabs():
        with gr.TabItem("Standard Generation"):
            with gr.Row():
                text_input_baseline = gr.Textbox()
                text_output_baseline = gr.Textbox()
            text_button_baseline = gr.Button("Create my move!")
        with gr.TabItem("Temperature Search"):
            with gr.Row():
                temperature = gr.Slider(minimum = 0.3, maximum = 4, value = 1, step = 0.1,
                                        label = "Temperature")
                text_input_temp = gr.Textbox(label="Move Name")
                text_output_temp = gr.Textbox(label = "Move Description")
            text_button_temp = gr.Button("Create my move!")

    #text_button_baseline.click(filler_move, inputs=[text_input_baseline, 0], outputs=text_output_baseline)
    text_button_temp.click(filler_move, inputs=[text_input_temp, temperature], outputs=text_output_temp)
demo.launch()