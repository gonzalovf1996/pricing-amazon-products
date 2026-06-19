import gradio as gr

from .utils import predict_price


with gr.Blocks(title="Amazon Product Price Estimator") as demo:
    gr.Markdown("# Amazon Product Price Estimator")
    gr.Markdown("Describe your product, discover its price")

    catalog_input = gr.Textbox(
        label="Product description",
        lines=8
    )

    predict_button = gr.Button("Predict price")

    price_output = gr.Number(
        label="Predicted price (USD)",
        precision=2
    )

    predict_button.click(
        fn=predict_price,
        inputs=catalog_input,
        outputs=price_output
    )