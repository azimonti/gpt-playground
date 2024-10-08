# GPT Playground

**GPT Playground** is a project focused on building and experimenting with Generative Pre-trained Transformer (GPT) models from scratch. This repository provides a simple yet powerful framework for training and testing custom GPT models, complete with tokenizer, model training, text generation, and model introspection modules.

Whether you're exploring the intricacies of language models or just curious about how GPT works under the hood, this playground offers a starting point for simulations, model training, and text generation.

## Features

- **Custom Tokenizer**: Convert raw text into tokenized input for the model.
- **Model Training**: Train the model on any input text.
- **Text Generation**: Generate text sequences based on a given prompt.
- **Flexible Testing**: Modify start sequences and parameters to test different outputs.

## Getting Started

To get started with these simulations, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/azimonti/gpt-playground.git
   ```

2. **Navigate to the repository directory**:
   ```bash
   cd gpt-playground
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your input data**: 
   Add an `input.txt` file under the `runs` directory with the text you'd like to tokenize. Then run the tokenizer:
   ```bash
   python tokenizer.py
   ```

5. **Train the model**:
   This will train the model:
   ```bash
   python train_model.py
   ```

6. **Run the model to generate some text**:
   Use the trained model to generate text based on a given start sequence. You can modify the start sequence to test different inputs:
   ```bash
   python run_model.py
   ```

7. (Optional) **Run the model introspection**:
   Use the introspection to check the number of parameters of the model. 
   ```bash
   python introspect_model.py
   ```


7. (Optional) **Plot the loss**:
   Once one or more run is completed, in addition to tensorboard, it is possible to plot the loss (and will be save on a file). Adding `-vl` will plot also the validation loss.
   ```bash
   python plot_loss.py -d [LOGDIR] [-vl] 
   ```

## Example

For a quick test, you can use the following steps:

- Add text to `input.txt` and run the tokenizer.
- Execute the training using these data.
- Experiment with different start sequences in `run_model.py` to generate diverse outputs.

The following plot illustrates the results of a 500-epoch run:

![Loss plot example](screenshots/500_epochs_run.png)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Contact

If you have any questions or want to get in touch regarding the project, please open an issue or contact the repository maintainers directly through GitHub.

Thank you for exploring Generative Pre-trained Transformers with us!
