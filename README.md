# cturk-ocr

## Motivation
The inspiration for this project came about when I was going through the
*Practical Deep Learning for Coders (2022)* lecture videos by Jeremy Howard.
I took a look at Lesson 0 and one of the things I remember Jeremy said was
that, to learn and apply what I watched in the lecture series, I should
do some projects and write things down. So, I went out and build a
project. This repo is what I call "homework" for Lesson 4 on Hugging
Face Transformers. But what project should I do?

I have a website and application blocking software and one of the
features it has is that to unblock whatever it is blocking, for
example, YouTube or Facebook, I have to type in a list of words.
It is inconvenient at best, so I thought, why not build an OCR
model from HuggingFace Transformers. It's a good learning experience
and in the future, if I want to make a better OCR model, I could.

## Technical details
### Model
I used `microsoft/trocr-base-printed` as my base pre-trained model
(tbh: it's the first model I found and it happens to be popular).
One thing I learned in *PDL4C* is that you are allowed to use
pre-trained models like this one on HuggingFace Models, or `resnet18` 
for computer vision trained with FastAI. From the name, I believe
that this model works best for typed or printed text, which is what
I wanted to train on.

### Training data
I took screenshots of one line of the blocking software's random
word lock screen, seen in `images` and `images_test`, and a CSV
associating the image with the correct text.

### Fine-tuning
I mainly used these resources below to set the boilerplate to train
the model.

This resource inspires how I set the model configuration (see I set the `model.config` values) as well as processing the text as PyTorch sensors with -100 near the end of the tensor:
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb

This guide inspires how I set the training parameters (see `Seq2SeqTrainer` part of the guide as well as the notebook) https://github.com/philschmid/document-ai-transformers/blob/main/training/donut_sroie.ipynb.

### Metrics
I used *character error rate* to measure how well the model correctly
predicts the words during training.

### Difficulties
Because training a HuggingFace Transformers model prefers GPUs,
and I do not have a GPU computer, I ended up creating a Paperspace
Gradient account (recommended by Jeremy in *PDL4C*) and rented
some GPU time for around $2-3 (is it US or Canadian, I don't know but
it's not too bad cost wise).

I also had some difficulties initially training the model, because
when training, the character error rate just stays high or gets worse.
I realized that that is because I did not set a learning rate nor
any weight decay in `TrainingArguments`. I set the learning rate
at `2e-5` and weight decay at `0.01` per recommended by the DocumentAI
Github documentation, and it works like a charm. The best final
validation character error rate is around 0.05.

### Testing
Typically, I should have a test set separate from the training and
validation data, and I do (see `images_test`), and I should actually
go through how well it handles the test set. However, I ended up
informally testing how well the model does by throwing an example
and visually spot check how well it does. Informally, I'd say it does
pretty well.

## Next steps
Some future next steps I can reasonably take:

- Create a Gradio app to let others demo
- Formally create a dataset for the testing data and evaluate how well it does
- Add more training data to increase accuracy
- Continue with the *Practical Deep Learning for Coders* videos (duh)