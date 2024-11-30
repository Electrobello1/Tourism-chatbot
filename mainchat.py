import random
import json
from amadeus import Client, ResponseError
import spacy



from textblob import TextBlob

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()





bot_name = "john"

def Amadeus():
    amadeus = Client(
        client_id='Vw9wz3iUaoAytHHKIcK4i5iEfDymDaEk',
        client_secret='xztqtf6cvN2dniBx'
    )
    print("john:i will need the coordinates of where you wish to visit")
    A=input('enter latitude of location you wish to :')
    B=input('enter longitude of location:')


    try:

         response = amadeus.reference_data.locations.points_of_interest.get(latitude=A,longitude=B)
         print(response.data)
    except ResponseError as error:
         raise error
def find_entities(sentences):

    all_entities = [sentences]
    nlp = spacy.load('en_core_web_sm')
    for sentence in sentences:
        doc = nlp(sentence)
        sentence_entities_list = []
        for ent in doc.ents:
            sentence_entities_list.append(ent)
            print(ent)
        all_entities.append(sentence_entities_list)
    if not all_entities:
            raise ValueError("No named entities found in input sentence.")

    return all_entities





def select_mode():
    # get input from the user.
    print("please select you working mode (enter offline or online): ")
    user_mode = input()
    return user_mode

def sentiment_analysis(sentence):

    text = ' '.join(sentence)  # join list of words into a single string
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"


def run_online_chatbot():
    print("Running on online mode")
    print('Hi! How I can help you? (enter exit to finish the conversation)')


    while True:
    # sentence = "do you use credit cards?"
          sentence = input("You: ")
          if sentence == "quit":
             break

          sentence = tokenize(sentence)
          X = bag_of_words(sentence, all_words)
          X = X.reshape(1, X.shape[0])
          X = torch.from_numpy(X).to(device)

          output = model(X)
          _,predicted = torch.max(output, dim=1)

          tag = tags[predicted.item()]

          probs = torch.softmax(output, dim=1)
          prob = probs[0][predicted.item()]
          if prob.item() > 0.75:
             for intent in intents['intents']:
                 if tag == intent["tag"]:

                     if tag == 'vacation':

                         Amadeus()
                     else:

                       print(f"{bot_name}: {random.choice(intent['responses'])}")
                       print(f"Matched intent tag: {intent['tag']}")
                       print(f"Intent ID: {intent['id']}")

                       sentiment = sentiment_analysis(sentence)
                       print(f"Sentiment: {sentiment}")
                       entities=find_entities(sentence)
                       print(f"entities: {entities}")
                       break


          else:

              print(f"{bot_name}: I do not understand...")

def run_offline_chatbot():
              user_text_file = open('text.txt', "r")
              sentence = user_text_file.readline()


              print("You: ", sentence)



              sentence = tokenize(sentence)
              X = bag_of_words(sentence, all_words)
              X = X.reshape(1, X.shape[0])
              X = torch.from_numpy(X).to(device)

              output = model(X)
              _,predicted = torch.max(output, dim=1)

              tag = tags[predicted.item()]

              probs = torch.softmax(output, dim=1)
              prob = probs[0][predicted.item()]
              if prob.item() > 0.75:
                 for intent in intents['intents']:

                     if tag == intent["tag"]:
                         if tag == 'vacation':
                             Amadeus()
                         else:

                              print(f"{bot_name}: {random.choice(intent['responses'])}")
                              print(f"Intent ID: {intent['id']}")
                              sentiment = sentiment_analysis(sentence)
                              print(f"Sentiment: {sentiment}")
                              entities = find_entities(sentence)
                              print(f"entities: {entities}")






              else:
                   print( "I do not understand...")



def main():
    user_mode = select_mode()
    if str.lower(user_mode) == "offline":
        run_offline_chatbot()

    if str.lower(user_mode) == "online":
        run_online_chatbot()


main()


