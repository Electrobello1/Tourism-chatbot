import tkinter as tk
from tkinter import ttk
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load the chatbot model and data
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

bot_name = "John"

# Create the main application
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("chatbot")
        self.root.geometry("600x700")
        self.root.resizable(False, False)
        self.root.configure(bg="#212F3C")

        # Chat display frame
        self.chat_frame = tk.Frame(self.root, bg="#212F3C")
        self.chat_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        # Allow the window to be resizable
        self.root.resizable(True, True)

        # Chat display
        self.chat_display = tk.Text(
            self.chat_frame,
            font=("Arial", 12),
            bg="#F0F0F0",
            fg="#000000",
            wrap=tk.WORD,
            state="disabled",
            height=20,
        )
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Input frame
        self.input_frame = tk.Frame(self.root, bg="#212F3C")
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Input field
        self.input_field = ttk.Entry(self.input_frame, font=("Arial", 14), width=50)
        self.input_field.pack(side=tk.LEFT, padx=(0, 10), pady=5, fill=tk.X, expand=True)
        self.input_field.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message,
            style="Custom.TButton",
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure("Custom.TButton", font=("Arial", 12), background="#5DADE2")

        # Initialize chat
        self.display_message(bot_name, "Hi! How can I assist you today?")

    def display_message(self, sender, message):
        """Displays messages in the chat window."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.yview(tk.END)
        self.chat_display.configure(state="disabled")

    def send_message(self, event=None):
        """Processes user input and displays chatbot responses."""
        user_message = self.input_field.get().strip()
        if not user_message:
            return

        self.display_message("You", user_message)
        self.input_field.delete(0, tk.END)

        # Generate chatbot response
        bot_response = self.generate_response(user_message)
        self.display_message(bot_name, bot_response)

    def generate_response(self, sentence):
        """Chatbot logic for generating responses."""
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])
        return "I do not understand..."

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
