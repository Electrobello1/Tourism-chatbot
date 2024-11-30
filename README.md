# Chatbot with a Lovely Tkinter GUI

## Project Description
This is a chatbot application built using Python's `tkinter` library for the GUI and a PyTorch-based machine learning model for natural language processing. The chatbot has a clean, user-friendly interface and is capable of responding to user inputs based on trained intents and responses.

The chatbot leverages:
- **Natural Language Processing (NLP)** to understand user queries.
- **A Feed Forward Neural Network Model** trained on predefined intents (`intents.json`) to generate appropriate responses.

---

## Features
1. **Beautiful User Interface:**
   - A modern design using `ttk` widgets for improved aesthetics.
   - A large chat display for smooth interaction.
   - An input field with a responsive "Send" button.

2. **Interactive Chatbot Logic:**
   - Processes user input and returns intelligent responses.
   - Uses a trained neural network model for intent recognition.

3. **Keyboard Shortcut:**
   - Press "Enter" to send messages without clicking the "Send" button.

---

## Technologies Used
- **Python Libraries:**
  - `tkinter`: For the graphical user interface.
  - `torch`: For running the PyTorch-based neural network model.
  - `json`: To manage intents and responses data.
  - `random`: For generating varied chatbot responses.
- **Machine Learning Model:**
  - A neural network trained with intent-tagged data (`data.pth`).

---

## Prerequisites
Ensure the following libraries are installed:
- `torch`
- `nltk` (for tokenization)

Install the dependencies using:
```bash
pip install torch nltk
```
### 1. Setup Instructions
Clone the repository:
```bash
git clone https://github.com/Electrobello1/Tourism-chatbot.git
```
### 2.Navigate to the project directory:

### 3.Ensure the following files are present:

- **intents.json: Contains the chatbot's intents and responses.**
- **data.pth: The trained model's state dictionary.**
- **model.py: Defines the neural network structure.**
- **nltk_utils.py: Provides helper functions for tokenization and bag-of-words conversion.**
### 4. Run the application


```bash
pip install requirements.txt
python gui.py
```

### How to Use
- **Start the application by running the script.**
- **Enter a query in the input field at the bottom of the window.**
- **Press "Enter" or click the "Send" button to submit your query.**
- **View the chatbot's response in the chat window.**

### Folder Structure
```bash
chatbot/
│
├── gui.py      # Main application script
├── intents.json        # Intent data file
├── data.pth            # Trained model
├── model.py            # Neural network definition
├── nltk_utils.py       # Helper functions for NLP
├── README.md           # Project documentation
```
### Screen shorts
![Chatbot UI](bot.png)


### Future Enhancements
- **Add support for external APIs (e.g., weather, news, or travel).**
- **Introduce voice input/output for a more interactive experience.**
- **Improve response accuracy by retraining the model on additional data.**