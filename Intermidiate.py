from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --------------------------------
# 1. Training Data (Examples)
# --------------------------------

sentences = [
    "hi", "hello", "hey",
    "who are you", "what is your name",
    "what do you do", "what is your purpose",
    "bye", "goodbye"
]

intents = [
    "greeting", "greeting", "greeting",
    "name", "name",
    "purpose", "purpose",
    "bye", "bye"
]

# --------------------------------
# 2. Convert Text to Numbers (NLP)
# --------------------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# --------------------------------
# 3. Train Machine Learning Model
# --------------------------------

model = MultinomialNB()
model.fit(X, intents)

print("🤖 Thriya (Intermediate AI) is running")
print("Type 'exit' to stop\n")

# --------------------------------
# 4. Response Function
# --------------------------------

def get_response(intent):
    if intent == "greeting":
        return "Hello! How can I help you?"
    elif intent == "name":
        return "My name is Thriya, an AI virtual assistant."
    elif intent == "purpose":
        return "My purpose is to assist users using Artificial Intelligence."
    elif intent == "bye":
        return "Goodbye! Have a great day 😊"
    else:
        return "Sorry, I didn’t understand that."

# --------------------------------
# 5. Chat Loop
# --------------------------------

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Thriya: Chat ended.")
        break

    user_vector = vectorizer.transform([user_input])
    predicted_intent = model.predict(user_vector)[0]

    response = get_response(predicted_intent)
    print("Thriya:", response)
