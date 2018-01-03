import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

"""
This is an implementation of Continuous Bag-of-Words based on the NLP tutorial
on Word Embedding from Robert Guthrie. The website of this tutorial is:
http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py 
"""

CONTEXT_SIZE = 3
EMBEDDING_DIM = 10
raw_text = """We are about to study the idea of a computational process
Computational processes are abstract beings that inhabit computers
As they evolve processes manipulate other abstract things called data
The evolution of a process is directed by a pattern of rules
called a program People create programs to direct processes In effect
we conjure the spirits of the computer with our spells. """.split()

# Create the total vocabulary and word-index dictionary
vocab = set(raw_text)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(3, len(raw_text) - 3):
    context = [raw_text[i - 3], raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2], raw_text[i + 3]]
    label = raw_text[i]
    data.append((context, label))


# Create CBOW network class
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embeddings(x).view((1, -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

# Optimization and Training
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


for epoch in range(10):
    for context, label in data:
        input = make_context_vector(context, word_to_ix)
        output = Variable(torch.LongTensor([word_to_ix[label]]))
        model.zero_grad()
        pred = model(input)
        loss = F.nll_loss(pred, output)
        loss.backward()
        optimizer.step()

    print("Loss: {}".format(loss))
