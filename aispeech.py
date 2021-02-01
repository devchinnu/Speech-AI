import discord 
import os
import requests 
import null 
import random 
from replit import db 
from keep_alive import keep_alive 

client = discord.Client() 

nquestion = ["facebook", "company", "users", "but"] 
cbaquestion = ["Cambridge Analytica", "mining", "data", "election", "meddling", "artifical intelligence"] 
intro = ["Hello Zuckerberg", "hello zuckerberg", "Welcome", "Thank you for being here"]

pconnotation = ["great job", "good", "protect", "working"]
nconnotation = ["threat", "hurt", "betraying"] 

introresponse = ["Hello Senator, thank you for allowing me to be here"]
genresponse = ["Senator, our mission here at Facebook is to connect the world and the people around it, and to spread positivity. Facebook has done a lot of good and has even helped millions of small businesses to prosper. There are many stories I would like to share with and the good we're going at Facebook, my team and I are happy to follow up with you after the hearing to answer any of your pressing questions", "Yes, senator", "I can make sure that our team follows up and gets you the information on that.", "We have a whole A.I. ethics team that is working on developing basically the technology. It's not just about philosophical principles; it's also a technological foundation for making sure that this goes in the direction that we want", "Senator, I don't know if there were any conversations at Facebook overall because I wasn't in a lot of them. But ...", "But I — I — in retrospect I think that was a mistake and knowing what we know now, we should have handled a lot of things here differently.", "Well senator, what we do know is that the IRA, the Internet Research Agency, the — the Russian firm ran about $100,000 worth of ads. I can't say that we've identified all of the foreign actors who are involved here. So, I — I — I can't say that that's all of the money but that is what we have identified.", "Facebook is an idealistic and optimistic company. For most of our existence, we focused on all of the good that connecting people can do. And, as Facebook has grown, people everywhere have gotten a powerful new tool for staying connected to the people they love, for making their voices heard and for building communities and businesses.", "We didn't take a broad enough view of our responsibility, and that was a big mistake. And it was my mistake. And I'm sorry. I started Facebook, I run it, and I'm responsible for what happens here", "It's not enough to just connect people. We have to make sure that those connections are positive. It's not enough to just give people a voice. We need to make sure that people aren't using it to harm other people or to spread misinformation. And it's not enough to just give people control over their information. We need to make sure that the developers they share it with protect their information, too", "Across the board, we have a responsibility to not just build tools, but to make sure that they're used for good. It will take some time to work through all the changes we need to make across the company, but I'm committed to getting this right. This includes the basic responsibility of protecting people's information, which we failed to do with Cambridge Analytica"]

cbaresponse = ["We didn't take a good look at Cambridge Analylitca, and that is my fault", "Fuck Cambridge Analytica", "There are three important steps that we're taking here. For Cambridge Analytica, first of all, we need to finish resolving this by doing a full audit of their systems to make sure that they delete all the data that they have and so we can fully understand what happened. There are two sets of steps that we're taking to make sure that this doesn't happen again", "The most important is restricting the amount of accessed information that developers will have going forward. The good news here is that back in 2014, we actually had already made a large change to restrict access on the platform that would have prevented this issue with Cambridge Analytica from happening again today. Clearly we did not do that soon enough", "Well, senator, we believe a bunch of the information that we — that we will be able to audit. I think you raise an important question and if we have issues, then we — if we are not able to do an audit to our satisfaction, we are going to take legal action to enable us to do that. And if — and also, I know that the U.K. and U.S. governments are also involved in working on this as well", "well senator, I truly believe that artifical intelligence will be the future of technology, but it does require a lot of data"]

def get_response(): 
  response = requests.get("https://www.washingtonpost.com/news/the-switch/wp/2018/04/10/transcript-of-mark-zuckerbergs-senate-hearing/")
  null_data = null.loads(response.text)
  response = null_data[0]['q'] + "-" + null_data[0]['a']
  return(response) 



@client.event 
async def on_ready(): 
  print('We have logged in as {0.user}'.format (client)) 

@client.event 
async def on_message(message): 
  if message.author == client.user: 
    return 

  if message.content.startswith('$hello'):
    await message.channel.send('Hello! Standing by... ')

  if message.content.startswith('what it do baby'):
    await message.channel.send('Give me your personal data')

  if message.content.startswith('Can you define Hate Speech?'):
    await message.channel.send('Senator, I think that this is a really hard question. And I think its one of the reasons why we struggle with it. There are certain definitions that — that we — that we have around, you know, calling for violence or ...') 
  msg = message.content 

  if any(word in msg for word in nquestion): 
    await message.channel.send(random.choice(genresponse))

  if any(word in msg for word in cbaquestion):
    await message.channel.send(random.choice(cbaresponse))
  
  if any(word in msg for word in intro):
    await message.channel.send(random.choice(introresponse))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, network):
        self.gamma = gamma
        self.reward_window = []
        self.model = network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
keep_alive() 
client.run(os.getenv('TOKEN')) 