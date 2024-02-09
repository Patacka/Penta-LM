from numpy import std, mean

class DataCollector:
    def __init__(self, gamma=0.99, lambda_=0.95):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []

    def store(self, observation, action, reward, log_prob, value):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.get_batch_data()

    def process_episode(self):
        # Berechne die zukünftigen Renditen und Vorteile
        next_value = 0
        self.returns = []
        self.advantages = []
        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * next_value - self.values[step]
            gae = delta + self.gamma * self.lambda_ * gae
            next_value = self.values[step]
            self.returns.insert(0, gae + self.values[step])
            self.advantages.insert(0, gae)

    def get_batch_data(self):
        self.process_episode()
        experiences = {
            'observations': self.observations,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'rewards': self.rewards,
            'values': self.values,
        }
        print(experiences['observations'])
        #self.reset()  # Bereite den Collector für die nächste Datensammlung vor
        return experiences
