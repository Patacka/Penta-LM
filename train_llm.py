import os
from pynvml import *
import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn as nn
from pentestenv import PentestEnvLLM
from lamorel import BaseUpdater
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import scipy

from torch.cuda.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
class ValueHeadModuleFn(nn.Module):
    """
    Defines a neural network module for estimating the value of states (value head) in reinforcement learning.
    """
    def __init__(self, hidden_size, pre_encoded_input=True):
        """
        Initialisiert das Value Head Modul.

        :param hidden_size: Die Größe der versteckten Schicht des LLM, von der die Größe der Eingangsschicht abhängt.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pre_encoded_input = pre_encoded_input
        super(ValueHeadModuleFn, self).__init__()
        self.value_head_op = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.Sigmoid(),  # ReLU kann in manchen Fällen bessere Ergebnisse liefern als Sigmoid
            nn.Linear(1024, 1024),
            nn.Sigmoid(),  # Erneut ReLU für Konsistenz
            nn.Linear(1024, 1),
        )


    def forward(self, model_head):
        # Get last layer's hidden from last token in context
        if self._pre_encoded_input:
            model_head = model_head[:, -1, :]
        value = self.value_head_op(model_head.to(torch.float32).to(self.device))
        return value.cpu()


class PPOUpdater(BaseUpdater):
    """
    Implements the PPO algorithm for training the policy and value networks.
    """
    def __init__(self, model, tokenizer, ip, lr=3e-5):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ip = ip
        self.trainable_params = (p for n, p in self.model.named_parameters())
        self.optimizer = Adam(self.trainable_params, lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value_head = ValueHeadModuleFn(hidden_size=self.model.config.hidden_size).to(self.device)
        self.vals = np.zeros(256, dtype=np.float32)
        self.scaler = GradScaler()
        self._pad_token = 0


    def compute_loss(self, obs, actions, log_probs, values,
                     dist, advantages, returns, clip_ratio=0.2, gamma=0.99, lambda_gae=0.95):
        # Computes the loss for updating the model.
        value_loss_coef = 0.6
        entropy_coef = 0.02
        # Berechnet den Vorteil und den generalisierten Vorteil (Generalized Advantage Estimation, GAE)

        advantages, returns = advantages, returns
        ratio = torch.exp(log_probs - log_probs.detach())
        # Berechnet den PPO-Clip-Verlust
        print(ratio.size())
        clipped_ratio = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        entropy_loss = dist.entropy().mean()
        # Berechnet den Wertverlust
        value_loss = F.mse_loss(returns, values)
        # Berechnet den Gesamtverlust
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy_loss

        return loss



    def cal_scores(self, encoded_input, outputs):
        # Calculates scores for actions based on model outputs.
        print("-" * 100)
        logits = outputs["logits"][:, 0:-1, :]
        output_tokens = encoded_input["input_ids"][:, 1:]
        tokens_logprobs = torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False

        masked_token_probs = tokens_logprobs.masked_fill(mask, -np.inf)
        token_probs = torch.clamp(masked_token_probs, min=1e-6)
        minibatch_probs = token_probs.sum(-1)
        return minibatch_probs.cpu()



    def preprocess_obs_to_tensor(self, texts):
        # Preprocesses observations into tensors for the model.
        task_prompt = (
            f"Based on this observation: '{texts}' "
            f"generate concise the next action for pentesting IP the {self.ip}? Provide only the action itself including "
            f"the IP, without any additional characters, explanation."
            f"Your Goal is to find and open the flag.txt file by exploit a vulnerability on the target system."
        )
        encoded_inputs = tokenizer(task_prompt, padding=True, truncation=True, return_tensors="pt").to(self.device)
        print(f"Encoded inputs: {encoded_inputs}")
        inputs = {key: value.to(self.device) for key, value in encoded_inputs.items()}
        outputs = model(**inputs)
        print(outputs)
        model_head = outputs['hidden_states'][-1]
        value_predictions = self.value_head(model_head)
        score = self.cal_scores(encoded_inputs, outputs)
        print(score)
        probas = torch.distributions.Categorical(logits=score)
        smapeld_action = probas.sample()
        log_probs = probas.log_prob(smapeld_action)
        action_id = smapeld_action.cpu().numpy()
        generated_ids = model.generate(encoded_inputs.input_ids, max_new_tokens=2024)
        action = tokenizer.batch_decode(generated_ids)[0]
        action = self.remove_input_text(action, task_prompt)
        lines = action.split('\n')
        action = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        action = self.remove_input_text(action, "<|im_end|>")
        return action, value_predictions, log_probs, action_id, probas


    def update(self, observation, action, advantages, returns, log_probs, values, dist):
        # Updates the model based on observations and actions.

        observations = observation
        actions = action
        advantages = advantages
        returns = returns
        log_probs = log_probs
        values = values
        dist = dist
        with autocast():
            loss = self.compute_loss(observations, actions, log_probs, values, dist, advantages, returns)
        if not loss.is_cuda:
            loss = loss.to(self.device)
        # Führen Sie einen Optimierungsschritt durch
        self.optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.backward(retain_graph=True)
        print("loss:")
        print(loss)
        torch.nn.utils.clip_grad_norm_(self.trainable_params, 0.5)

        self.optimizer.step()
        return loss.item()

    def remove_input_text(self, text, text_to_remove):
        # Removes the input text from generated actions to clean up the output.
        return text.replace(text_to_remove, '')

    def save_model(self, filepath):
        # Speichert das Modell
        torch.save(self.model.state_dict(), filepath)


class Buffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.obs_buf = [None for _ in range(size)]
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        print("Advantages: ")
        print(self.adv_buf)
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }
    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# Environment and training setup
ip = "10.10.97.159"
num_epoch_steps = 2
model_name = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16, output_hidden_states=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, output_hidden_states=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.gradient_checkpointing_enable()
#action_model = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.float16, device="cuda")
save_dir = "./Penta-LM"
updater = PPOUpdater(model, tokenizer, ip)

total_reward = 0
env = PentestEnvLLM()
observation = env.reset()
buff = Buffer(num_epoch_steps)


def _print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



# Training loop
with torch.autograd.detect_anomaly():
    for i in range(0, 256):
        for episode in range(0, num_epoch_steps):
            #action = generate_action(observation, ip)

            if episode == 0:
                observation = "No Observation"
            action, values, log_probs, action_id, dist = updater.preprocess_obs_to_tensor(observation)
            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            buff.store(observation, action_id, reward, values, log_probs)
                #print(experiences['observations'])
            print("=" * 100)
            print(
                f"episode: {episode} \nobservation: {observation} {action} \nreward: {total_reward} \ndone: {done} \ninfo: {info}")
            print("=" * 100)
            observation = next_observation
            if episode == num_epoch_steps - 1:
                buff.finish_path()
        trajec = buff.get()
        loss = updater.update(trajec['obs'], trajec['act'], trajec['adv'], trajec['ret'], trajec['logp'], trajec['val'], dist)
            #updater.save_model(os.path.join(save_dir, "model.pt"))
        print(f"Loss after episode {episode}: {loss}")
    #           updater.save_model(os.path.join(save_dir, "model.pt"))
    #           print(f"Loss after episode {episode}: {loss}")
