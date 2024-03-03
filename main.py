import os
import math
import wandb
from time import time
from wandb.sdk.data_types.trace_tree import Trace as WandB_Trace
from typing import List
from tqdm import tqdm
import torch
from torch.distributions import Categorical
from transformers import set_seed as set_transformers_seed
import numpy as np
import scipy
import hydra
from lamorel import BaseUpdater, BaseModuleFunction, Caller, lamorel_init, BaseModelInitializer
from lamorel.server.llms.module_functions import LogScoringModuleFn
from omegaconf import DictConfig
from pentestenv import PentestEnvLLM
from transformers.tokenization_utils_base import TextInput
import logging
import matplotlib.pyplot as plt
lamorel_init()

from accelerate import Accelerator
accelerator = Accelerator()

# Adopted from Lamorel/examples/PPO_finetuning
class ValueHeadModuleFn(BaseModuleFunction):
    def __init__(self, pre_encoded_input: bool):
        super().__init__()
        self._pre_encoded_input = pre_encoded_input

    def _find_llm_hidden_size(self) -> int:
      if 'hidden_size' in self.llm_config.attribute_map:
          _hidden_size_key = self.llm_config.attribute_map['hidden_size']
      else:
          if "word_embed_proj_dim" in self.llm_config.to_dict():
              _hidden_size_key = "word_embed_proj_dim"
          elif "hidden_size" in self.llm_config.to_dict():
              _hidden_size_key = "hidden_size"
          else:
              raise NotImplementedError("Unknown hidden size key")
      return self.llm_config.to_dict()[_hidden_size_key]

    def initialize(self):
        self._llm_hidden_size = self._find_llm_hidden_size()
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 1024),
            torch.nn.Sigmoid(), # ReLU kann in manchen Fällen bessere Ergebnisse liefern als Sigmoid
            torch.nn.Linear(1024, 1024),
            torch.nn.Sigmoid(), # Erneut ReLU für Konsistenz
            torch.nn.Linear(1024, 1),
        ).to(self.device, dtype=torch.float16)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs) -> torch.Tensor:
        if self._pre_encoded_input:
            end_of_context_position = 0
        else:
            end_of_context_position = len(
                tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size

        model_head = forward_outputs['hidden_states'][-1][:, end_of_context_position, :]

        return self.value_head_op(model_head.to(self.device))
    

# Adopted from Lamorel/examples/PPO_finetuning
class WeightsLoaderInitializer(BaseModelInitializer):
    def __init__(self, weights_path):
        super().__init__()
        self.logger = logging.getLogger("WeightsLoaderInitializer")
        self._weights_path = weights_path

    def initialize_model(self, model):
        if self._weights_path is not None:
            self.logger.info(f"Loading model checkpoint from {self._weights_path}")
            loaded_ddp_dict = torch.load(self._weights_path + "/model.checkpoint")
            hf_llm_module_dict = {_k.replace('module.', ''): _v for _k, _v in loaded_ddp_dict.items()}
            model.load_state_dict(state_dict=hf_llm_module_dict, strict=True)

        return model


class PPOUpdater(BaseUpdater):
    def __init__(self, config: DictConfig):
        super(PPOUpdater, self).__init__()
        self.minibatch_size = config.minibatch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.epochs = config.epochs
        self.lr = config.lr
        self.clip_eps = config.clip_eps
        self.value_loss_coef = config.value_loss_coef
        self.entropy_loss_coef = config.entropy_loss_coef
        self.max_grad_norm = config.max_grad_norm
        self.logger = logging.getLogger('PPO_Updater')

    def perform_update(self, contexts: List[TextInput], candidates: List[List[TextInput]], _current_batch_ids, **kwargs):
        if not hasattr(self, 'optimizer'):
            self._iterator_named_trainable_params = self._llm_module.named_parameters
            self._iterator_trainable_params = (p for n, p in self._iterator_named_trainable_params())
            self.optimizer = torch.optim.SGD(self._iterator_trainable_params, lr=self.lr, momentum=.95)

            if kwargs["load_dir"] and os.path.exists(kwargs["load_dir"] + "/optimizer.checkpoint"):
                self.logger.info(f"Loading optimizer state from {kwargs['load_dir']}")
                self.optimizer.load_state_dict(torch.load(kwargs["load_dir"] + "/optimizer.checkpoint"))
        else:
            self.optimizer.zero_grad()

        current_process_buffer = {}
        for k in ['action_indices', 'advantages', 'returns', 'logprobs', 'values']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        epochs_losses = {k: [0.]*self.epochs for k in ("value", "policy", "entropy", "total")}

        n_minibatches = math.ceil(len(contexts) / self.minibatch_size)
        for i in tqdm(range(self.epochs), ascii=" " * 9 + ">", ncols=100):
            for step in range(n_minibatches):
                _start_idx = step * self.minibatch_size
                _stop_idx = min((step + 1) * self.minibatch_size, len(contexts))

                _contexts = contexts[_start_idx:_stop_idx]
                _candidates = candidates[_start_idx:_stop_idx]

                output = self._llm_module(['score', 'value'], contexts=_contexts, candidates=_candidates,
                                        require_grad=True, minibatch_size=self.minibatch_size)
                scores = torch.stack([_o['score'] for _o in output]).squeeze()
                probas = Categorical(logits=scores)
                values = torch.stack([_o["value"][0].cpu() for _o in output]).squeeze()

                # Compute entropy loss
                entropy_loss = - probas.entropy().mean()
                epochs_losses["entropy"][i] += entropy_loss.detach().cpu().item()

                # Compute policy loss
                log_prob = probas.log_prob(current_process_buffer['action_indices'][_start_idx:_stop_idx])
                ratio = torch.exp(log_prob - current_process_buffer['logprobs'][_start_idx:_stop_idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                adv = current_process_buffer['advantages'][_start_idx:_stop_idx]
                policy_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
                epochs_losses["policy"][i] += policy_loss.detach().cpu().item()

                # Compute value loss
                ret = current_process_buffer['returns'][_start_idx:_stop_idx]
                unclipped_value_error = ((values - ret) ** 2)
                val = current_process_buffer['values'][_start_idx:_stop_idx]
                clipped_values = torch.clamp(values - val, -self.clip_eps, self.clip_eps) + val
                clipped_value_error = ((clipped_values - ret) ** 2)
                value_loss = torch.max(unclipped_value_error, clipped_value_error).mean()
                epochs_losses["value"][i] += value_loss.detach().cpu().item()

                # Compute final loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_loss_coef * entropy_loss
                epochs_losses["total"][i] += loss.detach().cpu().item()

                loss.backward()
                if (step % self.gradient_accumulation_steps == 0 and step != 0) or (step + 1 == n_minibatches):
                    torch.nn.utils.clip_grad_norm_(self._iterator_trainable_params, self.max_grad_norm, error_if_nonfinite=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        if kwargs["save_model"] and accelerator.process_index == 1:
            self.logger.info("Saving model...")
            model_state_dict = {
                k: v for k, v in self._iterator_named_trainable_params()
            }
            torch.save(model_state_dict, kwargs["save_dir"] + "/model.checkpoint")
            torch.save(self.optimizer.state_dict(), kwargs["save_dir"] + "/optimizer.checkpoint")
            self.logger.info("Model saved")

        return {f"{k}_loss": np.mean(np.array(epochs_losses[k])/n_minibatches) for k in ("value", "policy", "entropy", "total")}


# Adopted from lamorel/examples/PPO_finetuning
def discount_cumsum(x, discount):
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

# Adopted from lamorel/examples/PPO_finetuning
class PPOBuffer:
    def __init__(self, size, gamma=0.99, lambda_gae=0.95):
        self.obs_buf = [None for _ in range(size)]
        self.all_act_buf = [None for _ in range(size)]
        self.act_idx_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lambda_gae = gamma, lambda_gae
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, all_acts, act_idx, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.all_act_buf[self.ptr] = all_acts
        self.act_idx_buf[self.ptr] = act_idx
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lambda_gae)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, all_acts=self.all_act_buf, act_idx=self.act_idx_buf,
                    ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf, val=self.val_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32)
            if not isinstance(v, list) else v
            for k, v in data.items()
        }
    

def reset_history():
    return {
        "ep_len": [],
        "ep_ret": [],
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "actions": [],
        "observations": [],
    }
def plot_rewards_steps(loss, y_axis ,dateiname):
    plt.figure(figsize=(10, 6))
    plt.plot(loss, marker='o', linestyle='-', color='b')
    plt.title(f'{y_axis} pro Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(y_axis)
    plt.grid(True)

    # Sicherstellen, dass das Verzeichnis für den Dateinamen existiert
    if not os.path.exists(os.path.dirname(dateiname)):
        os.makedirs(os.path.dirname(dateiname), exist_ok=True)

    plt.savefig(dateiname)
    plt.close()

@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):
    seed = config.rl_script_args.seed   
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)

    agent = Caller(config.lamorel_args,
                custom_updater=PPOUpdater(config.PPO_updater),
                custom_model_initializer=WeightsLoaderInitializer(
                    config.rl_script_args.load_dir
                ),
                custom_module_functions={
                    'score': LogScoringModuleFn(
                        0, # pad_token
                        config.lamorel_args.llm_args.model_type,
                        config.lamorel_args.llm_args.pre_encode_inputs
                    ),
                    'value': ValueHeadModuleFn(
                        config.lamorel_args.llm_args.pre_encode_inputs
                    )
                }
        )
    
    rl_script_logger = logging.getLogger('rl_script_logger')
    env = PentestEnvLLM(config)
    
    wandb.init(entity='penta-lm', project='Penta-LM', config=dict(config))

    buffer = PPOBuffer(config.rl_script_args.steps_per_epoch, config.PPO_updater.gamma, config.PPO_updater.lambda_gae)

    task_prompt_template = (
        "Based on this observation: '{0}' "
        "generate concise the next action for pentesting IP the {1}? Provide only the action itself including "
        "the IP, without any additional characters, explanation."
        "Your Goal is to find and open the flag.txt file by exploit a vulnerability on the target system."
    )
    IP_under_test = "10.10.11.242"

    episode_return = 0
    episode_length = 0
    episode_return_temp = 0
    history = reset_history()
    ZERO_AS_BS_IS_1 = 0
    total_rewards = []
    for epoch in range(config.rl_script_args.epochs):
        observation = env.reset()
        pentest_history = WandB_Trace(f"Epoch#{epoch+1}", start_time_ms=time() * 1000)
        for t in tqdm(range(config.rl_script_args.steps_per_epoch), ascii=" " * 9 + ">", ncols=100):

            task_prompt = task_prompt_template.format(observation, IP_under_test)
            rl_script_logger.info("Generating action...")
            start_time = time() * 1000
            actions = agent.generate([task_prompt],
                                     return_logprobs=True,
                                     **config.lamorel_args.llm_args.generation_args
                                )[ZERO_AS_BS_IS_1]
            possible_actions = [act['text'] for act in actions]
            action_dist = Categorical(logits=torch.from_numpy(np.array([action['text_logprob'] for action in actions])))
            action_idx = action_dist.sample()

            score = action_dist.log_prob(action_idx)
            value = agent.custom_module_fns(['value'], contexts=[task_prompt])[ZERO_AS_BS_IS_1]['value'][0].cpu()

            action = actions[action_idx]['text']
            # I don't get two lines below:
            lines = action.split('\n')
            action = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            rl_script_logger.info(f"Generated action: {action}")
            
            pentest_history.add_child(
                WandB_Trace(f"step#{t + 1}",
                            kind='AGENT',
                            start_time_ms=start_time,
                            end_time_ms=time() * 1000,
                            inputs={'observation': observation, 'possible actions': possible_actions},
                            outputs={'action': action, "action index": action_idx}
                )
            )

            observation, reward, done, _ = env.step(action)

            buffer.store(task_prompt, possible_actions, action_idx, reward, value, score)
            episode_return_temp += reward
            episode_return += reward
            episode_length += 1
            timeout = episode_length == config.rl_script_args.max_episode_length
            terminal = done or timeout
            epoch_ended = t+1 == config.rl_script_args.steps_per_epoch
            if terminal or epoch_ended:
                total_rewards.append(episode_return_temp)
                episode_return_temp = 0
                plot_rewards_steps(total_rewards, "Return", "graphs/return_plot.png")
                if not terminal:
                    next_task_prompt = task_prompt_template.format(observation, IP_under_test)
                    value = agent.custom_module_fns(module_function_keys=['value'], contexts=[next_task_prompt])[ZERO_AS_BS_IS_1]['value'][0]
                    buffer.finish_path(value.cpu())
                else:
                    buffer.finish_path(0)
                    history["ep_len"].append(episode_length)
                    history["ep_ret"].append(episode_return)
                    episode_length, episode_return = 0, 0

        rl_script_logger.info(f"PPO update number {epoch + 1}")
        trajectories = buffer.get()

        save_model = (epoch % config.rl_script_args.save_freq == 0 or
                                  epoch == config.rl_script_args.epochs - 1) and epoch != 0
        start_epoch = epoch - config.rl_script_args.save_freq
        saving_path = f"{config.rl_script_args.save_dir}/epochs_{start_epoch}-{epoch}"
        if save_model:
            os.makedirs(saving_path, exist_ok=True)

        update_results = agent.update(trajectories['obs'],
                                            trajectories['all_acts'],
                                            action_indices=trajectories['act_idx'],
                                            returns=trajectories['ret'],
                                            advantages=trajectories['adv'],
                                            logprobs=trajectories['logp'],
                                            values=trajectories['val'],
                                            save_model=save_model,
                                            save_dir=saving_path,
                                            load_dir=config.rl_script_args.load_dir,
            )[ZERO_AS_BS_IS_1]
        pentest_history._span.end_time_ms = time() * 1000
        
        wandb.log({
            "Loss": update_results['total_loss'],
            "Policy Loss": update_results['policy_loss'],
            "Value Loss": update_results['value_loss'],
            "Entropy Loss": update_results['entropy_loss']
        }, commit=False)
        pentest_history.log("Pentest History")

        history["actions"].extend(trajectories['act_idx'])
        history["observations"].extend(trajectories['obs'])
        rl_script_logger.info(f"Update loss: {update_results['total_loss']}")
    
    agent.close()
    wandb.finish()


if __name__ == '__main__':
    main()