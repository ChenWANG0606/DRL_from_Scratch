import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self, capacity=None, n_step = 1, gamma = 0.99, with_priority = False, epsilon = 0.01):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=capacity) if capacity else deque()       
        self.n_step_buffer = deque(maxlen=n_step)
        self.with_priority = with_priority
        if with_priority:
            self.probability = deque(maxlen=capacity)
            self.epsilon = epsilon
        else:
            self.probability = None


    def _get_n_step_info(self):
        '''
        计算 n 步后的 reward 和 next_state'''
        last = self.n_step_buffer[-1]
        reward, next_state, mask = last.reward, last.next_state, last.mask

        for transition in list(self.n_step_buffer)[-2::-1]:
            r, n_s, m = transition.reward, transition.next_state, transition.mask
            reward = r + self.gamma * reward * m
            if m == 0:
                next_state, mask = n_s, m
                break

        first = self.n_step_buffer[0]
        state, action = first.state, first.action

        return state, next_state, action, reward, mask

    def _append_transition(self, state, next_state, action, reward, mask):
        """Append a transition to memory and update priority if enabled."""
        self.memory.append(Transition(state, next_state, action, reward, mask))
        if self.with_priority:
            max_probability = max(self.probability) if len(self.probability) > 0 else self.epsilon
            self.probability.append(max_probability)

    def push(self, state, next_state, action, reward, mask):
        '''
        将 transition 存入 n_step_buffer，只有当 n_step_buffer 满了之后才将 transition 存入 memory
        '''
        transition = Transition(state, next_state, action, reward, mask)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return 
        state, next_state, action, reward, mask = self._get_n_step_info()
        self._append_transition(state, next_state, action, reward, mask)
        if mask == 0:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                state, next_state, action, reward, mask = self._get_n_step_info()
                self._append_transition(state, next_state, action, reward, mask)

            self.n_step_buffer.clear()

    def pop(self):
        return self.memory.popleft()

    def sample(self, batch_size=None):
        '''
        均匀采样
        '''
        if batch_size is None:
            transitions = self.memory
        else:
            transitions = random.sample(self.memory, batch_size)
        return Transition(*zip(*transitions))
    
    def priority_sample(self, batch_size, beta):
        '''
        优先采样
        '''
        prob_sum = sum(self.probability)
        p = [p_i / prob_sum for p_i in self.probability]
        indices = random.choices(range(len(self.memory)), k = batch_size, weights = p)
        transitions = [self.memory[i] for i in indices]
        batch = Transition(*zip(*transitions))
        weights = [(len(self.memory) * p[i])**(-beta) for i in indices]
        weights_max = max(weights)
        weights = [w / weights_max for w in weights]
        return batch, indices, weights
    
    def update_priority(self, indices, td_errors, alpha):
        '''
        根据td-errors更新优先级
        '''
        for idx, td in zip(indices, td_errors):
            self.probability[idx] = (abs(td)+self.epsilon)**alpha

    def __len__(self):
        return len(self.memory)