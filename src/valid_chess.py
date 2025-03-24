from src.transformer_kv import GPT as GPT_kv
from src.transformer import GPT
import torch
import torch.nn.functional as F
import re
from collections import defaultdict
import interegular

class GPT_valid_chess(GPT_kv):
    def __init__(self, config):
        super().__init__(config)
        self.vocabulary = config.vocabulary
        self.space_index = self.vocabulary.index(' ')

    def get_fsm_from_board(self, board):
        legal_moves = [re.sub(r'\+', r'\\+', board.san(move)+" ") for move in board.legal_moves]
        reg = "|".join(move for move in legal_moves)
        fsm = interegular.parse_pattern(reg).to_fsm()
        return fsm

    def get_transitions_states(self, fsm, alphabet_to_re):
        states_to_vocab = defaultdict(set)
        transitions = defaultdict(dict)
        for state in fsm.map:
            for token in self.vocabulary:
                traversed_states = self.partial_match(state, token, fsm, alphabet_to_re)
                if traversed_states is not None:
                    states_to_vocab[state].add(token)
                    transitions[state][token] = traversed_states[-1]

        return transitions, states_to_vocab

    def partial_match(self, state, token, fsm, alphabet_to_re):
        """Partially match the token to the DFA starting from `state`.

        We iterate over the token's symbols, and at each step transition to the 
        next state if we find a valid transition. 
        If there is a stage without a valid transision, we return None, otherwise
        we return a tuple that contains the sequence of traversed states.

        """
        
        traversed_states = (state,)

        for s in token:
            matched = False
            for alphabet, next_state in fsm.map[state].items():
                pattern = re.escape(alphabet_to_re.get(alphabet, ""))
                try:
                    if re.fullmatch(pattern, s):
                        traversed_states += (next_state,)
                        state = next_state
                        matched = True
                        break
                except re.error as e:
                    print(f"Regex error with pattern '{pattern}': {e}")
            
            if not matched:
                return None
        
        return traversed_states

    @torch.no_grad()
    def generate_valid_chess_move(self, idx, board, max_nex_tokens=10, temperature=1.0, do_sample=False, kv_cache=None, return_kv_cache=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        fsm = self.get_fsm_from_board(board)

        alphabet_to_re = {}
        for k,v in fsm.alphabet._symbol_mapping.items():
            if isinstance(k, interegular.fsm._AnythingElseCls):
                k = "*"
            alphabet_to_re[v] = k

        transitions, states_to_vocab = self.get_transitions_states(fsm, alphabet_to_re)
        state = fsm.initial

        # create an empty kv cache
        if kv_cache is None:
            kv_cache = [None]*self.n_layer

        gen = True
        while gen and max_nex_tokens>0:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _, kv_cache = self(idx_cond, kv_cache=kv_cache)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            valid_tokens = states_to_vocab[state]
            mask = torch.zeros_like(logits)
            for token in self.vocabulary:
                if token not in valid_tokens:
                    mask[:,self.vocabulary.index(token)] = -torch.inf

            masked_logits = (logits + mask)

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(masked_logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue

            new_token = self.vocabulary[idx_next.item()]
            state = transitions[state][new_token]

            if idx_next.item() == self.space_index:
                gen = False

            idx = torch.cat((idx, idx_next), dim=1)

        if return_kv_cache:
            return idx, kv_cache
        else:
            return idx

class GPT_valid_chess_no_kv_cache(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.vocabulary = config.vocabulary
        self.space_index = self.vocabulary.index(' ')

    def get_fsm_from_board(self, board):
        legal_moves = [re.sub(r'\+', r'\\+', board.san(move)+" ") for move in board.legal_moves]
        reg = "|".join(move for move in legal_moves)
        fsm = interegular.parse_pattern(reg).to_fsm()
        return fsm

    def get_transitions_states(self, fsm, alphabet_to_re):
        states_to_vocab = defaultdict(set)
        transitions = defaultdict(dict)
        for state in fsm.map:
            for token in self.vocabulary:
                traversed_states = self.partial_match(state, token, fsm, alphabet_to_re)
                if traversed_states is not None:
                    states_to_vocab[state].add(token)
                    transitions[state][token] = traversed_states[-1]

        return transitions, states_to_vocab

    def partial_match(self, state, token, fsm, alphabet_to_re):
        """Partially match the token to the DFA starting from `state`.

        We iterate over the token's symbols, and at each step transition to the 
        next state if we find a valid transition. 
        If there is a stage without a valid transision, we return None, otherwise
        we return a tuple that contains the sequence of traversed states.

        """
        
        traversed_states = (state,)

        for s in token:
            matched = False
            for alphabet, next_state in fsm.map[state].items():
                pattern = re.escape(alphabet_to_re.get(alphabet, ""))
                try:
                    if re.fullmatch(pattern, s):
                        traversed_states += (next_state,)
                        state = next_state
                        matched = True
                        break
                except re.error as e:
                    print(f"Regex error with pattern '{pattern}': {e}")
            
            if not matched:
                return None
        
        return traversed_states

    @torch.no_grad()
    def generate_valid_chess_move(self, idx, board, max_nex_tokens=10, temperature=1.0, do_sample=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        fsm = self.get_fsm_from_board(board)

        alphabet_to_re = {}
        for k,v in fsm.alphabet._symbol_mapping.items():
            if isinstance(k, interegular.fsm._AnythingElseCls):
                k = "*"
            alphabet_to_re[v] = k

        transitions, states_to_vocab = self.get_transitions_states(fsm, alphabet_to_re)
        state = fsm.initial

        gen = True
        while gen and max_nex_tokens>0:
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            valid_tokens = states_to_vocab[state]
            mask = torch.zeros_like(logits)
            for token in self.vocabulary:
                if token not in valid_tokens:
                    mask[:,self.vocabulary.index(token)] = -torch.inf

            masked_logits = (logits + mask)

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(masked_logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue

            new_token = self.vocabulary[idx_next.item()]
            state = transitions[state][new_token]

            if idx_next.item() == self.space_index:
                gen = False

            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx