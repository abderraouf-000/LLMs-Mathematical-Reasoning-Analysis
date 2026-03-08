
import os
import time
import torch
import argparse
import jsonlines
import numpy as np
from typing import *
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, DynamicCache, GenerationConfig


# from LightThinker.utils import *
from configv1 import Config
from tokenizerv1 import Tokenizer
from model_llama import LlamaForCausalLM
from model_qwen import Qwen2ForCausalLM



import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence

            

import json

with open("../categories.json", "r") as f:
    data = json.load(f)

categories = data["categories"]
category_map = data["category_map"]








    
class AttentionUtils:

    """
    Attention Management.
    
    - Storing Attention tensors for all Generated tokens.
    - Updating attention Mask.
    - Different Attention visualization.
    
    """
    
    def __init__(self, num_layers:int, num_heads:int):
        self.token_generation_stored_attentions: List[torch.Tensor] = list();

    def format_stored_attention(self):

            flattened = [t.squeeze(2).transpose(0, -1) for t in self.token_generation_stored_attentions]  # [seq_len, 32, 32]
        
            # Pad along seq_len dimension
            padded = pad_sequence(flattened, batch_first=True, padding_value=-1)  # [batch, max_seq_len, 32, 32]
        
            # Restore shape [batch, 32, 32, 1, max_seq_len]
            padded = padded.permute(0, 2, 3, 1).unsqueeze(3)  # [batch, 32, 32, 1, max_seq_len]
        
            return padded
                
    
    def add_new_generation(self, generation: torch.Tensor):
        self.token_generation_stored_attentions.append(generation); ## length is #generations, Each element has (#layers, #heads, 1, #before tokens).






### Done

class KVUtils:
    """
    KV Cache Manager
    """

    def __init__(self):
        self.past_key_values: DynamicCache = DynamicCache()

    def get_cache(self) -> DynamicCache:
        return self.past_key_values

    def set_cache(self, past_key_values:DynamicCache):
        self.past_key_values = past_key_values

    @torch.no_grad()
    def reduce_cache(self, start:int, end:int):
        assert end <= self.past_key_values._seen_tokens
        assert self.past_key_values._seen_tokens == self.past_key_values.key_cache[0].shape[2]

        # 1. reduce the value of seen_tokens
        self.past_key_values._seen_tokens -= (end-start)

        # 2. reduce the key_cache and value_cache
        bsz, n_head, q_length, head_dim = self.past_key_values.key_cache[0].shape;
        
        new_q_length = q_length - (end-start)
        assert self.past_key_values._seen_tokens == new_q_length
        for layer_id in range(len(self.past_key_values.key_cache)):
            if new_q_length > end:
                # overlap
                # start:end
                for i in range(new_q_length-start):
                    self.past_key_values.key_cache[layer_id][:, :, start+i, :] = self.past_key_values.key_cache[layer_id][:, :, end+i, :]
                    self.past_key_values.value_cache[layer_id][:, :, start+i, :] = self.past_key_values.value_cache[layer_id][:, :, end+i, :]
            else:
                self.past_key_values.key_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.key_cache[layer_id][:, :, end:, :]
                self.past_key_values.value_cache[layer_id][:, :, start:new_q_length, :] = self.past_key_values.value_cache[layer_id][:, :, end:, :]

            self.past_key_values.key_cache[layer_id] = \
                self.past_key_values.key_cache[layer_id][:, :, 0:new_q_length, :]
            self.past_key_values.value_cache[layer_id] = \
                self.past_key_values.value_cache[layer_id][:, :, 0:new_q_length, :]

    def __del__(self):
        del self.past_key_values.value_cache
        del self.past_key_values.key_cache
        del self.past_key_values
        torch.cuda.empty_cache()
        time.sleep(1)

















## Done
class TokenUtils:

    """
    Generated Token Manager
    """
    
    def __init__(self, max_length:int, device:str, rolling_rope:bool):
        
        self.rolling_rope:bool = rolling_rope

        # self.input_ids[..., 0:self._seen_tokens]
        self.max_length:int = max_length
        self.input_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        self._seen_tokens:int = 0;


        # turn into tensor 
        self._input_types:List[int] = list();
        self._current_sentence_type:List[int] = list();
        self.seen_sentences:int = 0

        

        # complete token sequence
        self._whole_input_ids:List[int] = list()

        
        # dynamic token sequence (thoughts to be discarded.)
        self._current_input_ids:List[int] = list()

        # compression token will not be included
        self.show_prompt_input_ids:List[int] = list()
        self.show_output_input_ids:List[int] = list()

        self.position_ids:torch.Tensor = torch.arange(max_length, device=device).unsqueeze(dim=0)
        # used when rolling_rope==True
        self.arange_ids:torch.Tensor = torch.arange(max_length, device=device)

        self._whole_position_ids:List[int] = list()
        self._current_position_ids:List[int] = list()

        # peak token
        self.max_token = 0;



    
    
    def get_input_types(self) -> torch.Tensor:
        return torch.tensor(self._input_types);



    def get_num_generated_tokens(self) -> int:

        # print("num show prompt ids" , len(self.show_prompt_input_ids))
        # print("num seen tokens" , self._seen_tokens)
        # print(f"Num of generated_tokens : {self._seen_tokens - len(self.show_prompt_input_ids)}")
        
        return self._seen_tokens - len(self.show_prompt_input_ids) -1;



    def add_input_type(self, input_type: int):
        self._input_types.append(input_type);
    
    def update_input_type(self, idx: int, input_type: int):
        self._input_types[idx] = input_type
    
    def add_current_sentence_type(self, sentence_type: int):
        self._current_sentence_type.append(sentence_type);
        self.seen_sentences += 1;
        
    def set_current_sentence_type(self, idx:int, sentence_type: int):
        self._current_sentence_type[idx] = sentence_type;

    
    def get_current_sentence_type(self, idx: int) -> torch.tensor:    
        # print(idx);
        return torch.tensor(self._current_sentence_type[idx]);

    
    

    def get_input_ids(self) -> torch.Tensor:
        return self.input_ids[..., 0:self._seen_tokens]

    

    def get_input_ids(self, start:int, end:int) -> torch.Tensor:
        if start >= 0 and end >= 0:
            return self.input_ids[..., start:end]
        else:
            if start < 0:
                start = self._seen_tokens + start
            if end < 0:
                end = self._seen_tokens + end
            return self.input_ids[..., start:end]
    
    
    def get_input_ids(self, idx:int) -> torch.Tensor:
        if idx >= 0:
            return self.input_ids[..., idx:idx+1]
        else:
            idx = self._seen_tokens + idx
            return self.input_ids[..., idx:idx+1]

    
    
    
    def get_position_ids(self) -> torch.Tensor:
        return self.position_ids[..., 0:self._seen_tokens]



    
    def set_input_id(self, idx:int):
        
        self.input_ids[..., self._seen_tokens] = idx
        self._whole_input_ids.append(idx)
        self._current_input_ids.append(idx)

        if self.rolling_rope:
            new_pos = len(self._current_position_ids)
        else:
            new_pos = self._whole_position_ids[-1] + 1
    
        self.position_ids[..., self._seen_tokens] = new_pos
        self._current_position_ids.append(new_pos)
        self._whole_position_ids.append(new_pos)

        self._seen_tokens += 1
        self.max_token = max(self.max_token, self._seen_tokens);

    

    def set_input_ids(self, input_ids:List[int], return_tensors:bool=False):
        assert isinstance(input_ids, list)
        _start = self._seen_tokens
        for i in range(len(input_ids)):
            self.input_ids[..., self._seen_tokens + i] = input_ids[i]
            if not self.rolling_rope:
                if len(self._whole_position_ids) == 0:
                    self.position_ids[..., self._seen_tokens + i] = 0
                    self._current_position_ids.append(0)
                    self._whole_position_ids.append(0)
                else:
                    self.position_ids[..., self._seen_tokens + i] = self.position_ids[0,self._seen_tokens + i - 1] + 1
                    self._current_position_ids.append(self._whole_position_ids[-1] + 1)
                    self._whole_position_ids.append(self._whole_position_ids[-1] + 1)
        _end = _start + len(input_ids)
        
        if self.rolling_rope:
            self.position_ids[..., 0:self._seen_tokens+len(input_ids)] = self.arange_ids[0:self._seen_tokens]
            self._current_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            self._whole_position_ids.extend([self._current_position_ids[-1] + i + 1 for i in range(len(input_ids))])
            # assert False
        self._seen_tokens += len(input_ids)
        self._whole_input_ids.extend(input_ids)
        self._current_input_ids.extend(input_ids)
        self.max_token = max(self.max_token, self._seen_tokens)
        if return_tensors:
            return self.input_ids[..., _start:_end], self.position_ids[..., _start:_end]

    
    
    def reduce_input_ids(self, start:int, end:int):
        origin_length = self._seen_tokens
        self._seen_tokens -= (end-start)
        if self._seen_tokens > end:
            for i in range(self._seen_tokens-start):
                self.input_ids[..., start+i] = self.input_ids[..., end+i]
                self._current_input_ids[start+i] = self._current_input_ids[end+i]
        else:
            self.input_ids[..., start:self._seen_tokens] = self.input_ids[..., end:origin_length]
            self._current_input_ids[start:self._seen_tokens] = self._current_input_ids[end:origin_length]
        
        self._current_input_ids = self._current_input_ids[0:self._seen_tokens]

        if not self.rolling_rope:
            if self._seen_tokens > end:
                for i in range(self._seen_tokens-start):
                    self.position_ids[..., start+i] = self.position_ids[..., end+i]
                    self._current_position_ids[start+i] = self._current_position_ids[end+i]
            else:
                self.position_ids[..., start:self._seen_tokens] = self.position_ids[..., end:origin_length]
                self._current_position_ids[start:self._seen_tokens] = self._current_position_ids[end:origin_length]
        else:
            self._current_position_ids[start:self._seen_tokens] = [start+i for i in range(self._seen_tokens - start)]
            self.position_ids[..., 0:self._seen_tokens] = self.arange_ids[0:self._seen_tokens]
        self._current_position_ids = self._current_position_ids[0:self._seen_tokens]


    
    def reset(self):
        self._seen_tokens = 0
        self.max_token = 0
        self._seen_sentences:int = 0
        self._whole_input_ids.clear()
        self._current_input_ids.clear()
        self._whole_position_ids.clear()
        self._current_position_ids.clear()
        self.show_prompt_input_ids.clear()
        self.show_output_input_ids.clear()
        self._current_sentence_type.clear();
        self._seen_sentences.clear();






















class KVsemanticAttentionMaskingUtils:

    """
    Managing generation masking during inference.


    """

    def __init__(
        self, 
        max_length:int, 
        device:str = "cuda", 
        dtype:torch.dtype = torch.float16,
        token_utils:TokenUtils = None,
    ):
        """
        args:
            - max_length: 
                max generated tokens
            - device: 
                "cuda"
            - dtype: 
                fp16 or bf16
        """
        self.dtype = dtype
        self.device:str = device
        self.max_length:int = max_length
        self.amplifier_mask_value = 1.02;
        self.mask_value = 0.0;
        self.show_value = 1.0;
        self.token_utils:TokenUtils = token_utils;
    

    @torch.no_grad()

    def make_generation_mask(self, mask_pattern_type = "causal"):
        # Mask for the new token: shape [1, past_length+1]
        mask = torch.full((1, self.token_utils._seen_tokens), self.show_value, device=self.device);
    
        if mask_pattern_type != "causal" and self.token_utils.seen_sentences > 0:
            # print("Applying non-causal mask", self.token_utils.seen_sentences)

            if mask_pattern_type == "semantic":
                # mask_pattern: list of bools (length = past_length + 1)

                # # print("Num generated_tokens in make_generation_mask : ",self.token_utils.get_num_generated_tokens())
                # for idx in range(self.token_utils.get_num_generated_tokens()):
                #     # print(self.token_utils.get_input_types());
                #     # print("IDX",idx,"Current sentence type" ,current_sentence_type, "Input ids" ,self.token_utils.get_input_types());
                #     token_type = self.token_utils.get_input_types()[idx];
                #     if token_type != current_sentence_type and token_type == categories.index("Computation"):  
                #         mask[0, self.token_utils._seen_tokens - idx] = self.mask_value;
                current_sentence_type = self.token_utils.get_current_sentence_type(-1);
                input_types = self.token_utils.get_input_types()[:self.token_utils.get_num_generated_tokens()];
                computation_idx = categories.index("Computation");
                partial_results_idx = categories.index("Partial results");
                uncertainty_idx = categories.index("Uncertainty Management");
                self_checking_idx = categories.index("Self Checking");
                conversion_idx = categories.index("Conversion");
                plan_generation_idx = categories.index("Plan Generation");
        

                # Convert input types to tensor
                
                input_types_tensor = torch.tensor(input_types, device=mask.device);

                # Create boolean mask: True where token_type != current_sentence_type and token_type == "Computation"

                # masking_condition = (current_sentence_type == uncertainty_idx)  & (input_types_tensor == uncertainty_idx);
                masking_condition = (current_sentence_type != computation_idx)  & (input_types_tensor == computation_idx);
                masking_condition2 = (current_sentence_type != conversion_idx) & (input_types_tensor == conversion_idx);
                masking_condition3 = (current_sentence_type != plan_generation_idx) & (input_types_tensor == plan_generation_idx);

                amplifying_condition = (current_sentence_type != uncertainty_idx) & (input_types_tensor == uncertainty_idx);
                amplifying_condition2 = (current_sentence_type != self_checking_idx) & (input_types_tensor == self_checking_idx);
                amplifying_condition3 = (current_sentence_type != partial_results_idx) & (input_types_tensor == partial_results_idx);
                
                amplifying_condition = amplifying_condition | amplifying_condition2 | amplifying_condition3;


                masking_condition = masking_condition | masking_condition2 | masking_condition3;
                # masking_condition2 = (current_sentence_type == self_checking_idx) & (input_types_tensor == self_checking_idx);
                # masking_condition = masking_condition | masking_condition2;
                
                # amplifying_condition = (partial_results_idx == current_sentence_type) & (input_types_tensor == computation_idx);
                # amplifying_condition = (input_types_tensor == partial_results_idx) & (current_sentence_type == uncertainty_idx);

                # Get indices where condition is True

                indices_to_update = torch.nonzero(masking_condition, as_tuple=True)[0];
                indices_to_update_amplifying = torch.nonzero(amplifying_condition, as_tuple=True)[0];

                # Update mask efficiently
                
                mask[0, self.token_utils._seen_tokens - indices_to_update_amplifying] = self.amplifier_mask_value;

                mask[0, self.token_utils._seen_tokens - indices_to_update] = self.mask_value;

                # print(f"Generating custom masking {mask}");
                # print(f"Does the mask have any masked values ? {torch.any(mask==self.mask_value)}");
                        
            else:   
                raise ValueError("Not implemented : mask_pattern_type must be in [causal, semantic]");
    
    

        return mask













class MyModelOutput:
    def __init__(
        self,
        logits,
        past_key_values: Optional[Tuple] = None,
        attentions: Optional[Tuple] = None,
        all_layers_regressor_features: Optional[Tuple] = None
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.attentions = attentions
        self.all_layers_regressor_features = all_layers_regressor_features



def validate_gradients(model):
    """Check for problematic gradients"""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                return False
        else:
            print(f"No gradient for {name}")
    return True




def _token_level_generate(
    model,
    tokenizer,
    token_utils : TokenUtils,
    kv_utils :KVUtils ,
    attention_utils : AttentionUtils,
    kvsemanticAttentionMaskingUtils : KVsemanticAttentionMaskingUtils,
    max_new_tokens : int,
    predicted_token_id: int,
    mask_pattern_type: str = "semantic",
    activate_training: bool = False,
    predictive_attention: bool = False, 
    attention_score_regressor = None,
    optimizer = None,
    layer_idx_training: int = 0,
):
    """
    Simplified autoregressive generation (no compression).
    """

    new_token_counters = 0
    eos_token_id = tokenizer.eos_token_id

    # Starting points for local/global token tracking
    # global_start = len(token_utils._whole_input_ids)
    # local_start = len(token_utils._current_input_ids)

    # sentence_change_flag = True;

    # attention_matrix = 
    
    # Loop until EOS or max_new_tokens

    losses = [];
    
    while predicted_token_id != eos_token_id and new_token_counters < max_new_tokens:
        
        # 1. Prepare input for this step.
        
        new_input_ids = [predicted_token_id];

        token_utils.show_output_input_ids.append(predicted_token_id);

        
        # 2. Update attention mask Based on current sentence type
        ## Make sure that seen_sentences > 0
        
        # if token_utils.seen_sentences > 0:
        #     current_sentence_type = token_utils.get_current_sentence_type(-1);
        

        # 3. Convert tokens to tensors
        input_ids, position_ids = token_utils.set_input_ids(
            new_input_ids, return_tensors=True
        );



        attention_mask = kvsemanticAttentionMaskingUtils.make_generation_mask(mask_pattern_type = mask_pattern_type);



        if mask_pattern_type == "causal":
            attention_mask = None;
        


        prompt_len = len(token_utils.show_prompt_input_ids);


        base_val = torch.full((prompt_len,), 10, device=input_ids.device, dtype=input_ids.dtype);


        base_val = base_val.unsqueeze(0);

        # print("Base val shape", base_val);


        input_types = token_utils.get_input_types().to(input_ids.device, dtype=input_ids.dtype).unsqueeze(0)

        last_added_type_for_generated_token = torch.tensor([-2], device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)

        current_tokens_types_plus_question_type  = torch.cat([base_val, input_types,last_added_type_for_generated_token], dim=1)
        
        # print(current_tokens_types_plus_question_type)

        model_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask, ## This is what I should target.
            past_key_values=kv_utils.get_cache(),
            use_cache=True,
            return_dict=False,
            position_ids=position_ids,
            output_attentions = True,            
            current_tokens_types = current_tokens_types_plus_question_type,  
            Inference_attention_training = predictive_attention,
        );


        model_output = MyModelOutput(model_output[0],model_output[1], model_output[2], model_output[-1])
    
        ## Train our regressors

        if activate_training == True:

            with torch.enable_grad():

                attention_score_regressor.train();
                
                optimizer.zero_grad()

                # Process features
                heads_outputs = [];

                layer_features = model_output.all_layers_regressor_features[layer_idx_training];


                for features in layer_features:
                
                    heads_output = attention_score_regressor(features);

                    heads_outputs.append(heads_output);


                
                predicted_logits = torch.cat(heads_outputs, dim=0).T;  # (1, 1, L)
                # predicted_logits = torch.cat(heads_outputs, dim=0).T.squeeze(1);  # (1, 1, L)


                true_attn_probs = model_output.attentions[layer_idx_training].squeeze(2);

                
                true_attn_probs = true_attn_probs.mean(dim = 1);

                # print("true_attn_probs mean value",true_attn_probs);
                
                true_attn_probs = true_attn_probs.squeeze(dim = 1);
                

                predicted_log_probs = F.log_softmax(predicted_logits, dim=-1)
                
                true_attn_probs /= true_attn_probs.sum(dim=-1, keepdim=True) + 1e-8

                
                loss = F.kl_div(predicted_log_probs, true_attn_probs, reduction="batchmean", log_target=False);
                print("Current KL loss value",loss.item());

                losses.append(loss.item());
                
                loss.backward();



                optimizer.step()





        attention_list = [layer[0] for layer in model_output.attentions];

        attention_utils.add_new_generation(torch.stack(attention_list))

        
        # 5. Get predicted token (argmax or sampling inside InferenceUtils)

        predicted_token_id = InferenceUtils.get_predicted_token_ids(
            model_output, idx=-1
        );

        



        new_token_counters += 1;

        ## 6. Managing token-sentence types
        
        ## In the same sentence we can Change the sentence type only once (we should make sure we are in a different sentence + we have not made a change).        
    
        sentence_type_management(tokenizer.decode(predicted_token_id), token_utils);
    
        ## If the current sentence type is not "Computation", we can reduce the KV cache and input_ids to the last "Computation" token.
        ## This will ensure that we do not have too many tokens in the KV cache & input_ids.

        # reduce_kv_cache_to_sentence_type(token_utils, kv_utils, target_sentence_type=categories.index("Computation"));

        
        ### End of Sentence-token type management
    

    
    # Append last token (EOS or cutoff)
    token_utils.show_output_input_ids.append(predicted_token_id);

    # attention_utils.token_token_attention_visualization(30,31,key_start = 0, key_end = None, query_start = 0, query_end = None, generated_ids = token_utils.show_output_input_ids)
    
    decoded_prompt = tokenizer.decode(token_utils.show_prompt_input_ids, skip_special_tokens=True);

    decoded_output = tokenizer.decode(token_utils.show_output_input_ids, skip_special_tokens=True);

    ### saving regressor weights:
    if activate_training == True:
        torch.save({
        'model_state_dict': attention_score_regressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),

    }, f'attention_regressor_model_positional_layer{layer_idx_training}.pt')
        
        print(sum(losses) / len(losses));
    
        
    print("The response is : ",decoded_output);
    
    return (
        attention_utils.format_stored_attention(),
        token_utils._current_sentence_type,
        decoded_prompt,
        decoded_output,
    )





def reduce_kv_cache_to_sentence_type(token_utils, kv_utils, target_sentence_type):
    """
    Remove the last contiguous block of tokens with type == target_sentence_type
    immediately before the current token (going backward).
    """
    current_sentence_type = token_utils.get_current_sentence_type(-1)
    input_types = token_utils.get_input_types().tolist()
    if current_sentence_type == target_sentence_type:
        print(f"Current token is of type {target_sentence_type}, nothing to remove.")
        return

    # Find the end of the last contiguous block of target type before the current token
    end = len(input_types) - 1  # current token index
    start = end
    # Go backward, skipping the current token
    i = end - 1
    while i >= 0 and input_types[i] == target_sentence_type:
        start = i
        i -= 1

    # Only remove if there is a block
    if start < end:
        kv_utils.reduce_cache(start, end)
        token_utils.reduce_input_ids(start, end)
        # Remove only the block from input_types
        token_utils._input_types = input_types[:start] + input_types[end:]
        # print(f"Removed tokens of type {target_sentence_type} from indices {start}:{end}")
    # else:
        # print(f"No contiguous block of type {target_sentence_type} before the current token to remove.")












import re

def identify_token_type(token: str, categories: list, category_map: dict) -> int:
    """
    Identify the category index of a decoded token based on substring matching.

    Args:
        token (str): the decoded token string
        categories (list): ordered list of category names
        category_map (dict): mapping from keyword -> category

    Returns:
        int: index in categories if matched, else -1
    """
    # Normalize token
    normalized = token.lower().strip()
    normalized = re.sub(r"[^a-z0-9=+\-÷× ]", "", normalized)  # keep useful chars

    for keyword, category in category_map.items():
        key_norm = keyword.lower().strip()
        if key_norm in normalized:  # substring match
            return categories.index(category)  # return index from categories list

    return -1 ## Unclassified







    
def sentence_type_management(token: str, token_utils: TokenUtils):
    

        ## Whenever finding a point a default "unclassified" category should be added to sentence type.  
        new_sentence_flag = '.' in token

        if new_sentence_flag or token_utils.seen_sentences == 0:
           ## No more attention masking by default.
           ## Change previous "to be classified" tokens to current sentence type.
           if new_sentence_flag and token_utils.seen_sentences != 0:
           
                previous_sentence_type = token_utils.get_current_sentence_type(-1); ## It should something different from -1

                if previous_sentence_type == -2: ## -2 means "to be classified"
                        ## In this case we need to set the previous sentence type to "unclassified"
                    token_utils.set_current_sentence_type(-1,-1); ## To implement: must be tensor multiplication in a batch token types (parallelism)
                        ## By default: The token should get the same type of its sentence
                        ## So, all previous tokens with type "to be classified" should be set to "unclassified"
                previous_sentence_type = token_utils.get_current_sentence_type(-1);
        
                for idx in range(token_utils.get_num_generated_tokens() -1, -1, -1):
                        token_type = token_utils.get_input_types()[idx];
                        if token_type == -2:
                            ## Set input type to sentence type
                            token_utils.update_input_type(idx,previous_sentence_type);
                        if token_type != -2:
                            break;
           
           token_utils.add_current_sentence_type(-2); ## -2 means "to be classified"

        
    
        identified_token_type = identify_token_type(token,categories, category_map); ## let it be "Computation" = 3
         
        current_sentence_type = token_utils.get_current_sentence_type(-1); ## It is -2 in this case
    
        sentence_change_flag = current_sentence_type == -2 ## to be classified ; 
        
        ## Found a classified token_type
        if identified_token_type != current_sentence_type and identified_token_type!= -1 and sentence_change_flag == True: ## True
             ## Sentence type must be "Unclassified".
             token_utils.set_current_sentence_type(-1,identified_token_type); ## To implement: must be tensor multiplication in a batch token types (parallelism) ## Sentence is -1 now
             ## By default: The token should get the same type of its sentence  

             token_utils.add_input_type(identified_token_type); ## Input type is -1

             ## Make all previous -2 tokens into the new category
             for idx in range(token_utils.get_num_generated_tokens() -1, -1, -1):
                    token_type = token_utils.get_input_types()[idx];
                    if token_type == -2:
                        ## Set input type to sentence type
                        token_utils.update_input_type(idx,identified_token_type);
                    if token_type != -2:
                        break;
                
            
            
        else:
            token_utils.add_input_type(current_sentence_type); 






@torch.no_grad()



def generate(
    model: Union[Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    question:str,
    question_list:List[str],
    token_utils:TokenUtils,
    max_new_tokens:int,
    apply_standard_tokenizer_template: bool = True,
    mask_pattern_type: str = "semantic",
    activate_training :bool = False,
    predictive_attention: bool = False, 
    attention_score_regressor = None,
    optimizer = None,
    layer_idx_training: int = 0,
)-> int:
        kv_utils = KVUtils();
        attention_utils = AttentionUtils(num_layers = 32,num_heads = 32);
        ## Prefill
        predicted_token_id:int = prefill(
        model= model,
        tokenizer=tokenizer,
        question=question,
        question_list=question_list,
        kv_utils=kv_utils,
        apply_standard_tokenizer_template = apply_standard_tokenizer_template,
        token_utils=token_utils,
        
                );

        kvsemanticAttentionMaskingUtils = KVsemanticAttentionMaskingUtils(
            max_length=token_utils.max_length,
            device=token_utils.input_ids.device,
            dtype=model.dtype,
            token_utils=token_utils,
        );

        ## Auto-regressive
        attention_tensor,sentence_types,prompt, output = _token_level_generate(
        model,
        tokenizer,
        token_utils,
        kv_utils,
        attention_utils,
        kvsemanticAttentionMaskingUtils,
        max_new_tokens,
        predicted_token_id,
        mask_pattern_type = mask_pattern_type,
        activate_training= activate_training,
        predictive_attention = predictive_attention,
        attention_score_regressor = attention_score_regressor,
        optimizer = optimizer,
        layer_idx_training = layer_idx_training, 
                      );
    

        del kv_utils,attention_utils,kvsemanticAttentionMaskingUtils;
        torch.cuda.empty_cache();
        time.sleep(1);
    
        return attention_tensor,sentence_types,prompt, output;
            

        








@torch.no_grad()
def prefill(
    model: Union[Qwen2ForCausalLM],
    tokenizer: Tokenizer,
    question:str,
    question_list:List[str],
    kv_utils:KVUtils,
    token_utils:TokenUtils,
    apply_standard_tokenizer_template: bool = True,
    # attention_config:Dict,
    # attn_utils:AttentionUtils,
) -> int:
    """
    prompt will not be compressed
    """

    past_key_values:DynamicCache = kv_utils.get_cache();

    prompt = question;
    
    if apply_standard_tokenizer_template:
        # messages = [{"role": "system", "content": f"detailed thinking on"}, {"role": "user", "content": question}]
        messages = [{"role": "system", "content": f"detailed thinking on"},{"role": "user", "content": question}]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False   # <-- this makes it return a plain string
        )
    
    # 2. tokenize

    input_ids = tokenizer(
        prompt, return_tensors=None
    )['input_ids'];



    token_utils.show_prompt_input_ids.extend(input_ids);

    token_utils.set_input_ids(input_ids);

    

    # 3. model.forward()
    model_output = model(

        input_ids=torch.as_tensor(
            [input_ids], device="cuda"
        ),
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
        output_attentions = True,
        current_tokens_types = None,
    );
    


    ## check for null values in model_output.logits
    if torch.any(torch.isnan(model_output.logits)):
        raise ValueError("NaN values found in model output logits!")
    if torch.any(torch.isinf(model_output.logits)):
        raise ValueError("Inf values found in model output logits!")


    # 4. get the generated token id
    # print("PREFILL Function");

    
    predicted_token_id:int = InferenceUtils.get_predicted_token_ids(
        model_output=model_output, idx=-1
    );



    return predicted_token_id






import torch.nn.functional as F
class InferenceUtils:

    @classmethod
    def get_predicted_token_ids(
        cls,
        model_output,
        idx: int = -1,
        temperature: float = 0.4,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> int:
        """
        Sample a token from model output logits.

        Args:
            model_output: HuggingFace CausalLM output (with .logits).
            idx (int): Which timestep to sample from (default: -1 = last).
            temperature (float): Softmax temperature. Lower -> greedier.
            top_k (int): Keep only top-k logits before sampling (0 = disabled).
            top_p (float): Nucleus sampling (0.0–1.0). 1.0 = disabled.

        Returns:
            predicted_token_id (int): The sampled token id.
        """
        # [batch_size, seq_length, vocab_size]


        logits = model_output.logits
        target_logits = logits[0, idx, :] / temperature

        # --- Top-k filtering ---
        if top_k > 0:
            values, _ = torch.topk(target_logits, top_k)
            min_values = values[-1]
            target_logits = torch.where(
                target_logits < min_values,
                torch.tensor(float("-inf"), device=target_logits.device),
                target_logits,
            )

        # --- Top-p (nucleus) filtering ---
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(target_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens with cumulative prob > top_p
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_mask]
            target_logits[indices_to_remove] = float("-inf")

        # --- Sample from distribution ---
        
        probs = F.softmax(target_logits, dim=-1)
        assert torch.all(probs >= 0), "Negative probs found!"
        predicted_token_id = torch.multinomial(probs, num_samples=1).item()

        return predicted_token_id











def get_model_and_tokenizer(
    args, 
    comp_config:Config,
    bnb_config,
) -> Tuple[
    Union[Qwen2ForCausalLM],
    Tokenizer
]:
    
    model_path = args.model_path
        
    print(f"load model from `{model_path}` ...")
    
    special_token_list:List[str] = list();

    print("tokenizer_path:", args.tokenizer_path, type(args.tokenizer_path))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path);

    if args.model_type.lower() == 'qwen':
        
        if args.use_quantization:
            model = Qwen2ForCausalLM.from_pretrained(
                model_path, device_map='auto', quantization_config=bnb_config
            )
        else:
            model = Qwen2ForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )

    
    elif args.model_type.lower() == 'llama':
         if args.use_quantization:
            model = LlamaForCausalLM.from_pretrained(
                model_path, device_map='auto', quantization_config=bnb_config
            )
         else:
            model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )
        
    comp_config.convert2id(tokenizer)

    return model, tokenizer




    




def get_args():

    args = Namespace(
        model_type = "llama",
        model_path = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        max_new_tokens = 1100,
        model_short_tag="",
        model_tag="",
        use_quantization = True,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        compress_config = "/kaggle/input/qwen-config/v1.json", 
        tokenizer_path = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        
    )
    
    return args




