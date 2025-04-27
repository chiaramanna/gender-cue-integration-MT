import logging
import numpy as np
import torch
from transformers import AutoTokenizer, MBart50TokenizerFast, AutoModelForSeq2SeqLM
import inseq
import argparse
import os
import re
import json

def get_subword_indices(tokens, word):
    """
    Get the corresponding token indices (subword units) for a given word.

    Args:
        tokens (list of str): List of tokens for each (tokenized) sentence.
        word (str): The target word to search for in the tokens.

    Returns:
        list: Indices of the tokens that correspond to the given word.
    """    
 
    word = word.lower()
    tokens = [token.lower() for token in tokens]
    word_parts = word.split()

    indices = []
    i = 0
    for part in word_parts:
        part_indices = []
        while i < len(tokens):
            token = tokens[i]

            if part in ['she', 'he', 'her', 'him', 'his'] and i + 1 < len(tokens):  
                    # we check if the next token is valid and - if so - we make sure that we do not append indices of partial matches ( --> _she vs _she riff)
                    # we do this by making sure that when the pronoun is matched, the next token is either a punctuation mark or a new work (as indicated by "▁")
                next_token = tokens[i + 1]
                if not (next_token.startswith('▁') or next_token in ['.', ',', '!', '?', ':', ';', '-', '(', ')', '"']):
                    i += 1  
                    continue  

            if token.startswith('▁'):
              subword = token[1:]  
            if part.startswith(subword):
                part_indices.append(i)
                part = part[len(subword):]

            elif part.startswith(token):
                part_indices.append(i)
                part = part[len(token):]

            i += 1
            if not part:
              break

        indices.extend(part_indices)

    logging.info(f"Final subword indices for word '{word}': {indices}")
    return sorted(indices)


def find_target_word_index(target_word, sentence):
    """
    Find the index of the target word in a sentence.

    Args:
        target_word (str): The target word to find.
        sentence (str): The sentence to search in.

    Returns:
        int: The (word-)index of the target word in the sentence (if found).
    """

    input_words = sentence.split()
    target_word = target_word.lower()

    for i, word in enumerate(input_words):
        if word.lower().strip(',.') == target_word:
            return i


def get_target_output_token_indices(alignment_file, line_num, target_word_index, translated_sentence, translated_tokens):
    """
    Get the indices of translated tokens that correspond to the target word in the input.

    Args:
        alignment_file (str): Path to the alignment file.
        line_num (int): The line number of the current sentence.
        target_word_index (int): The index of the target word in the input sentence.
        translated_sentence (str): The translated sentence.
        translated_tokens (list of str): List of tokenized words from the translated sentence.

    Returns:
        list: Sorted indices of translated tokens that correspond to the target word in the input.
    """
    
    with open(alignment_file, 'r') as file:
        alignments = file.readlines()
    alignment = alignments[line_num - 1].strip()
    
    target_output_index = {int(tgt_idx) for src_idx, tgt_idx in (pair.split('-') for pair in alignment.split()) if int(src_idx) == target_word_index}
    if not target_output_index:
        raise ValueError("Target word index not found in alignment file.")

    translated_words = translated_sentence.split()
    target_words = [translated_words[tgt_idx] for tgt_idx in target_output_index]

    target_word_subword_indices = []
    for word in target_words:
        target_word_subword_indices.extend(get_subword_indices(translated_tokens, word.strip(',.')))

    target_indices_set = set(target_word_subword_indices)

    return sorted(list(target_indices_set))


def load_translation_pipeline(model_name, src_lang=None, tgt_lang=None):

    """
    Load the model and tokenizer based on the model name, with optional quantization.
    """

    models = {
        "nllb": "facebook/nllb-200-distilled-600M",
        "opus": "Helsinki-NLP/opus-mt-en-it",
        "mbart": "facebook/mbart-large-50-many-to-many-mmt"
    }

    tokenizer_kwargs = {}
    generation_args = {}


    if model_name == 'nllb':
        tokenizer_kwargs = {"src_lang": src_lang, "tgt_lang": tgt_lang}
        model = inseq.load_model(models[model_name], "attention", tokenizer_kwargs=tokenizer_kwargs)
        generation_args = {"forced_bos_token_id": model.tokenizer.convert_tokens_to_ids(tgt_lang)}
        tokenizer = AutoTokenizer.from_pretrained(models[model_name], src_lang=src_lang, tgt_lang=tgt_lang)
        layers, heads = 12, 16
    elif model_name == 'mbart':
        tokenizer_kwargs = {"src_lang": src_lang, "tgt_lang": tgt_lang}
        model = inseq.load_model(models[model_name], "attention", tokenizer_kwargs=tokenizer_kwargs)
        generation_args = {"forced_bos_token_id": model.tokenizer.lang_code_to_id[tgt_lang]}
        tokenizer = MBart50TokenizerFast.from_pretrained(models[model_name], src_lang=src_lang, tgt_lang=tgt_lang)
        layers, heads = 12, 16
    else:
        model = inseq.load_model(models[model_name], "attention")
        tokenizer = AutoTokenizer.from_pretrained(models[model_name])
        layers, heads = 6, 8

    return model, tokenizer, generation_args, layers, heads


def store_attribution_scores_full(attributions, mode, target_token_indices, cue_token_indices):
    """
    Compute attribution scores for the encoder and cross layers.

    Args:
        attributions (tensor): Attribution scores tensor (from Inseq).
        mode (str): Mode ('encoder' or 'cross') to indicate the layer type.
        target_token_indices (list): Indices of target tokens.
        cue_token_indices (list): Indices of cue tokens.

    Returns:
        tuple: Cue and full encoder/cross cores.
    """
    layers = attributions.size(-2) 
    heads = attributions.size(-1) 
    num_tokens = attributions.size(0)

    scores = np.zeros((layers, heads))
    scores_full = np.zeros((layers, heads, num_tokens))

    target_token_tensors = torch.tensor(target_token_indices, dtype=torch.long)
    cue_token_tensors = torch.tensor(cue_token_indices, dtype=torch.long)
    all_token_tensors = torch.tensor([i for i in range(attributions.size(0))], dtype=torch.long)

    if mode == 'encoder':
        for layer in range(layers):
            for head in range(heads):
                layer_head_score = attributions[:, :, layer, head]
                target_scores = layer_head_score[cue_token_tensors, target_token_tensors].mean()
                scores[layer, head] = target_scores.item()
                scores_full[layer, head] = layer_head_score[all_token_tensors[:, None], target_token_tensors].mean(dim=1).cpu().numpy()

    elif mode == 'cross':
        for layer in range(layers):
            for head in range(heads):
                layer_head_score = attributions[:, :, layer, head] 
                cue_scores = layer_head_score[cue_token_tensors, target_token_tensors].mean()
                scores[layer, head] = cue_scores.item()
                scores_full[layer, head] = layer_head_score[all_token_tensors[:, None], target_token_tensors].mean(dim=1).cpu().numpy()

    return scores, scores_full

def attribute_and_save_json(input_file, translation_file, model_name, alignment_file, suffix='', src_lang=None, tgt_lang=None):
    """
    For each input and translated sentence, compute and save attention scores to JSON.
    """

    model, _, generation_args, layers, heads = load_translation_pipeline(model_name, src_lang, tgt_lang)

    output_dir = f'./data/attribution_scores/attention/{model_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_output_path = os.path.join(output_dir, f'attribution_scores_{suffix}.json')

    results = []
    with open(input_file, 'r') as infile, open(translation_file, 'r') as tfile:
        for line_num, (line, tline) in enumerate(zip(infile, tfile)):
            cols = line.strip().split('\t')
            input_sentence, target_word, gender = cols[2], cols[3], cols[0]

            tcols = tline.strip().split(" ||| ")
            translated_sentence = tcols[1].strip()
            
            logging.info(f"Attributing sentence {line_num}: {input_sentence}")
            out = model.attribute(input_sentence, generated_texts=translated_sentence, generation_args=generation_args)

            input_tokens = [token_with_id.token for token_with_id in out.sequence_attributions[0].source]
            target_tokens = [token_with_id.token for token_with_id in out.sequence_attributions[0].target][1:]
            
            num_input_tokens = len(input_tokens)

            sentence_data = {
                'sentence_num': line_num,
                'input_sentence': input_sentence,
                'translated_sentence': translated_sentence,
                'input_tokens': input_tokens,
                'target_tokens': target_tokens,
                'attributions': {}
            }
            
            # initialize "dummy" attributions scores for following conditions
            sentence_data['attributions'] = {
                'cue_word': "",
                'cue_input_indices': [],  
                'target_input_indices': [],
                'encoder_scores':  np.ones((layers, heads)).tolist(),
                'encoder_full_cue_scores': np.ones((layers, heads, num_input_tokens)).tolist(),
                'cross_scores': np.ones((layers, heads)).tolist(),
                'cross_full_scores': np.ones((layers, heads, num_input_tokens)).tolist()
            }

            #CONDITIONS TO SKIP - APPEND DUMMY SCORES (ones)
            #if target word is more than one word, skip
            if len(target_word.split()) > 1:
                logging.info(f"Skipping sentence {line_num}: Target word '{target_word}' is more than one word.")
                results.append(sentence_data)
                continue
                
            #if gender is neutral, skip
            if gender == 'neutral':
                logging.info(f"Skipping sentence {line_num}: Gender is {gender}.")
                results.append(sentence_data)
                continue

            #identify cue word 
            cue_words = {'female': ['she', 'her'], 'male': ['he', 'him', 'his']}
            clean_sentence = re.sub(r'[^\w\s]', '', input_sentence.lower())
            cue_word = next((word for word in cue_words[gender.lower()] if word in clean_sentence.split()), None)

            if cue_word:
                cue_input_indices = get_subword_indices(input_tokens, cue_word)
                logging.info(f'Cue index is {cue_input_indices}')
                    
                try:
                    target_word_index = find_target_word_index(target_word, input_sentence)
                    target_output_indices = get_target_output_token_indices(alignment_file, line_num, target_word_index, translated_sentence, target_tokens)
                    
                    if not target_output_indices:
                        logging.info(f"Skipping sentence {line_num}: No target output indices found in alignment file.")
                        results.append(sentence_data)
                        continue 

                except (ValueError, IndexError) as e:
                    logging.info(f'Skipping sentence {line_num}: {e} - No target output indices found in alignment file.')
                    results.append(sentence_data)
                    continue
                    
                target_input_indices = get_subword_indices(input_tokens, target_word)


                encoder_scores, encoder_full_scores = store_attribution_scores_full(
                    out[0].sequence_scores["encoder_self_attentions"], 'encoder',
                    input_tokens, cue_word, target_word, target_input_indices, cue_input_indices
                    )
                
                cross_scores, cross_full_scores = store_attribution_scores_full(
                    out[0].source_attributions, 'cross',
                    input_tokens, cue_word, target_word, target_output_indices, cue_input_indices
                    )

                # store the attribution data in the dictionary
                sentence_data['attributions'] = {
                    'cue_word': cue_word,
                    'cue_input_indices': cue_input_indices,
                    'target_input_indices': target_input_indices,
                    'encoder_scores': encoder_scores.tolist(),
                    'encoder_full_cue_scores': encoder_full_scores.tolist(),
                    'cross_scores': cross_scores.tolist(),
                    'cross_full_scores': cross_full_scores.tolist()
                }


            #append the result for each line to the results list (list of dictionary of ...)
            results.append(sentence_data)

    #save results to json
    with open(json_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Script to compute and save attention scores")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument("--translation_file", type=str, required=True, help="Path to the translation file (e.g., WinoMT format).")    
    parser.add_argument('--model_name', type=str, required=True, choices=['nllb', 'mbar', 'opus'], 
                        help='Pre-trained model name. Choose from: nllb, mbar, or opus')
    parser.add_argument('--src_lang', type=str, required=True, help='Source language code (model specific)')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language code (model specific)')
    parser.add_argument('--alignment_file', type=str, required=True, help='Path to the alignment file')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for the saved files')

    args = parser.parse_args()

    log_filename = f'./logs/translation_log_{args.model_name}_{args.suffix}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    attribute_and_save_json(
        input_file=args.input_file,
        translation_file=args.translation_file,
        model_name=args.model_name,
        alignment_file=args.alignment_file,
        suffix=args.suffix,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )

if __name__ == "__main__":
    main()
