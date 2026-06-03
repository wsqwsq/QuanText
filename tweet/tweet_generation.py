import json
import random
import copy
import pandas as pd
import numpy as np
import csv
import os
import sys
import collections
csv.field_size_limit(sys.maxsize)

from pe.api import API
from pe.api.util import ConstantList
from pe.logging import execution_logger
from pe.data import Data
from pe.llm import Request
from pe.constant.data import TEXT_DATA_COLUMN_NAME
from pe.constant.data import LLM_REQUEST_MESSAGES_COLUMN_NAME
from pe.constant.data import LLM_PARAMETERS_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.llm import HuggingfaceLLM
import re


current_folder = os.path.dirname(os.path.abspath(__file__))

temp = 0.5


llm = HuggingfaceLLM(max_completion_tokens=1000, 
                     batch_size = 8,
                     model_name_or_path="meta-llama/Llama-3.1-8B-Instruct", 
                     temperature=temp)


class LLMAugPE_test(API):
    """The text API that uses open-source or API-based LLMs. This algorithm is initially proposed in the ICML 2024
    Spotlight paper, "Differentially Private Synthetic Data via Foundation Model APIs 2: Text"
    (https://arxiv.org/abs/2403.01749)"""

    def __init__(
        self,
        llm,
        labeling_prompt_file="labeling_topic_prompt.json",
    ):
        """Constructor.

        :param llm: The LLM utilized for the random and variation generation
        :type llm: :py:class:`pe.llm.llm.LLM`
        :param random_api_prompt_file: The prompt file for the random API. See the explanations to
            ``variation_api_prompt_file`` for the format of the prompt file
        :type random_api_prompt_file: str
        :param variation_api_prompt_file: The prompt file for the variation API. The file is in JSON format and
            contains the following fields:

            * ``message_template``: A list of messages that will be sent to the LLM. Each message contains the
              following fields:

              * ``content``: The content of the message. The content can contain variable placeholders (e.g.,
                {variable_name}). The variable_name can be label name in the original data that will be replaced by
                the actual label value; or "sample" that will be replaced by the input text to the variation API;
                or "masked_sample" that will be replaced by the masked/blanked input text to the variation API
                when the blanking feature is enabled; or "word_count" that will be replaced by the target word
                count of the text when the word count variation feature is enabled; or other variables
                specified in the replacement rules (see below).
              * ``role``: The role of the message. The role can be "system",  "user", or "assistant".
            * ``replacement_rules``: A list of replacement rules that will be applied one by one to update the variable
              list. Each replacement rule contains the following fields:

              * ``constraints``: A dictionary of constraints that must be satisfied for the replacement rule to be
                applied. The key is the variable name and the value is the variable value.
              * ``replacements``: A dictionary of replacements that will be used to update the variable list if the
                constraints are satisfied. The key is the variable name and the value is the variable value or a
                list of variable values to choose from in a uniform random manner.
        :type variation_api_prompt_file: str
        :param min_word_count: The minimum word count for the variation API, defaults to 0
        :type min_word_count: int, optional
        :param word_count_std: The standard deviation for the word count for the variation API. If None, the word count
            variation feature is disabled and "{word_count}" variable will not be provided to the prompt. Defaults to
            None
        :type word_count_std: float, optional
        :param token_to_word_ratio: The token to word ratio for the variation API. If not None, the maximum completion
            tokens will be set to ``token_to_word_ratio`` times the target word count when the word count variation
            feature is enabled. Defaults to None
        :type token_to_word_ratio: float, optional
        :param max_completion_tokens_limit: The maximum completion tokens limit for the variation API, defaults to None
        :type max_completion_tokens_limit: int, optional
        :param blank_probabilities: The token blank probabilities for the variation API utilized at each PE iteration.
            If a single float is provided, the same blank probability will be used for all iterations. If None, the
            blanking feature is disabled and "{masked_sample}" variable will not be provided to the prompt. Defaults
            to None
        :type blank_probabilities: float or list[float], optional
        :param tokenizer_model: The tokenizer model used for blanking, defaults to "gpt-3.5-turbo"
        :type tokenizer_model: str, optional
        """
        super().__init__()
        self._llm = llm

        self.HUMAN = 'HUMAN: '
        self.GPT = 'GPT: '
        self.Token = '\n\n'
        #self.dataset = dataset

        with open(labeling_prompt_file, "r") as f:
            self._labeling_prompt_config = json.load(f)

    def random_api(self):
        return
    
    def variation_api(self):
        return

    def _construct_prompt(self, prompt_config, variables={}):
        """Applying the replacement rules to construct the final prompt messages.

        :param prompt_config: The prompt configuration
        :type prompt_config: dict
        :param variables: The inital variables to be used in the prompt messages
        :type variables: dict
        :return: The constructed prompt messages
        :rtype: list[dict]
        """
        if "replacement_rules" in prompt_config:
            for replacement_rule in prompt_config["replacement_rules"]:
                constraints = replacement_rule["constraints"]
                replacements = replacement_rule["replacements"]
                satisfied = True
                for key, value in constraints.items():
                    if key not in variables or variables[key] != value:
                        satisfied = False
                        break
                if satisfied:
                    for key, value in replacements.items():
                        if isinstance(value, list):
                            value = random.choice(value)
                        variables[key] = value
        messages = copy.deepcopy(prompt_config["message_template"])
        for message in messages:
            message["content"] = message["content"].format(**variables)
        return messages
    
    def snippet_extraction(self, sample, description):
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: dict
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.data.Data`
        """
        num_samples = len(sample)
        messages_list = [
            self._construct_prompt(prompt_config=self._labeling_prompt_config, 
                                   variables={'sample': sample[i], 
                                              'topic': description['topic'][i],
                                              'stance': description['stance'][i],
                                              'sentiment': description['sentiment'][i]})
            for i in range(num_samples)
        ]
        requests = [Request(messages=messages) for messages in messages_list]
        snippets = self._llm.get_responses(requests)
        return snippets
    
    def rewrite(self, snippet, rewrite_instruction, duplicate = 1):
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: dict
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.data.Data`
        """
        num_samples = len(snippet)
        messages_list = [
            self._construct_prompt(prompt_config=self._labeling_prompt_config, 
                                   variables={'snippet': snippet[i//duplicate],
                                              'rewrite_topic': rewrite_instruction['topic'][i//duplicate],
                                              'rewrite_stance': rewrite_instruction['stance'][i//duplicate],
                                              'rewrite_sentiment': rewrite_instruction['sentiment'][i//duplicate]})
            for i in range(num_samples * duplicate)
        ]
        requests = [Request(messages=messages) for messages in messages_list]
        text = self._llm.get_responses(requests)
        return text
    
    def _construct_samples(self, samples):
        n = len(samples)
        text = ''
        for i in range(n):
            text += f"Sample {i+1}:\n{samples[i].strip()}\n\n"
        return text
    
    def _select_sample(self, sample, judement):
        for i in range(len(sample)):
            if f'{i+1}' in judement:
                return sample[i]
        return random.choice(sample)
    
    def judge(self, description, sample, duplicate):
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: dict
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.data.Data`
        """
        num_samples = len(sample) // duplicate
        messages_list = [
            self._construct_prompt(prompt_config=self._labeling_prompt_config, 
                                   variables={'topic': description['topic'][i],
                                              'stance': description['stance'][i],
                                              'sentiment': description['sentiment'][i],
                                              'sample': self._construct_samples(sample[i*duplicate:(i+1)*duplicate])})
            for i in range(num_samples)
        ]
        requests = [Request(messages=messages) for messages in messages_list]
        judgements = self._llm.get_responses(requests)
        res = []
        for i in range(num_samples):
            res.append(self._select_sample(sample[i*duplicate:(i+1)*duplicate], judgements[i]))
        return res



def var_for_snippet_extraction(path):
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    sample = df['Tweet'].tolist()
    description = {
        'topic': df['Target'].tolist(),
        'stance': df['Stance'].tolist(),
        'sentiment': df['Sentiment'].tolist()
    }
    return sample, description

def var_for_rewrite(snippet_path, rewrite_path):
    df_snippet = pd.read_csv(snippet_path)
    df_rewrite = pd.read_csv(rewrite_path, sep=None, engine='python', encoding='utf-8-sig')

    snippet = df_snippet['Snippet'].tolist()
    rewrite_instruction = {
        'topic': df_rewrite['Target'].tolist(),
        'stance': df_rewrite['Stance'].tolist(),
        'sentiment': df_rewrite['Sentiment'].tolist()
    }
    return snippet, rewrite_instruction


def tweet_generation(ori_path, snippet_path, rewrite_path, save_path, dulipicate = 1):
    sample, description = var_for_snippet_extraction(ori_path)
    llm_snippet_extract = LLMAugPE_test(llm, labeling_prompt_file=current_folder+"/snippet_extraction_prompt.json")
    snippets = llm_snippet_extract.snippet_extraction(sample, description)

    with open(snippet_path.replace('.csv', '_whole.csv'), 'w', newline='', encoding = 'utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(['Snippet'])
        for i in range(len(snippets)):
            csv_writer.writerow([snippets[i]])

    def _extract_after_step3(text):
        m = re.search(r'(?i)Step\s*3\s*:\s*(.*)', text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    snippets = [_extract_after_step3(s) for s in snippets]

    with open(snippet_path, 'w', newline='', encoding = 'utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(['Snippet'])
        for i in range(len(snippets)):
            lines = str(snippets[i]).splitlines()
            kept = [ln for ln in lines if ln.strip() and ln.strip() != '###']
            cleaned = "\n".join(kept)
            csv_writer.writerow([cleaned])

    snippet, rewrite_instruction = var_for_rewrite(snippet_path, rewrite_path)
    llm_rewrite = LLMAugPE_test(llm, labeling_prompt_file=current_folder+"/tweet_rewrite_prompt.json")
    texts = llm_rewrite.rewrite(snippet, rewrite_instruction, duplicate = dulipicate)

    tweet_path = save_path if dulipicate == 1 else save_path.replace('.csv', f'_dup{dulipicate}.csv')

    with open(tweet_path.replace('.csv', '_whole.csv'), 'w', newline='', encoding = 'utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(['Tweet'])
        for i in range(len(texts)):
            csv_writer.writerow([texts[i]])

    def _extract_after_step2(text):
        m = re.search(r'(?i)Step\s*2\s*:\s*(.*)', text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    texts = [_extract_after_step2(s) for s in texts]

    with open(tweet_path, 'w', newline='', encoding = 'utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(['Tweet'])
        for i in range(len(texts)):
            lines = str(texts[i]).splitlines()
            kept = [ln for ln in lines if ln.strip() and ln.strip() != '###']
            cleaned = "\n".join(kept)
            # remove trailing '###' if present (with optional surrounding whitespace)
            cleaned = re.sub(r'\s*#{3}\s*$', '', cleaned).rstrip()
            texts[i] = cleaned
            csv_writer.writerow([texts[i]])

    if dulipicate > 1:
        llm_judge = LLMAugPE_test(llm, labeling_prompt_file=current_folder+"/tweet_judge_prompt.json")
        judged_texts = llm_judge.judge(rewrite_instruction, texts, dulipicate)
        judged_tweet_path = save_path
        with open(judged_tweet_path, 'w', newline='', encoding = 'utf-8') as wf:
            csv_writer = csv.writer(wf)
            csv_writer.writerow(['Tweet'])
            for i in range(len(judged_texts)):
                csv_writer.writerow([judged_texts[i]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ori",           default=current_folder + '/data/tweet.csv',
                    help="Private tweet CSV (with Tweet/Target/Stance/Sentiment columns).")
    ap.add_argument("--snippets",      default=current_folder + '/res/snippets.csv',
                    help="Output path for extracted snippets CSV.")
    ap.add_argument("--rewrite-attrs", default=current_folder + '/res/new_attributes.csv',
                    help="Generated property arrays (Target/Stance/Sentiment) from quantization.py.")
    ap.add_argument("--save",          default=current_folder + '/res/rewritten_tweets.csv',
                    help="Output path for rewritten tweets CSV.")
    ap.add_argument("--duplicate", type=int, default=3,
                    help="Number of candidate rewrites per tweet; >1 enables LLM judge.")
    args = ap.parse_args()

    for p in [args.snippets, args.save]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    tweet_generation(args.ori, args.snippets, args.rewrite_attrs, args.save,
                     dulipicate=args.duplicate)