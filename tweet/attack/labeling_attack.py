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

query_labeling = False

current_folder = os.path.dirname(os.path.abspath(__file__))

temp = 0.5

Topic = ["Hillary Clinton", 
         "Feminist Movement", 
         "Atheism", 
         "Climate Change is a Real Concern", 
         "Legalization of Abortion",
         "Donald Trump"]
Topic_brief = ["Clinton", 
         "Feminist", 
         "Atheism", 
         "Climate", 
         "Abortion",
         "Trump"]

Stance = ["FAVOR", "AGAINST", "NONE"]

Sentiment = ["POSITIVE", "NEGATIVE", "NEITHER"]


# Default LLM (overridden by --model at the bottom of this file).
llm = None


class LLMAugPE_test(API):
    """The text API that uses open-source or API-based LLMs. This algorithm is initially proposed in the ICML 2024
    Spotlight paper, "Differentially Private Synthetic Data via Foundation Model APIs 2: Text"
    (https://arxiv.org/abs/2403.01749)"""

    def __init__(
        self,
        llm,
        dataset,    # a list of samples
        topic = None,
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
        self.dataset = dataset
        self.topic = topic

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

    def labeling(self):
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: dict
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.data.Data`
        """
        num_samples = len(self.dataset)
        messages_list = [
            self._construct_prompt(
            prompt_config=self._labeling_prompt_config,
            variables={'text': self.dataset[i], 'topic': self.topic[i] if self.topic else ""}
            )
            for i in range(num_samples)
        ]
        requests = [Request(messages=messages) for messages in messages_list]
        labels = self._llm.get_responses(requests)
        return labels


def read_csv(path, col_name = 'Tweet'):
    df = pd.read_csv(path)
    s = df[col_name].fillna('').astype(str).map(lambda x: x.strip())
    return s.tolist()


def label_dataset(dataset, save_path):
    if os.path.exists(save_path):
        return

    topic = LLMAugPE_test(llm, dataset, 
                          labeling_prompt_file=current_folder+f"/labeling_topic_prompt.json").labeling()
    
    topic_list = []
    for i in range(len(dataset)):
        final_topic = random.choice(Topic)
        t_text = topic[i]

        for topic_cate in Topic:
            if topic_cate.lower() in t_text.lower() or \
                Topic_brief[Topic.index(topic_cate)].lower() in t_text.lower():
                final_topic = topic_cate
                break
        topic_list.append(final_topic)


    stance = LLMAugPE_test(llm, dataset, topic_list, 
                           labeling_prompt_file=current_folder+f"/labeling_stance_prompt.json").labeling()
    sentimen = LLMAugPE_test(llm, dataset, 
                           labeling_prompt_file=current_folder+f"/labeling_sentiment_prompt.json").labeling()

    csv_title = ["Tweet", "Target", "Stance", "Sentiment"]
    with open(save_path, 'w', newline='', encoding = 'utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(csv_title)
        for i in range(len(dataset)):
            final_stance = "NONE"
            final_sentiment = "NEITHER"

            s_text = stance[i]
            sen_text = sentimen[i]

            for stance_cate in Stance:
                if stance_cate.lower() in s_text.lower():
                    final_stance = stance_cate
                    break
            for sentiment_cate in Sentiment:
                if sentiment_cate.lower() in sen_text.lower():
                    final_sentiment = sentiment_cate
                    break

            csv_writer.writerow([dataset[i], topic_list[i], final_stance, final_sentiment])


def majority_voting(array):
    if not array:
      return None

    counts = collections.Counter(array)
    if not counts:
        return None
    max_frequency = max(counts.values())

    most_frequent_elements = [
        element for element, frequency in counts.items()
        if frequency == max_frequency
    ]
    return random.choice(most_frequent_elements)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="CSV with a Tweet column.")
    ap.add_argument("--output", required=True, help="Output labeled CSV path.")
    ap.add_argument("--model",  default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HuggingFace model id used as the labeling LLM.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    llm = HuggingfaceLLM(max_completion_tokens=100, batch_size=8,
                         model_name_or_path=args.model, temperature=temp)
    dataset = read_csv(args.input)
    label_dataset(dataset, args.output)