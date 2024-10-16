
###########################################################################################################
"""
ner_module.py : 추론 관련 코드.
- 실행하는 폴더에 predict.py, label.py, kpf-bert, kpf-bert-ner 폴더가 있어야함.
input : text (sentence)
output : word, label, desc (predict results by kpf-bert-ner)
"""
###########################################################################################################

import os
import time
from contextlib import ContextDecorator
from typing import Generator

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    PreTrainedTokenizerBase,
    logging,
)
from transformers.modeling_outputs import TokenClassifierOutput

import label

logging.set_verbosity_error()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#############################################################################################################
"""
    predict(text, tokenizer, model) : 추론 함수.
    - 문장을 입력받아 model input 에 맞게 변환.
    - model에 입력 후 추론 클래스들을 output으로 반환.
    input : text (text를 tokenize한 결과물을 넣음)
    output : word, label, desc (dict형태로 토큰에 대한 결과 반환)
"""
###############################################################################################################

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("KPF/KPF-bert-ner")
model = BertForTokenClassification.from_pretrained("KPF/KPF-bert-ner")

class MeasureTime(ContextDecorator):
    def __enter__(self):
        self.start_time = time.time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        print(f"elapsed time: {elapsed_time:.4f}초")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunk_text(text: str, tokenizer: PreTrainedTokenizerBase, chunk_size=512) -> Generator[list[int], None, None]:
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("KPF/KPF-bert-ner")
    model = BertForTokenClassification.from_pretrained("KPF/KPF-bert-ner")

    tokens = (
        tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        .squeeze()
        .tolist()
    )
    for i in range(0, len(tokens), chunk_size):
        yield tokens[i : i + chunk_size]


@MeasureTime()
def ner_predict(text: str):
    text = text.replace("\n", "")
    model.to(device)

    decoding_ner_sentence = ""
    word_list = []

    for chunk in tqdm(chunk_text(text, tokenizer), desc="step1"):
        chunk_tensor = torch.tensor(chunk).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(chunk_tensor).to(device)

        inputs = {
            "input_ids": chunk_tensor,
            "attention_mask": attention_mask,
        }

        # 모델 추론
        outputs: TokenClassifierOutput = model(**inputs)
        token_predictions = outputs.logits.argmax(dim=2)
        predicted_labels = [
            label.id2label[pred] for pred in token_predictions.squeeze(0).tolist()
        ]

        is_entity_active = False
        current_entity = ""
        entity_tokens = ""

        for token_id, predicted_label in tqdm(zip(chunk, predicted_labels), desc="step2", leave=False):
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            token = token.replace("#", "").replace("-", " ").strip()

            if not token:
                continue

            if predicted_label.startswith("B-"):
                if is_entity_active:
                    word_list.append(
                        {
                            "word": entity_tokens,
                            "label": current_entity,
                            "desc": label.ner_code.get(current_entity, "Unknown"),
                        }
                    )
                decoding_ner_sentence += f"<{token}:{predicted_label[2:]}>"
                entity_tokens = token
                current_entity = predicted_label[2:]
                is_entity_active = True

            elif predicted_label.startswith("I-") and is_entity_active:
                decoding_ner_sentence += token
                entity_tokens += token

            else:
                if is_entity_active:
                    word_list.append(
                        {
                            "word": entity_tokens,
                            "label": current_entity,
                            "desc": label.ner_code.get(current_entity, "Unknown"),
                        }
                    )
                    is_entity_active = False
                decoding_ner_sentence += token

    return word_list


##################################################################################################################
"""
    추론 함수를 실행하는 메인 함수.
"""
####################################################################################################################

if __name__ == "__main__":
    
    text = """
    더불어민주당 이재명 대표가 이른바 '성남FC 후원금 의혹' 사건과 관련해 오는 10일 검찰에 출석해 조사를 받는다.

민주당 안호영 수석대변인은 6일 국회 브리핑을 통해 "이 대표가 10일 오전 10시 30분에 수원지검 성남지청에 출석하는 일정이 합의됐다"고 밝혔다.

안 수석대변인은 "검찰과 변호인단이 출석 날짜를 조율했고, 그 날짜가 적당하다고 판단한 것"이라고 설명했다.

공개적으로 출석하느냐는 질문에는 "이 대표는 당당히 출석해서 입장을 말씀하신다고 했다"며 "구체적으로 어떤 사람과 갈지, 어떻게 할지는 지켜봐야 한다"고 말했다.

앞서 검찰은 이 사건과 관련해 이 대표에게 지난해 12월 28일 소환을 통보했으나, 이 대표는 미리 잡아 둔 일정이 있다며 출석을 거부했다.

다만 이 대표는 "가능한 날짜와 조사 방식에 대해 변호인을 통해 협의해서 결정하겠다"며 조사에 응하겠다는 뜻을 밝혔고, 이후 검찰이 다시 요청한 10∼12일 중에서 출석 일자를 조율해 왔다.

성남FC 후원금 의혹 사건은 이 대표가 성남시장 재직 시절 성남FC 구단주로 있으면서 2016∼2018년 네이버·두산건설 등 기업들로부터 160억여원의 후원금을 유치하고, 이들 기업은 건축 인허가나 토지 용도 변경 등 편의를 받았다는 내용이다.

이 대표는 2018년 당시 바른미래당 등으로부터 이 의혹으로 고발당했다. 현재 제3자 뇌물공여 혐의를 받는 피의자 신분이다.

이 대표가 취임 이후 검찰의 소환조사에 응하는 것은 처음이다.

검찰은 앞서 지난 8월에도 대선 과정에서 허위 사실을 공표했다는 혐의로 이 대표에게 소환을 통보했으나, 당시 이 대표는 출석을 거부하고 서면 답변서만 제출한 바 있다.
    """

    list = ner_predict(text)
    print(list)