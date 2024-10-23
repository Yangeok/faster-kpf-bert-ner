import os
from typing import List

import kss
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    PreTrainedTokenizer,
    logging,
)
from transformers.modeling_outputs import TokenClassifierOutput

from faster_kpf_ber_ner import label
from faster_kpf_ber_ner.utils import MeasureTime, WordList, chunk_text

logging.set_verbosity_error()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("KPF/KPF-bert-ner")
model = BertForTokenClassification.from_pretrained("KPF/KPF-bert-ner")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@MeasureTime()
def ner_predict(text: str):
    """
    predict(text, tokenizer, model) : 추론 함수.
    - 문장을 입력받아 model input 에 맞게 변환.
    - model에 입력 후 추론 클래스들을 output으로 반환.
    input : text (text를 tokenize한 결과물을 넣음)
    output : word, label, desc (dict형태로 토큰에 대한 결과 반환)
    """

    text = text.replace("\n", "")
    model.to(device)

    decoding_ner_sentence = ""
    word_list: List[WordList] = []

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

        for token_id, predicted_label in tqdm(
            np.column_stack([chunk, predicted_labels]),
            desc="step2",
            leave=False,
        ):
            token_id: int
            predicted_label: str

            token = (
                tokenizer.convert_ids_to_tokens([token_id])[0]
                .replace("#", "")
                .replace("-", " ")
                .strip()
            )

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


@MeasureTime()
def ner_predict_old(text):
    text = text.replace("\n", "")
    sents = kss.split_sentences(text)
    decoding_ner_sentence = ""
    word_list = list()
    pred_str = list()

    # text to model input
    for idx, sent in enumerate(sents):

        sent = sent.replace(" ", "-")
        test_tokenized = tokenizer(sent, return_tensors="pt")

        test_input_ids = test_tokenized["input_ids"].to(device)
        test_attention_mask = test_tokenized["attention_mask"].to(device)
        test_token_type_ids = test_tokenized["token_type_ids"].to(device)

        inputs = {
            "input_ids": test_input_ids,
            "attention_mask": test_attention_mask,
            "token_type_ids": test_token_type_ids,
        }

        if inputs["input_ids"].size()[1] > 512:
            cnt = int(inputs["input_ids"].size()[1])

            inp_np = inputs["input_ids"].cpu().numpy()
            att_np = inputs["attention_mask"].cpu().numpy()
            tok_np = inputs["token_type_ids"].cpu().numpy()

            for i in range(cnt):
                slice_inp = inp_np[0][(i * 512) : ((i + 1) * 512)]
                slice_att = att_np[0][(i * 512) : ((i + 1) * 512)]
                slice_tok = tok_np[0][(i * 512) : ((i + 1) * 512)]

                slice_inp = slice_inp.reshape(1, len(slice_inp))
                slice_att = slice_att.reshape(1, len(slice_att))
                slice_tok = slice_tok.reshape(1, len(slice_tok))

                slice_inp = torch.tensor(slice_inp)
                slice_att = torch.tensor(slice_att)
                slice_tok = torch.tensor(slice_tok)

                slice_inp = torch.tensor(slice_inp).to(device)
                slice_att = torch.tensor(slice_att).to(device)
                slice_tok = torch.tensor(slice_tok).to(device)

                slice_inputs = {
                    "input_ids": slice_inp,
                    "attention_mask": slice_att,
                    "token_type_ids": slice_tok,
                }

                # predict
                outputs = model(**slice_inputs)

                token_predictions = outputs[0].argmax(dim=2)
                token_prediction_list = token_predictions.squeeze(0).tolist()

                pred = [label.id2label[l] for l in token_prediction_list]

                pred_str = np.concatenate((pred_str, pred))
        else:
            # predict
            outputs = model(**inputs)

            token_predictions = outputs[0].argmax(dim=2)
            token_prediction_list = token_predictions.squeeze(0).tolist()

            pred_str = [label.id2label[l] for l in token_prediction_list]
        tt_tokenized = tokenizer(sent).encodings[0].tokens

        # decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False
        _word = ""
        # word_list = list()

        # model output to text
        for i, (token, pred) in enumerate(zip(tt_tokenized, pred_str)):
            if i == 0 or i == len(pred_str) - 1:
                continue
            token = token.replace("#", "").replace("-", " ")

            if token == "":
                continue

            if "B-" in pred:
                if is_prev_entity is True:
                    decoding_ner_sentence += ":" + prev_entity_tag + ">"
                    word_list.append(
                        {"word": _word, "label": prev_entity_tag, "desc": "1"}
                    )
                    _word = ""

                if token[0] == " ":
                    token = list(token)
                    token[0] = " <"
                    token = "".join(token)
                    decoding_ner_sentence += token
                    _word += token
                else:
                    decoding_ner_sentence += "<" + token
                    _word += token
                is_prev_entity = True
                prev_entity_tag = pred[2:]
                is_there_B_before_I = True

            elif "I-" in pred:
                decoding_ner_sentence += token
                _word += token

                if is_there_B_before_I is True:
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ":" + prev_entity_tag + ">" + token
                    is_prev_entity = False
                    is_there_B_before_I = False
                    word_list.append(
                        {
                            "word": _word,
                            "label": prev_entity_tag,
                            "desc": label.ner_code[prev_entity_tag],
                        }
                    )
                    _word = ""
                else:
                    decoding_ner_sentence += token

    # print("OUTPUT")
    # print("sentence : ", decoding_ner_sentence)
    # print("result : ", word_list)
    return word_list


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
