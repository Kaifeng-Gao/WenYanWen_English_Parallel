

# **WenYanWen\_English\_Parallel**

This repo is the code for generating and preprocessing the [WenYanWen\_English\_Parallel](https://huggingface.co/datasets/KaifengGGG/WenYanWen_English_Parrallel) dataset.

## **Dataset Summary**

The WenYanWen\_English\_Parallel dataset is a multilingual parallel corpus in Classical Chinese (Wenyanwen), modern Chinese, and English. The Classical Chinese and modern Chinese parts are sourced from the NiuTrans/Classical-Modern dataset, while the corresponding English translations are generated using Gemini Pro.

## **Data Fields**
- `info`: A string representing the title or source information of the text.
- `classical`: Classical Chinese (Wenyanwen) text corresponding to the modern text.
- `modern`: A string containing the translation of the original Classical Chinese text into modern Chinese.
- `english`: English translation of the Chinese text.
- `text`: instruction/answer pair in string format
- `messages`: instruction/answer pair in conversation format:
  - `content`: String representing the content of a message.
  - `role`: String representing the role associated with the message (e.g., system, assistent, user).
 
Here is an example for a dataset entry:

| Field      | Type           | Description                                                                              |
|------------|----------------|------------------------------------------------------------------------------------------|
| info       | string         | 《辽史·列传·卷二十八》                                                                    |
| modern     | string         | 乾统三年，徙封为秦国公。                                                                   |
| classical  | string         | 乾统三年，徙封秦国。                                                                       |
| english    | string         | In the third year of the Qingtong Era, he was re-enfeoffed as Prince of the Qin State.  |
| text       | string         | `<s>`[INST] 将以下现代汉语文本改写为文言文: 乾统三年，徙封为秦国公。 [/INST] 乾统三年，徙封秦国。`</s>`  |
| messages   | list           | [{"content": "将以下现代汉语文本改写为文言文: 乾统三年，徙封为秦国公。", "role": "user"}, {"content": "乾统三年，徙封秦国。", "role": "assistant"}] | 

## **Dataset Structure**

The dataset consists of four subsets: `default`, `instruct`, `instruct-augment`, and `instruct-large`.

- `default` is a parallel translation dataset.
- `instruct` serves as an instruction-tuning dataset and consists of prompt/answer pairs created from a 10,000-sample subset of the `default` dataset.
- `instruct-augment` is similar to `instruct`, with the distinction being that the prompt/answer pairs have been augmented by Gemini-Pro.
- `instruct-large` is an expanded version of `instruct` that includes all samples from the `default` dataset.

### **Default**
| `info`   | `modern` | `classical` | `english` |
|----------|-------------|-----------|-----------|
| string   | string   | string      | string    |

| Split | Examples  |
|-------|-----------|
| Train | 972,467   |

### **Instruct**
| `info`   | `modern` | `classical` | `english` | `text` | `messages`             |
|----------|----------|-------------|-----------|--------|------------------------|
| string   | string   | string      | string    | string | list of {`content`: string, `role`: string}|

| Split | Examples  |
|-------|-----------|
| Train | 9,000   |
| Test  | 1,000   |

### **Instruct-Augmented**
| `info`   | `modern` | `classical` | `english` | `text` | `messages`             |
|----------|----------|-------------|-----------|--------|------------------------|
| string   | string   | string      | string    | string | list of {`content`: string, `role`: string}|

| Split | Examples  |
|-------|-----------|
| Train | 9,000   |
| Test  | 1,000    |
   
### **Instruct-Large**
| `info`   | `modern` | `classical` | `english` | `text` | `messages`             |
|----------|----------|-------------|-----------|--------|------------------------|
| string   | string   | string      | string    | string | list of {`content`: string, `role`: string}|

| Split | Examples  |
|-------|-----------|
| Train | 875,214   |
| Test  | 97,246    |

## **Supported Tasks and Leaderboard**

This dataset can be used for various multilingual and translation tasks, including but not limited to:

1. Neural Machine Translation (Classical Chinese to Modern Chinese)
2. Neural Machine Translation (Modern Chinese to English)
3. Neural Machine Translation (Classical Chinese to English)
4. Multilingual Text-to-Text Transfer

There is currently no official leaderboard for this dataset.

## **License**

Please refer to the license of the [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern) dataset and the terms of use of Gemini Pro for more information regarding the dataset license.

## **Citation Information**

If you use this dataset in your research, please cite the original sources:

1. [NiuTrans/Classical-Modern](https://github.com/NiuTrans/Classical-Modern)
2. [Gemini Pro](https://arxiv.org/abs/2403.05530)

## **Potential Bias**

Since the English translations are generated using Gemini Pro, there might be inconsistencies or errors in the translations, which may introduce bias into the dataset. Additionally, the choice of Classical Chinese texts and their modern Chinese translations may also introduce bias. Finally, the use of a single translation tool for the English translations may result in limited linguistic diversity.

## **Potential Social Impact**

This dataset can be used for various multilingual and translation tasks, which can have a positive impact on facilitating cross-cultural communication and understanding. However, it is important to be aware of the potential biases in the dataset and to use the dataset responsibly. Additionally, as with any dataset, it is important to consider the ethical implications of using this dataset, including issues related to data privacy, consent, and representation.

